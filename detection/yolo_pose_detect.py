import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
from datetime import datetime
from collections import deque, defaultdict
import math
from typing import Dict


class SmartSurveillanceDetector:
    def __init__(self, yolo_model_path="yolov8n.pt"):
        # Model and parameter setup
        self.yolo_model = YOLO(yolo_model_path)
        self.device = "cuda" if hasattr(self.yolo_model, 'device') and self.yolo_model.device.type == 'cuda' else "cpu"
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.weapon_classes = {'knife': [74], 'baseball bat': [35], 'gun': []}
        self.aggression_threshold = 0.6
        self.crowd_threshold = 3
        self.fighting_threshold = 0.5
        self.alert_cooldown = 20
        self.last_alert_time = {}
        self.pose_history = defaultdict(lambda: deque(maxlen=5))
        self.prev_centers = {}
        self.aggression_score_buffer = deque(maxlen=15)


    def _landmark_to_np(self, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])


    def analyze_fighting_pose(self, pose_history):
        # Analyze pose history for fighting/aggression events
        r = {'punch': False, 'push': False, 'fall': False, 'stampede': False, 'fighting_score': 0.0}
        if len(pose_history) < 2:
            return r
        times, lms = zip(*pose_history)
        dt = (times[-1] - times[-2]).total_seconds() or 1/30
        try:
            lw = self._landmark_to_np(lms[-1][self.mp_pose.PoseLandmark.LEFT_WRIST.value])
            rw = self._landmark_to_np(lms[-1][self.mp_pose.PoseLandmark.RIGHT_WRIST.value])
            lw0 = self._landmark_to_np(lms[-2][self.mp_pose.PoseLandmark.LEFT_WRIST.value])
            rw0 = self._landmark_to_np(lms[-2][self.mp_pose.PoseLandmark.RIGHT_WRIST.value])
            ls = self._landmark_to_np(lms[-1][self.mp_pose.PoseLandmark.LEFT_SHOULDER.value])
            rs = self._landmark_to_np(lms[-1][self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
            nose = self._landmark_to_np(lms[-1][self.mp_pose.PoseLandmark.NOSE.value])
            nose0 = self._landmark_to_np(lms[-2][self.mp_pose.PoseLandmark.NOSE.value])
            torso = (ls + rs) / 2
            torso0 = (self._landmark_to_np(lms[-2][self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]) + self._landmark_to_np(lms[-2][self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])) / 2
        except Exception:
            return r
        lwv = (lw - lw0) / dt
        rwv = (rw - rw0) / dt
        torso_v = (torso - torso0) / dt
        nose_v = (nose - nose0) / dt
        if np.linalg.norm(lwv) > 0.8 or np.linalg.norm(rwv) > 0.8:
            r['punch'] = True
            r['fighting_score'] += 0.6
        if lwv[1] < -0.6 and rwv[1] < -0.6:
            r['push'] = True
            r['fighting_score'] += 0.5
        if nose_v[1] > 0.6 or torso_v[1] > 0.6:
            r['fall'] = True
            r['fighting_score'] += 0.5
        if len(pose_history) >= 3:
            tvs = []
            for i in range(-3, -1):
                lm1, lm2 = lms[i], lms[i+1]
                t1, t2 = times[i], times[i+1]
                dt_ = (t2 - t1).total_seconds() or 1/30
                t1_torso = (self._landmark_to_np(lm1[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]) + self._landmark_to_np(lm1[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])) / 2
                t2_torso = (self._landmark_to_np(lm2[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]) + self._landmark_to_np(lm2[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value])) / 2
                tvs.append((t2_torso - t1_torso) / dt_)
            if np.mean([np.linalg.norm(v) for v in tvs]) > 0.8:
                r['stampede'] = True
                r['fighting_score'] += 0.4
        arm_move = 0.5
        if abs(lw[1] - ls[1]) > 0.1 and abs(rw[1] - rs[1]) > 0.1:
            if np.linalg.norm(lwv) > arm_move or np.linalg.norm(rwv) > arm_move:
                r['fighting_score'] += 0.3
        r['fighting_score'] = min(r['fighting_score'], 1.0)
        return r


    def detect_weapons(self, results):
        # Detect weapons from YOLO results
        weapons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    for weapon, ids in self.weapon_classes.items():
                        if class_id in ids and conf > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            weapons.append({'type': weapon, 'confidence': conf, 'bbox': (int(x1), int(y1), int(x2), int(y2))})
        return weapons


    def detect_people(self, results):
        # Detect people from YOLO results
        people = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if class_id == 0 and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        people.append({'confidence': conf, 'bbox': (int(x1), int(y1), int(x2), int(y2)), 'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))})
        return people


    def calculate_pose_aggression_score(self, landmarks):
        # Calculate aggression score based on pose landmarks
        if not landmarks:
            return 0.0
        try:
            ls = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            le = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            re = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            lw = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            rw = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            score = 0.0
            if lw.y < ls.y and rw.y < rs.y:
                score += 0.3
            if abs(lw.x - ls.x) > abs(le.x - ls.x) or abs(rw.x - rs.x) > abs(re.x - rs.x):
                score += 0.4
            scy = (ls.y + rs.y) / 2
            if nose.y > scy + 0.1:
                score += 0.3
            return min(score, 1.0)
        except:
            return 0.0


    def detect_crowd_formation(self, people):
        # Detect if people are forming a crowd
        if len(people) < self.crowd_threshold:
            return False
        centers = [p['center'] for p in people]
        total, count = 0, 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = math.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
                total += d
                count += 1
        if count > 0:
            return (total / count) < 200
        return False


    def should_send_alert(self, camera_id, event_type):
        # Cooldown logic for alerts
        import time
        now = time.time()
        key = (camera_id, event_type)
        last_time = self.last_alert_time.get(key, 0)
        if now - last_time > self.alert_cooldown:
            self.last_alert_time[key] = now
            return True
        return False


    def process_frame(self, frame, camera_id="cam_001"):
        # Main detection pipeline for a single frame
        try:
            orig_h, orig_w = frame.shape[:2]
            input_w, input_h = 320, 180
            input_frame = cv2.resize(frame, (input_w, input_h))
            yolo_results = self.yolo_model(input_frame, verbose=False, device=self.device)
            weapons = self.detect_weapons(yolo_results)
            people = self.detect_people(yolo_results)
            pose_results = None
            if people:
                frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(frame_rgb)
            aggression_detected = False
            max_aggression_score = 0.0
            fighting_event = {'punch': False, 'push': False, 'fall': False, 'stampede': False, 'fighting_score': 0.0}
            person_aggression_scores = [0.0 for _ in people]
            prev_centers = self.prev_centers.get(camera_id, [])
            curr_centers = [p['center'] for p in people]
            speeds = [0.0 for _ in people]
            for i, center in enumerate(curr_centers):
                for prev_center, prev_idx in prev_centers:
                    if abs(center[0] - prev_center[0]) < 50 and abs(center[1] - prev_center[1]) < 50:
                        dx = center[0] - prev_center[0]
                        dy = center[1] - prev_center[1]
                        speed = (dx ** 2 + dy ** 2) ** 0.5
                        speeds[i] = speed
                        break
            for i, p1 in enumerate(people):
                close = False
                for j, p2 in enumerate(people):
                    if i == j:
                        continue
                    dist = ((p1['center'][0] - p2['center'][0]) ** 2 + (p1['center'][1] - p2['center'][1]) ** 2) ** 0.5
                    if dist < 120:
                        close = True
                        if speeds[i] > 15 or speeds[j] > 15:
                            person_aggression_scores[i] = min(1.0, 0.7 + 0.3 * (max(speeds[i], speeds[j]) / 30))
                        else:
                            person_aggression_scores[i] = max(person_aggression_scores[i], 0.3)
                if not close or speeds[i] < 8:
                    person_aggression_scores[i] = 0.0 if speeds[i] < 8 else 0.15
            self.prev_centers[camera_id] = [(c, i) for i, c in enumerate(curr_centers)]
            if pose_results and pose_results.pose_landmarks:
                now = datetime.now()
                self.pose_history[camera_id].append((now, pose_results.pose_landmarks.landmark))
                fighting_event = self.analyze_fighting_pose(self.pose_history[camera_id])
                aggression_score = self.calculate_pose_aggression_score(pose_results.pose_landmarks.landmark)
                max_aggression_score = max(max_aggression_score, aggression_score, fighting_event['fighting_score'])
                if (aggression_score > self.aggression_threshold or 
                    fighting_event['fighting_score'] > self.fighting_threshold or
                    fighting_event['punch'] or fighting_event['push'] or fighting_event['fall']):
                    aggression_detected = True
            crowd_detected = self.detect_crowd_formation(people)
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            for idx, p in enumerate(people):
                x1, y1, x2, y2 = p['bbox']
                p['bbox'] = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
                cx, cy = p['center']
                p['center'] = (int(cx * scale_x), int(cy * scale_y))
                p['aggression_score'] = min(1.0, max(0.0, person_aggression_scores[idx]))
            for w in weapons:
                x1, y1, x2, y2 = w['bbox']
                w['bbox'] = (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
            if aggression_detected or max(person_aggression_scores) > 0.5:
                self.aggression_score_buffer.append(max_aggression_score)
            else:
                self.aggression_score_buffer.append(0.0)
            smoothed_aggression_score = sum(self.aggression_score_buffer) / max(1, len(self.aggression_score_buffer))
            detection_results = {
                'timestamp': datetime.now().isoformat(),
                'camera_id': camera_id,
                'people_count': len(people),
                'weapons': weapons,
                'aggression_detected': aggression_detected,
                'aggression_score': smoothed_aggression_score,
                'crowd_detected': crowd_detected,
                'people_positions': people,
                'pose_landmarks': pose_results.pose_landmarks if pose_results and pose_results.pose_landmarks else None,
                'fighting_event': fighting_event
            }
            detection_results['alert_event_type'] = None
            if weapons:
                detection_results['alert_event_type'] = 'WEAPON_DETECTED'
            elif fighting_event.get('punch') or fighting_event.get('push') or fighting_event.get('fall'):
                detection_results['alert_event_type'] = 'FIGHTING_DETECTED'
            elif aggression_detected:
                detection_results['alert_event_type'] = 'AGGRESSION'
            elif crowd_detected:
                detection_results['alert_event_type'] = 'CROWD_FORMATION'
            return detection_results
        except Exception as e:
            return {'timestamp': datetime.now().isoformat(), 'error': str(e)}


    def draw_detections(self, frame: np.ndarray, results: Dict, show_aggression_per_person=False) -> np.ndarray:
        # Draw detection results on the frame
        annotated_frame = frame.copy()
        for person in results['people_positions']:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person {person['confidence']:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pose_result = self.pose.process(crop_rgb)
                if pose_result.pose_landmarks:
                    overlay = np.zeros_like(crop)
                    self.mp_draw.draw_landmarks(overlay, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    mask = np.any(overlay != 0, axis=2)
                    annotated_frame[y1:y2, x1:x2][mask] = overlay[mask]
                    if show_aggression_per_person:
                        score = self.calculate_pose_aggression_score(pose_result.pose_landmarks.landmark)
                        cv2.putText(annotated_frame, f"Agg: {score:.2f}", (x1, max(y1-30, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for weapon in results['weapons']:
            x1, y1, x2, y2 = weapon['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"{weapon['type']} {weapon['confidence']:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset = 30
        if results['aggression_detected']:
            cv2.putText(annotated_frame, "AGGRESSION DETECTED!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30
        if results['crowd_detected']:
            cv2.putText(annotated_frame, "CROWD FORMATION DETECTED!", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            y_offset += 30
        if results['weapons']:
            cv2.putText(annotated_frame, f"WEAPONS DETECTED: {len(results['weapons'])}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"People: {results['people_count']}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if not show_aggression_per_person:
            cv2.putText(annotated_frame, f"Aggression Score: {results['aggression_score']:.2f}", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Camera: {results['camera_id']}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return annotated_frame
