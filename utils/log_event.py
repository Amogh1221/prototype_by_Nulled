import os
import csv
import cv2
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import json


class EventLogger:
    def __init__(self, log_file: str = "logs/log.csv", alerts_dir: str = "alerts"):
        # Setup log file and alerts directory
        self.log_file = log_file
        self.alerts_dir = alerts_dir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(alerts_dir, exist_ok=True)
        self._initialize_log_file()


    def _initialize_log_file(self):
        # Create CSV log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            headers = [
                'timestamp',
                'camera_id',
                'event_type',
                'severity',
                'people_count',
                'weapons_detected',
                'aggression_score',
                'crowd_detected',
                'screenshot_path',
                'details'
            ]
            with open(self.log_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)


    def save_screenshot(self, frame, camera_id: str, event_type: str) -> str:
        # Save a screenshot of the current frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{camera_id}_{event_type}_{timestamp}.jpg"
        filepath = os.path.join(self.alerts_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath


    def determine_event_severity(self, results: Dict) -> str:
        # Calculate severity level for the detected event
        severity_score = 0
        if results.get('weapons'):
            severity_score += 3 * len(results['weapons'])
        if results.get('aggression_detected'):
            aggression_score = results.get('aggression_score', 0)
            if aggression_score > 0.7:
                severity_score += 3
            elif aggression_score > 0.5:
                severity_score += 2
            else:
                severity_score += 1
        if results.get('crowd_detected'):
            people_count = results.get('people_count', 0)
            if people_count > 10:
                severity_score += 3
            elif people_count > 5:
                severity_score += 2
            else:
                severity_score += 1
        fighting_event = results.get('fighting_event', {})
        fighting_score = fighting_event.get('fighting_score', 0)
        if fighting_score > 0.7:
            severity_score += 3
        elif fighting_score > 0.5:
            severity_score += 2
        elif fighting_score > 0.3:
            severity_score += 1
        if fighting_event.get('punch', False):
            severity_score += 2
        if fighting_event.get('push', False):
            severity_score += 1
        if fighting_event.get('fall', False):
            severity_score += 2
        if fighting_event.get('stampede', False):
            severity_score += 2
        people_count = results.get('people_count', 0)
        if people_count <= 2 and (fighting_score > 0.3 or fighting_event.get('punch', False)):
            severity_score += 2
        if severity_score >= 4:
            return 'CRITICAL'
        elif severity_score >= 2:
            return 'HIGH'
        elif severity_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'


    def determine_event_type(self, results: Dict) -> str:
        # Determine the primary event type from detection results
        event_types = []
        if results.get('weapons'):
            event_types.append('WEAPON_DETECTED')
        if results.get('aggression_detected'):
            event_types.append('AGGRESSION')
        if results.get('crowd_detected'):
            event_types.append('CROWD_FORMATION')
        fighting_event = results.get('fighting_event', {})
        if fighting_event.get('punch', False):
            event_types.append('PUNCH_DETECTED')
        if fighting_event.get('push', False):
            event_types.append('PUSH_DETECTED')
        if fighting_event.get('fall', False):
            event_types.append('FALL_DETECTED')
        if fighting_event.get('stampede', False):
            event_types.append('STAMPEDE_DETECTED')
        if fighting_event.get('fighting_score', 0) > 0.3:
            event_types.append('FIGHTING_DETECTED')
        if len(event_types) == 1:
            return event_types[0]
        elif len(event_types) > 1:
            return 'MULTIPLE_THREATS'
        else:
            return 'UNKNOWN'


    def log_event(self, results: Dict, frame=None, save_screenshot: bool = True) -> bool:
        # Log a security event to the CSV file
        try:
            event_type = self.determine_event_type(results)
            severity = self.determine_event_severity(results)
            screenshot_path = ""
            if frame is not None and save_screenshot:
                screenshot_path = self.save_screenshot(
                    frame, 
                    results.get('camera_id', 'unknown'), 
                    event_type.lower()
                )
            details = {
                'weapons': results.get('weapons', []),
                'aggression_score': results.get('aggression_score', 0),
                'people_positions': len(results.get('people_positions', [])),
                'timestamp_detailed': results.get('timestamp', datetime.now().isoformat())
            }
            row_data = [
                results.get('timestamp', datetime.now().isoformat()),
                results.get('camera_id', 'unknown'),
                event_type,
                severity,
                results.get('people_count', 0),
                len(results.get('weapons', [])),
                results.get('aggression_score', 0),
                results.get('crowd_detected', False),
                screenshot_path,
                json.dumps(details)
            ]
            with open(self.log_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row_data)
            return True
        except Exception as e:
            print(f"Error logging event: {e}")
            return False


    def get_recent_events(self, hours: int = 24) -> pd.DataFrame:
        # Get recent events from the log file
        try:
            if not os.path.exists(self.log_file):
                return pd.DataFrame()
            df = pd.read_csv(self.log_file)
            if df.empty:
                return df
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
            recent_df = df[df['timestamp'] >= cutoff_time]
            return recent_df.sort_values('timestamp', ascending=False)
        except Exception as e:
            print(f"Error retrieving recent events: {e}")
            return pd.DataFrame()


    def get_event_statistics(self, hours: int = 24) -> Dict:
        # Get statistics about recent events
        try:
            recent_events = self.get_recent_events(hours)
            if recent_events.empty:
                return {
                    'total_events': 0,
                    'by_severity': {},
                    'by_type': {},
                    'by_camera': {},
                    'weapons_total': 0,
                    'avg_people_count': 0
                }
            stats = {
                'total_events': len(recent_events),
                'by_severity': recent_events['severity'].value_counts().to_dict(),
                'by_type': recent_events['event_type'].value_counts().to_dict(),
                'by_camera': recent_events['camera_id'].value_counts().to_dict(),
                'weapons_total': recent_events['weapons_detected'].sum(),
                'avg_people_count': recent_events['people_count'].mean()
            }
            return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}


    def clean_old_logs(self, days: int = 30):
        # Clean old log entries and screenshots
        try:
            if not os.path.exists(self.log_file):
                return
            df = pd.read_csv(self.log_file)
            if df.empty:
                return
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_time = datetime.now() - pd.Timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_time]
            old_df = df[df['timestamp'] < cutoff_time]
            screenshots_to_delete = old_df['screenshot_path'].dropna().tolist()
            for screenshot_path in screenshots_to_delete:
                if os.path.exists(screenshot_path):
                    os.remove(screenshot_path)
            recent_df.to_csv(self.log_file, index=False)
            print(f"Cleaned {len(old_df)} old log entries and {len(screenshots_to_delete)} screenshots")
        except Exception as e:
            print(f"Error cleaning old logs: {e}")
