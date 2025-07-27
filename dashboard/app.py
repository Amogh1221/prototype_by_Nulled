import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import json
from PIL import Image
import time
import cv2
import tempfile
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log_event import EventLogger

st.set_page_config(
    page_title="Smart Campus Surveillance",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #1f77b4;
        margin: 2rem 0;
        text-align: center;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active {
        background-color: #4caf50;
    }
    .status-alert {
        background-color: #ff9800;
    }
    .status-danger {
        background-color: #f44336;
    }
</style>
""", unsafe_allow_html=True)


class SimpleSurveillanceDashboard:
    def __init__(self):
        # Initialize logger and session state
        self.logger = EventLogger()
        self.init_session_state()


    def init_session_state(self):
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = None


    def render_header(self):
        st.markdown('<div class="main-header">Smart Campus Surveillance Dashboard</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                '<div class="metric-card">'
                '<span class="status-indicator status-active"></span>'
                '<strong>System Status:</strong> Active'
                '</div>', 
                unsafe_allow_html=True
            )

        with col2:
            recent_events = self.logger.get_recent_events(1)
            alert_count = len(recent_events)
            status_class = "status-danger" if alert_count > 5 else "status-alert" if alert_count > 0 else "status-active"
            st.markdown(
                f'<div class="metric-card">'
                f'<span class="status-indicator {status_class}"></span>'
                f'<strong>Recent Alerts:</strong> {alert_count} (1h)'
                f'</div>', 
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f'<div class="metric-card">'
                f'<span class="status-indicator status-active"></span>'
                f'<strong>Last Update:</strong> {st.session_state.last_refresh.strftime("%H:%M:%S")}'
                f'</div>', 
                unsafe_allow_html=True
            )


    def render_video_upload(self):
        st.markdown(
            '<div class="upload-section">'
            '<h2>Upload Video for AI Security Analysis</h2>'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a video file to analyze for weapons, aggression, and crowd formation",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Supported formats: MP4, AVI, MOV, MKV, WMV (Max size: 200MB recommended)",
            key="video_uploader"
        )

        if uploaded_file is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"File: {uploaded_file.name}")
            with col2:
                st.info(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")
            with col3:
                st.info(f"Type: {uploaded_file.type}")

            st.subheader("Processing Settings")
            col1, col2 = st.columns(2)
            with col1:
                camera_id = st.text_input("Camera ID", value="uploaded_video", key="camera_id_input")
            with col2:
                save_video = st.checkbox("Save processed video", value=True, key="save_video")
                show_preview = st.checkbox("Show live preview", value=True, key="show_preview")

            if st.button("Start AI Analysis", type="primary", use_container_width=True, key="process_btn"):
                self.process_video(uploaded_file, camera_id, save_video, show_preview)


    def process_video(self, uploaded_file, camera_id, save_video, show_preview):
        # Main video processing loop
        import time as pytime
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        detector = None
        try:
            from detection.yolo_pose_detect import SmartSurveillanceDetector
            detector = SmartSurveillanceDetector()
        except ImportError:
            st.warning("AI detection module not found. Running in demo mode...")

        cap = cv2.VideoCapture(temp_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.markdown(f"<div style='margin-bottom:1rem;'><b>Video Properties:</b> {width}x{height}, {fps} FPS, {total_frames} frames</div>", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        if show_preview:
            video_placeholder = st.empty()

        frame_count = 0
        alert_count = 0
        detected_weapons = 0
        detected_aggression = 0
        alert_acknowledged = False
        last_alert_wall_time = 0
        alert_event_type = None
        alert_frame_idx = -1
        alert_message_displayed = False
        cooldown_seconds = 20
        error_displayed = False
        frames_to_skip = 2
        import time

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if (frame_count % frames_to_skip) != 0:
                continue
            now = time.time()
            if detector is not None:
                try:
                    results = detector.process_frame(frame, camera_id)
                    if not isinstance(results, dict) or 'alert_event_type' not in results:
                        if not error_displayed:
                            st.error("Detection error: Invalid detection result.")
                            error_displayed = True
                        annotated_frame = frame
                        continue
                    event_type = results.get('alert_event_type')
                    if event_type:
                        if not alert_acknowledged and (now - last_alert_wall_time >= cooldown_seconds):
                            last_alert_wall_time = now
                            alert_event_type = event_type
                            alert_frame_idx = frame_count
                            self.logger.log_event(results, frame)
                            st.warning(f"ALERT: {event_type} detected at frame {frame_count}! (Unacknowledged)")
                            if st.button("Acknowledge Alert", key=f"ack_{frame_count}"):
                                alert_acknowledged = True
                                alert_message_displayed = False
                        elif alert_acknowledged and not alert_message_displayed:
                            st.info("<b>Threat or suspicious activity found. Video analysis will continue.</b>", unsafe_allow_html=True)
                            alert_message_displayed = True
                    detected_weapons += len(results.get('weapons', [])) if 'weapons' in results else 0
                    if results.get('aggression_detected', False):
                        detected_aggression += 1
                    try:
                        annotated_frame = detector.draw_detections(frame, results, show_aggression_per_person=False)
                    except Exception as e:
                        if not error_displayed:
                            st.error(f"Detection error: {e}")
                            error_displayed = True
                        annotated_frame = frame
                except Exception as e:
                    if not error_displayed:
                        st.error(f"Detection error: {e}")
                        error_displayed = True
                    annotated_frame = frame
            else:
                annotated_frame = frame.copy()
                if frame_count % 50 == 0:
                    alert_count += 1
                    cv2.putText(annotated_frame, "DEMO ALERT DETECTED!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if show_preview and video_placeholder:
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.markdown(f"<b>Processing:</b> {frame_count}/{total_frames} frames ({progress*100:.1f}%)", unsafe_allow_html=True)
            if frame_count % 10 == 0:
                pytime.sleep(0.01)
        cap.release()
        os.remove(temp_path)
        st.success("Video analysis completed!")
        st.subheader("Final Analysis Results")
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        with result_col1:
            st.metric("Total Frames", total_frames)
        with result_col2:
            st.metric("Security Alerts", alert_count)
        with result_col3:
            st.metric("Weapons Detected", detected_weapons)
        with result_col4:
            st.metric("Aggression Events", detected_aggression)
        st.markdown(f"<div style='font-weight:bold;font-size:1.2rem;margin-top:1.5rem;'>Aggression Score: <span style='color:#e67e22'>{results.get('aggression_score', 0):.2f}</span></div>", unsafe_allow_html=True)
        st.session_state.last_refresh = datetime.now()


    def render_clear_logs_button(self):
        # Button to clear logs and screenshots
        if st.button("Clear All Logs & Screenshots", use_container_width=True):
            self.logger.clean_old_logs(days=0)
            st.success("All logs and screenshots have been cleared.")
            st.rerun()


    def run(self):
        self.render_header()
        self.render_video_upload()
        self.render_overview_metrics()
        self.render_recent_alerts()
        self.render_alert_screenshots()
        if st.button("Refresh Dashboard", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        self.render_clear_logs_button()


    def render_overview_metrics(self):
        st.header("Overview Metrics")
        events_df = self.logger.get_recent_events(24)
        if events_df.empty:
            st.info("No security events recorded in the last 24 hours.")
            return
        total_events = len(events_df)
        critical_events = len(events_df[events_df['severity'] == 'CRITICAL'])
        high_events = len(events_df[events_df['severity'] == 'HIGH'])
        total_weapons = events_df['weapons_detected'].sum()
        avg_people = events_df['people_count'].mean()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Events (24h)", total_events)
        with col2:
            st.metric("Critical Alerts", critical_events)
        with col3:
            st.metric("High Priority", high_events)
        with col4:
            st.metric("Weapons Detected", int(total_weapons))
        with col5:
            st.metric("Avg People/Event", f"{avg_people:.1f}" if avg_people > 0 else "0")


    def render_recent_alerts(self):
        st.header("Recent Security Alerts")
        events_df = self.logger.get_recent_events(24)
        if events_df.empty:
            st.info("No recent alerts to display.")
            return
        display_df = events_df.head(10).copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        columns_to_show = [
            'timestamp', 'camera_id', 'event_type', 'severity', 
            'people_count', 'weapons_detected', 'aggression_score'
        ]
        display_df = display_df[columns_to_show]
        def highlight_severity(row):
            if row['severity'] == 'CRITICAL':
                return ['background-color: #ffcdd2'] * len(row)
            elif row['severity'] == 'HIGH':
                return ['background-color: #ffe0b2'] * len(row)
            elif row['severity'] == 'MEDIUM':
                return ['background-color: #fff3e0'] * len(row)
            else:
                return [''] * len(row)
        styled_df = display_df.style.apply(highlight_severity, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)


    def render_alert_screenshots(self):
        st.header("Alert Screenshots")
        events_df = self.logger.get_recent_events(24)
        if events_df.empty:
            st.info("No alert screenshots available.")
            return
        events_with_screenshots = events_df[
            events_df['screenshot_path'].notna() & 
            (events_df['screenshot_path'] != '')
        ].head(12)
        if events_with_screenshots.empty:
            st.info("No screenshots found for recent alerts.")
            return
        cols = st.columns(4)
        for idx, (_, event) in enumerate(events_with_screenshots.iterrows()):
            col_idx = idx % 4
            with cols[col_idx]:
                screenshot_path = event['screenshot_path']
                if os.path.exists(screenshot_path):
                    try:
                        image = Image.open(screenshot_path)
                        st.image(
                            image,
                            caption=f"{event['event_type']} - {event['severity']}",
                            use_container_width=True
                        )
                        st.caption(f"{event['timestamp']}")
                        st.caption(f"{event['people_count']} people | {event['weapons_detected']} weapons")
                    except Exception as e:
                        st.error(f"Error loading screenshot: {e}")
                else:
                    st.warning("Screenshot file not found")


if __name__ == "__main__":
    dashboard = SimpleSurveillanceDashboard()
    dashboard.run()
