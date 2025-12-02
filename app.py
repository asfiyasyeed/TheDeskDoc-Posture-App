# app.py (Final Version 2.0 - Wider Neck Calibration)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Page Configuration ---
st.set_page_config(
    page_title="TheDeskDoc - AI Posture Coach",
    page_icon="🧘‍♀️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Helper Function for Angle Calculation ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# --- The Core Video Processing Class ---
class PostureTransformer(VideoTransformerBase):
    def __init__(self, mode, thresholds):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mode = mode
        self.thresholds = thresholds
        self.feedback_list = []
        self.bad_posture_start_time = None
        self.alert_triggered = False
        self.ALERT_COOLDOWN = 10.0
        self.last_alert_time = 0

    def _get_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results.pose_landmarks
    
    def _draw_bounding_box(self, image, landmarks):
        h, w, _ = image.shape
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        if not all(0 <= c <= 1 for c in x_coords + y_coords): return
        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
        cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 255), 2)
        cv2.putText(image, "Tracking", (x_min - 20, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    def _analyze_sitting_posture(self, landmarks, image):
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        
        horizontal_offset = abs(shoulder.x - hip.x)
        if horizontal_offset > self.thresholds['hip_shoulder_align']:
            self.feedback_list.append(f"Straighten Back (Offset: {horizontal_offset:.2f})")
            
        neck_angle = calculate_angle([ear.x, ear.y], [shoulder.x, shoulder.y], [hip.x, hip.y])
        if neck_angle < self.thresholds['neck_angle']:
             self.feedback_list.append(f"Tuck Chin In (Angle: {int(neck_angle)}deg)")

    def _analyze_standing_posture(self, landmarks, image):
        h, w, _ = image.shape
        shoulder = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
        hip = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * w, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * h]
        knee = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * w, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * h]
        ankle = [landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * w, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * h]
        back_angle = calculate_angle(shoulder, hip, knee)
        if back_angle < self.thresholds['back_angle']: self.feedback_list.append(f"Back Angle: {int(back_angle)}deg")
        leg_angle = calculate_angle(hip, knee, ankle)
        if leg_angle < self.thresholds['leg_angle']: self.feedback_list.append(f"Leg Angle: {int(leg_angle)}deg")

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        self.feedback_list = []
        landmarks = self._get_landmarks(image)
        if landmarks:
            self._draw_bounding_box(image, landmarks)
            mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            if self.mode == "Sitting": self._analyze_sitting_posture(landmarks, image)
            elif self.mode == "Standing": self._analyze_standing_posture(landmarks, image)
            is_bad_posture = len(self.feedback_list) > 0
            if is_bad_posture:
                if self.bad_posture_start_time is None: self.bad_posture_start_time = time.time()
                elif time.time() - self.bad_posture_start_time > 3.0 and time.time() - self.last_alert_time > self.ALERT_COOLDOWN:
                    self.alert_triggered = True; self.last_alert_time = time.time()
                feedback_text = "FIX: " + " | ".join(self.feedback_list); color = (0, 0, 255)
            else:
                self.bad_posture_start_time = None; self.alert_triggered = False; feedback_text = "GOOD POSTURE"; color = (0, 255, 0)
        else:
            feedback_text = "NO PERSON DETECTED"; color = (255, 255, 0)

        cv2.putText(image, feedback_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        if self.alert_triggered:
            cv2.putText(image, "ALERT: FIX POSTURE!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        return image

# --- Streamlit UI ---
st.title("TheDeskDoc AI 🧘‍♀️ - AI Posture Coach")
st.markdown("An intelligent system to monitor and improve your posture in real-time. **Select your mode and calibrate for your body in the sidebar.**")
st.markdown("---")

with st.sidebar:
    st.image("https://i.imgur.com/2avP34A.png", width=150) 
    st.header("⚙️ Controls & Calibration")
    app_mode = st.radio("Select Your Mode", ("Sitting", "Standing"))
    
    with st.expander("ℹ️ How to Calibrate For Your Body"):
        st.markdown("""**1. Get into your best posture.**
**2. Look at the values** on the video feed (e.g., `Offset: 0.02`).
**3. Adjust the sliders** so your current value is in the 'good' range.""")

    st.subheader("Sensitivity Thresholds")
    thresholds = {}
    if app_mode == "Sitting":
        thresholds['hip_shoulder_align'] = st.slider(
            "Shoulder-Hip Alignment Tolerance", 0.0, 0.1, 0.05,
            help="Measures how far your shoulder can be from being perfectly above your hip. A lower value is stricter."
        )
        # <-- CHANGE: Lowered the minimum value to allow for more flexible calibration.
        thresholds['neck_angle'] = st.slider("Min Neck Angle (vs body)", 135, 180, 165)
    else: # Standing
        thresholds['back_angle'] = st.slider("Min Back Straightness", 160, 180, 170)
        thresholds['leg_angle'] = st.slider("Min Leg Straightness", 160, 180, 175)

webrtc_streamer(
    key=f"posture-analysis-{app_mode}",
    video_transformer_factory=lambda: PostureTransformer(mode=app_mode, thresholds=thresholds),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}, 
    async_processing=True,
)