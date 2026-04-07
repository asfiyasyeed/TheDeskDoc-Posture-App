# app.py (Final Version 3.0 - Polished UI)
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Page Configuration ---
st.set_page_config(
    page_title="TheDeskDoc — AI Posture Coach",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* Base font */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #f4f3ef !important;
    border-right: 1px solid #e5e3dc;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1.25rem;
}

/* Hide default sidebar header padding */
[data-testid="stSidebarHeader"] { display: none; }

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: #1D9E75 !important;
}
[data-testid="stSlider"] .st-emotion-cache-1gv3huu {
    background: #1D9E75 !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background: #1D9E75 !important;
    border-color: #1D9E75 !important;
}

/* ── Radio buttons → pill style ── */
[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] > div {
    display: flex;
    gap: 6px;
    flex-direction: row !important;
    background: #ebe9e3;
    border-radius: 10px;
    padding: 4px;
}
[data-testid="stRadio"] > div > label {
    flex: 1;
    text-align: center;
    padding: 6px 0;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
    color: #6b6b5e;
}
[data-testid="stRadio"] > div > label:has(input:checked) {
    background: white;
    color: #1a1a16;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
[data-testid="stRadio"] > div > label > div:first-child { display: none; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #f4f3ef;
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid #e5e3dc;
}
[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9b9b8a !important;
}
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 400 !important;
    font-family: 'DM Mono', monospace !important;
    color: #1a1a16 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e3dc !important;
    border-radius: 10px !important;
    background: white;
}
[data-testid="stExpander"] summary {
    font-size: 13px;
    font-weight: 500;
    color: #4a4a3f;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* ── Alert banner ── */
.alert-banner {
    background: #FFF3EE;
    border: 1px solid #F0997B;
    border-left: 4px solid #D85A30;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 500;
    color: #993C1D;
    margin-bottom: 1rem;
}
.good-banner {
    background: #E1F5EE;
    border: 1px solid #5DCAA5;
    border-left: 4px solid #1D9E75;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 500;
    color: #0F6E56;
    margin-bottom: 1rem;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #e5e3dc;
    margin: 1rem 0;
}

/* ── Section headers ── */
.section-label {
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9b9b8a;
    margin-bottom: 8px;
    margin-top: 4px;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
}
.status-good { background: #E1F5EE; color: #0F6E56; }
.status-bad  { background: #FAECE7; color: #993C1D; }
.status-idle { background: #f4f3ef; color: #6b6b5e; }

/* ── Calibration tip steps ── */
.tip-step {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-size: 13px;
    color: #4a4a3f;
    margin-bottom: 8px;
    line-height: 1.5;
}
.tip-num {
    min-width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #E1F5EE;
    color: #0F6E56;
    font-size: 11px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Brand header ── */
.brand-block {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e5e3dc;
}
.brand-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #1D9E75;
    flex-shrink: 0;
}
.brand-name {
    font-size: 15px;
    font-weight: 500;
    color: #1a1a16;
    letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 11px;
    color: #9b9b8a;
}
</style>
""", unsafe_allow_html=True)

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Session state for metrics ---
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
if "good_frames" not in st.session_state:
    st.session_state.good_frames = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

# --- Helper Function ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# --- Video Processing Class ---
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
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        if not all(0 <= c <= 1 for c in x_coords + y_coords): return
        x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
        y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
        cv2.rectangle(image, (x_min-20, y_min-20), (x_max+20, y_max+20), (29, 158, 117), 2)
        cv2.putText(image, "tracking", (x_min-20, y_min-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (29, 158, 117), 1)

    def _analyze_sitting_posture(self, landmarks, image):
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip      = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        ear      = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        offset   = abs(shoulder.x - hip.x)
        if offset > self.thresholds['hip_shoulder_align']:
            self.feedback_list.append(f"Straighten back  ({offset:.2f})")
        neck_angle = calculate_angle([ear.x, ear.y], [shoulder.x, shoulder.y], [hip.x, hip.y])
        if neck_angle < self.thresholds['neck_angle']:
            self.feedback_list.append(f"Tuck chin in  ({int(neck_angle)}°)")

    def _analyze_standing_posture(self, landmarks, image):
        h, w, _ = image.shape
        def pt(lm): return [lm.x*w, lm.y*h]
        shoulder = pt(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
        hip      = pt(landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])
        knee     = pt(landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE])
        ankle    = pt(landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])
        back_angle = calculate_angle(shoulder, hip, knee)
        if back_angle < self.thresholds['back_angle']:
            self.feedback_list.append(f"Back angle: {int(back_angle)}°")
        leg_angle = calculate_angle(hip, knee, ankle)
        if leg_angle < self.thresholds['leg_angle']:
            self.feedback_list.append(f"Leg angle: {int(leg_angle)}°")

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        self.feedback_list = []
        landmarks = self._get_landmarks(image)

        if landmarks:
            self._draw_bounding_box(image, landmarks)
            mp_drawing.draw_landmarks(
                image, landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(29, 158, 117), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(93, 202, 165), thickness=2, circle_radius=2)
            )
            if self.mode == "Sitting":   self._analyze_sitting_posture(landmarks, image)
            elif self.mode == "Standing": self._analyze_standing_posture(landmarks, image)

            is_bad = len(self.feedback_list) > 0
            if is_bad:
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = time.time()
                elif (time.time() - self.bad_posture_start_time > 3.0 and
                      time.time() - self.last_alert_time > self.ALERT_COOLDOWN):
                    self.alert_triggered = True
                    self.last_alert_time = time.time()
                # Overlay feedback
                for i, msg in enumerate(self.feedback_list):
                    y = 44 + i * 32
                    cv2.rectangle(image, (30, y-22), (30 + len(msg)*11, y+8), (0,0,0), -1)
                    cv2.putText(image, msg, (34, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (255, 120, 80), 2)
                if self.alert_triggered:
                    cv2.rectangle(image, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
                    cv2.putText(image, "POSTURE ALERT — please adjust", (20, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 60), 2)
            else:
                self.bad_posture_start_time = None
                self.alert_triggered = False
                cv2.putText(image, "good posture", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (29, 158, 117), 2)
        else:
            cv2.putText(image, "no person detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 140), 2)

        return image


# ═══════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div class="brand-block">
        <div class="brand-dot"></div>
        <div>
            <div class="brand-name">TheDeskDoc</div>
            <div class="brand-sub">posture intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">mode</div>', unsafe_allow_html=True)
    app_mode = st.radio("mode", ("Sitting", "Standing"), label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">sensitivity</div>', unsafe_allow_html=True)

    thresholds = {}
    if app_mode == "Sitting":
        thresholds['hip_shoulder_align'] = st.slider(
            "Shoulder-hip alignment tolerance",
            min_value=0.00, max_value=0.10, value=0.05, step=0.01,
            help="Max allowed horizontal distance between shoulder and hip. Lower = stricter."
        )
        thresholds['neck_angle'] = st.slider(
            "Min neck angle",
            min_value=135, max_value=180, value=165,
            help="Angle between ear, shoulder and hip. Below this triggers a warning."
        )
    else:
        thresholds['back_angle'] = st.slider(
            "Min back straightness",
            min_value=160, max_value=180, value=170,
            help="Hip-knee angle. Lower = more lenient."
        )
        thresholds['leg_angle'] = st.slider(
            "Min leg straightness",
            min_value=160, max_value=180, value=175,
            help="Knee-ankle angle. Lower = more lenient."
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("How to calibrate for your body"):
        st.markdown("""
        <div class="tip-step">
            <div class="tip-num">1</div>
            <span>Sit or stand in your best posture first</span>
        </div>
        <div class="tip-step">
            <div class="tip-num">2</div>
            <span>Note the values shown on the video feed</span>
        </div>
        <div class="tip-step">
            <div class="tip-num">3</div>
            <span>Adjust sliders until your good posture clears all warnings</span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════

# Header row
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"## Posture monitor")
    st.markdown(
        f"<span style='font-size:13px; color:#9b9b8a;'>Real-time analysis · {app_mode} mode</span>",
        unsafe_allow_html=True
    )
with col_status:
    st.markdown(
        "<div style='padding-top:18px; text-align:right;'>"
        "<span class='status-pill status-idle'>● waiting for camera</span>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# Video stream
webrtc_streamer(
    key=f"posture-{app_mode}",
    video_transformer_factory=lambda: PostureTransformer(mode=app_mode, thresholds=thresholds),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Session metrics
elapsed = int(time.time() - st.session_state.session_start)
mins = elapsed // 60
secs = elapsed % 60

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Session time", f"{mins:02d}:{secs:02d}", "active")
with m2:
    total = st.session_state.total_frames
    good  = st.session_state.good_frames
    pct   = int((good / total * 100) if total > 0 else 0)
    st.metric("Good posture", f"{pct}%", "of session")
with m3:
    st.metric("Alerts", st.session_state.alert_count, "this session")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Tips footer
with st.expander("Tips for better detection"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Camera placement**
        - Position camera at eye level
        - Make sure your full torso is visible
        - Avoid strong backlighting
        """)
    with c2:
        st.markdown("""
        **Getting accurate readings**
        - Wear fitted clothing for better landmark detection
        - Use in a well-lit room
        - Calibrate sliders while in good posture
        """)