import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

# =====================
# MediaPipe Setup
# =====================
mp_pose = mp.solutions.pose

# =====================
# Sit-Up Transformer
# =====================
class SitUpVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.state = "waiting_for_down"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 1.2
        self.sit_ups_count = 0
        self.baseline_nose_y = None
        self.down_threshold = None
        self.up_threshold = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            current_time = time.time()

            if self.baseline_nose_y is None:
                self.baseline_nose_y = nose_y
                self.down_threshold = self.baseline_nose_y + 0.15
                self.up_threshold = self.baseline_nose_y - 0.15
                self.sit_ups_count = 0

            if self.state == "waiting_for_down" and nose_y > self.down_threshold:
                self.state = "waiting_for_up"
            elif self.state == "waiting_for_up" and nose_y < self.up_threshold and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                self.sit_ups_count += 1
                self.last_rep_time = current_time
                self.state = "waiting_for_down"

            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return img


# =====================
# Jump Transformer
# =====================
class JumpCounterTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.state = "waiting_for_jump"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 1.0
        self.jump_count = 0
        self.baseline_y = None
        self.up_threshold = None
        self.down_threshold = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            current_time = time.time()

            if self.baseline_y is None:
                self.baseline_y = hip_y
                self.down_threshold = self.baseline_y + 0.1
                self.up_threshold = self.baseline_y - 0.1

            if self.state == "waiting_for_jump" and hip_y < self.up_threshold:
                self.state = "waiting_for_land"
            elif self.state == "waiting_for_land" and hip_y > self.down_threshold and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                self.jump_count += 1
                self.last_rep_time = current_time
                self.state = "waiting_for_jump"

            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.rectangle(img, (10, 10), (320, 90), (204, 0, 122), -1)
            cv2.putText(img, f"Jumps: {self.jump_count}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return img


# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="🏃 Fitness Tracker", layout="wide")

st.markdown(f"""
    <div style='background: linear-gradient(to right, #06b6d4, #3b82f6); padding: 30px; border-radius: 15px; text-align: center;'>
        <h1 style='color: #ffffff; font-size: 42px;'>🏃 Fitness Tracker</h1>
        <p style='color: #f0f0f0; font-size: 18px;'>Track your Sit-Ups and Jumps in real-time using your webcam</p>
    </div>
""", unsafe_allow_html=True)

activity = st.selectbox("Choose Activity", ["Sit-Ups", "Jumps"])

rtc_config = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
}

if activity == "Sit-Ups":
    webrtc_ctx = webrtc_streamer(
        key="situp-counter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SitUpVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
        rtc_configuration=rtc_config
    )
    if webrtc_ctx.video_transformer:
        st.metric("💪 Sit-Ups Count", webrtc_ctx.video_transformer.sit_ups_count)

elif activity == "Jumps":
    webrtc_ctx = webrtc_streamer(
        key="jump-counter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=JumpCounterTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
        rtc_configuration=rtc_config
    )
    if webrtc_ctx.video_transformer:
        st.metric("🦵 Jump Count", webrtc_ctx.video_transformer.jump_count)
