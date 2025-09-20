import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

# ------------------------
# MediaPipe Setup
# ------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------------
# Sit-Up Transformer
# ------------------------
class SitUpTransformer(VideoTransformerBase):
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

            if self.state == "waiting_for_down" and nose_y > self.down_threshold:
                self.state = "waiting_for_up"
            elif self.state == "waiting_for_up" and nose_y < self.up_threshold and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                self.sit_ups_count += 1
                self.last_rep_time = current_time
                self.state = "waiting_for_down"

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(img, f"Sit-Ups: {self.sit_ups_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return img

# ------------------------
# Jump Transformer
# ------------------------
class JumpTransformer(VideoTransformerBase):
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

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.putText(img, f"Jumps: {self.jump_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        return img

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="🏃 Fitness Tracker", layout="centered")
st.markdown("""
    <div style='background: linear-gradient(90deg, #ff7e5f, #feb47b); padding: 25px; border-radius: 15px; text-align:center;'>
        <h1 style='color:white; font-size:48px;'>🏋️‍♂️ Fitness Tracker</h1>
        <p style='color:white; font-size:18px;'>Count Sit-Ups & Jumps in real-time using your camera</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------
# Activity Selection
# ------------------------
activity = st.selectbox("Choose Activity", ["Sit-Ups", "Jumps"])

st.info("⚠️ Please allow camera access in your browser!")

# ------------------------
# Start WebRTC Streamer
# ------------------------
rtc_configuration = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun2.l.google.com:19302"},
        {"urls": "stun:stun.services.mozilla.com"}
    ]
}

if activity == "Sit-Ups":
    webrtc_streamer(
        key="situps",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SitUpTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
        rtc_configuration=rtc_configuration
    )
else:
    webrtc_streamer(
        key="jumps",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=JumpTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
        rtc_configuration=rtc_configuration
    )
