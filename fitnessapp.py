import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

# -------------------
# MediaPipe Setup
# -------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# -------------------
# Sit-Up Transformer
# -------------------
class SitUpVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.state = "waiting_for_down"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 1.0
        self.sit_ups_count = 0
        self.baseline_nose_y = None
        self.down_threshold = None
        self.up_threshold = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        if results.pose_landmarks:
            nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
            current_time = time.time()

            if self.baseline_nose_y is None:
                self.baseline_nose_y = nose_y
                self.down_threshold = self.baseline_nose_y + 0.15
                self.up_threshold = self.baseline_nose_y - 0.12

            if self.state == "waiting_for_down" and nose_y > self.down_threshold:
                self.state = "waiting_for_up"
            elif self.state == "waiting_for_up" and nose_y < self.up_threshold and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                self.sit_ups_count += 1
                self.last_rep_time = current_time
                self.state = "waiting_for_down"

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlay situp count
        cv2.rectangle(img, (10, 10), (360, 100), (16, 185, 129), -1)
        cv2.putText(img, f"Sit-Ups: {self.sit_ups_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        return img

# -------------------
# Jump Transformer
# -------------------
class JumpVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.state = "waiting_for_jump"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 0.8
        self.jump_count = 0
        self.baseline_hip_y = None
        self.up_threshold = None
        self.down_threshold = None
        self.hip_history = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        if results.pose_landmarks:
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            hip_y = (left_hip + right_hip) / 2

            # Smooth the hip Y
            self.hip_history.append(hip_y)
            if len(self.hip_history) > 3:
                self.hip_history.pop(0)
            smoothed_hip = sum(self.hip_history) / len(self.hip_history)

            current_time = time.time()
            if self.baseline_hip_y is None:
                self.baseline_hip_y = smoothed_hip
                self.down_threshold = self.baseline_hip_y + 0.08
                self.up_threshold = self.baseline_hip_y - 0.12

            if self.state == "waiting_for_jump" and smoothed_hip < self.up_threshold:
                self.state = "waiting_for_land"
            elif self.state == "waiting_for_land" and smoothed_hip > self.down_threshold and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                self.jump_count += 1
                self.last_rep_time = current_time
                self.state = "waiting_for_jump"

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlay jump count
        cv2.rectangle(img, (10, 10), (360, 100), (236, 72, 153), -1)
        cv2.putText(img, f"Jumps: {self.jump_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        return img

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="🏋️ Fitness Tracker", layout="wide")

# Custom CSS for modern UI
st.markdown("""
<style>
body {
    background: #f0f4f8;
}
h1 {
    text-align:center;
    font-size: 50px;
    font-weight: bold;
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    -webkit-background-clip: text;
    color: transparent;
}
.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white;
    font-size: 20px;
    padding: 10px 25px;
    border-radius: 15px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    opacity: 0.8;
    transform: scale(1.05);
}
.selectbox {
    text-align:center;
    margin: 20px auto;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🏃 Live Fitness Tracker</h1>', unsafe_allow_html=True)

# Activity dropdown
activity = st.selectbox("Select Activity", ["Sit-Ups", "Jumps"])

# WebRTC streamer
if activity == "Sit-Ups":
    ctx = webrtc_streamer(
        key="situp-counter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SitUpVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )
else:
    ctx = webrtc_streamer(
        key="jump-counter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=JumpVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )

# Live counter card
counter_placeholder = st.empty()
if ctx.video_transformer:
    while True:
        count = ctx.video_transformer.sit_ups_count if activity=="Sit-Ups" else ctx.video_transformer.jump_count
        color = "#10B981" if activity=="Sit-Ups" else "#EC4899"
        counter_placeholder.markdown(f"""
            <div style='margin:20px auto; max-width:400px; background: linear-gradient(135deg,{color},#FACC15);
                        padding:30px; border-radius:25px; text-align:center; box-shadow:0px 8px 20px rgba(0,0,0,0.2);'>
                <h1 style='font-size:70px; color:white;'>{count}</h1>
                <h2 style='color:white; margin-top:-10px;'>{activity} Count</h2>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.5)

# Footer
st.markdown("""
<div style='text-align:center; color:gray; margin-top:20px; font-size:14px;'>
Made with ❤️ using <b>Streamlit</b>, <b>OpenCV</b>, and <b>MediaPipe</b>
</div>
""", unsafe_allow_html=True)
