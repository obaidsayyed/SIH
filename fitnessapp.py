import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import numpy as np

# MediaPipe setup
mp_pose = mp.solutions.pose

# RTC Configuration (STUN + TURN for Render HTTPS)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Free Google STUN
            {
                "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

# Situps counter logic
class SitupCounter(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.pose = mp_pose.Pose()

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            if shoulder.y < hip.y:
                self.stage = "up"
            if shoulder.y > hip.y and self.stage == "up":
                self.stage = "down"
                self.counter += 1

            cv2.putText(image, f"Situps: {self.counter}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return image


# Jump counter logic
class JumpCounter(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.pose = mp_pose.Pose()

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

            if ankle.y > 0.9:  # Standing
                self.stage = "down"
            if ankle.y < 0.8 and self.stage == "down":  # Jump detected
                self.stage = "up"
                self.counter += 1

            cv2.putText(image, f"Jumps: {self.counter}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        return image


# Streamlit UI
st.set_page_config(page_title="Fitness Tracker", layout="centered")

st.markdown(
    """
    <style>
    body {background: linear-gradient(135deg, #74ABE2, #5563DE);}
    .stSelectbox label {font-size:20px; color:white;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏋️ AI Fitness Tracker")
activity = st.selectbox("Choose Activity", ["Situps", "Jumps"])

if activity == "Situps":
    webrtc_streamer(
        key="situps",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=SitupCounter,
        media_stream_constraints={"video": True, "audio": False},
    )

elif activity == "Jumps":
    webrtc_streamer(
        key="jumps",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=JumpCounter,
        media_stream_constraints={"video": True, "audio": False},
    )
