import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

# MediaPipe setup
mp_pose = mp.solutions.pose

# Video Transformer Class
class SitUpVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.state = "waiting_for_down"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 1.0
        self.sit_ups_count = 0
        self.STANDING_THRESHOLD = 0.5
        self.SITTING_THRESHOLD = 0.7

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            current_time = time.time()

            if self.state == "waiting_for_down":
                if nose_y > self.SITTING_THRESHOLD:
                    self.state = "waiting_for_up"

            elif self.state == "waiting_for_up":
                if nose_y < self.STANDING_THRESHOLD and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                    self.sit_ups_count += 1
                    self.last_rep_time = current_time
                    self.state = "waiting_for_down"

            mp.solutions.drawing_utils.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Overlay sit-up count
            cv2.rectangle(img, (10, 10), (300, 90), (255, 0, 100), -1)
            cv2.putText(img, f"🏋️ Sit-ups: {self.sit_ups_count}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return img


# Streamlit page config
st.set_page_config(page_title="🏋️‍♂️ Sit-Up Counter", layout="wide")

# 🎨 Colorful Header
st.markdown("""
    <div style='background: linear-gradient(to right, #4e54c8, #8f94fb); padding: 30px; border-radius: 15px; text-align: center;'>
        <h1 style='color: #ffffff; font-size: 48px;'>🏋️‍♂️ Live Sit-Up Counter</h1>
        <p style='color: #f0f0f0; font-size: 20px;'>Count sit-ups in real-time using your webcam</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout columns
col1, col2 = st.columns([1, 2])

with col1:
    sit_up_display = st.empty()
    st.markdown("""
        <div style='background-color:#10B981; padding: 20px; border-radius: 10px; text-align:center;'>
            <h2 style='color: white;'>✅ Sit-Up Count</h2>
            <h1 style='color: white; font-size: 70px;' id='count'>0</h1>
        </div>
    """, unsafe_allow_html=True)

with col2:
    webrtc_ctx = webrtc_streamer(
        key="modern-situp-counter",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=SitUpVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

# Periodically update sit-up count display
if webrtc_ctx.video_transformer:
    count = webrtc_ctx.video_transformer.sit_ups_count
    sit_up_display.markdown(f"""
        <div style='background-color:#6366F1; padding: 20px; border-radius: 10px; text-align:center;'>
            <h2 style='color: white;'>💪 Current Sit-Up Count</h2>
            <h1 style='color: #FACC15; font-size: 80px;'>{count}</h1>
        </div>
    """, unsafe_allow_html=True)

# 🎯 Footer
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color: gray; font-size: 14px; padding: 10px;'>
        Made with ❤️ using <b>Streamlit</b>, <b>OpenCV</b>, and <b>MediaPipe</b>.
    </div>
""", unsafe_allow_html=True)
