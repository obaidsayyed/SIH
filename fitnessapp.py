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
# Sit-Up Transformer (Unchanged)
# =====================
class SitUpVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.state = "waiting_for_down"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 0.8
        self.sit_ups_count = 0
        self.baseline_nose_y = None
        self.down_threshold = None
        self.up_threshold = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            current_time = time.time()

            if self.baseline_nose_y is None:
                self.baseline_nose_y = nose_y
                self.down_threshold = self.baseline_nose_y + 0.1
                self.up_threshold = self.baseline_nose_y - 0.1

            if self.state == "waiting_for_down":
                if nose_y > self.down_threshold:
                    self.state = "waiting_for_up"
            elif self.state == "waiting_for_up":
                if nose_y < self.up_threshold and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                    self.sit_ups_count += 1
                    self.last_rep_time = current_time
                    self.state = "waiting_for_down"

            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.rectangle(img, (10, 10), (300, 80), (16, 185, 129), -1)
        cv2.putText(img, f"Sit-Ups: {self.sit_ups_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return img

# =====================
# Jump Transformer (Unchanged)
# =====================
class JumpCounterTransformer(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.jump_count = 0
        self.state = "waiting_for_jump"
        self.last_rep_time = time.time()
        self.rep_cooldown_sec = 0.5
        self.baseline_y = None
        self.min_jump_displacement = 0.08

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y +
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            current_time = time.time()

            if self.baseline_y is None:
                self.baseline_y = hip_y

            displacement = self.baseline_y - hip_y

            if self.state == "waiting_for_jump":
                if displacement > self.min_jump_displacement:
                    self.state = "waiting_for_land"
            elif self.state == "waiting_for_land":
                if displacement < self.min_jump_displacement and (current_time - self.last_rep_time) > self.rep_cooldown_sec:
                    self.jump_count += 1
                    self.last_rep_time = current_time
                    self.state = "waiting_for_jump"

            mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.rectangle(img, (10, 10), (300, 80), (204, 0, 122), -1)
        cv2.putText(img, f"Jumps: {self.jump_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return img

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="🏃 Fitness Tracker", layout="wide")
st.sidebar.title("🏋️‍♂️ Fitness Tracker")
activity = st.sidebar.selectbox("Select Activity", ["Sit-Ups", "Jumps"])
camera_option = st.sidebar.radio("Select Camera", ["Front", "Back"])

facing_mode = "user" if camera_option == "Front" else "environment"

# Header
st.markdown(f"""
    <div style='background: linear-gradient(to right, #06b6d4, #3b82f6); padding: 30px; border-radius: 15px; text-align: center;'>
        <h1 style='color: #ffffff; font-size: 48px;'>🏃 Live {activity} Counter</h1>
        <p style='color: #f0f0f0; font-size: 20px;'>Count {activity.lower()} in real-time using your camera</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2])
with col2:
    if activity == "Sit-Ups":
        webrtc_ctx = webrtc_streamer(
            key="situp-counter",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=SitUpVideoTransformer,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "facingMode": facing_mode}, "audio": False},
            async_transform=True,
        )
    else:
        webrtc_ctx = webrtc_streamer(
            key="jump-counter",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=JumpCounterTransformer,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "facingMode": facing_mode}, "audio": False},
            async_transform=True,
        )

# =====================
# Post-session Performance
# =====================
# Initialize session state
if "session_stopped" not in st.session_state:
    st.session_state["session_stopped"] = False

stop_session = st.button("🛑 Stop Session")
if stop_session:
    st.session_state["session_stopped"] = True

if st.session_state["session_stopped"]:
    view_perf = st.button("📊 View Performance")

    if view_perf:
        # Get count safely
        count = 0
        if activity == "Sit-Ups" and webrtc_ctx.video_transformer:
            count = webrtc_ctx.video_transformer.sit_ups_count
        elif activity == "Jumps" and webrtc_ctx.video_transformer:
            count = webrtc_ctx.video_transformer.jump_count

        # Rating logic
        def calculate_rating(count, activity_type):
            if activity_type == "Sit-Ups":
                return min(round(count / 3), 10)
            elif activity_type == "Jumps":
                return min(round(count / 5), 10)
            return 0

        rating = calculate_rating(count, activity)
        st.success(f"🏆 {activity} Rating: {rating}/10")
        st.info(f"Total {activity} performed: {count}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color: gray; font-size: 14px; padding: 10px;'>
        Made with ❤️ using <b>Streamlit</b>, <b>OpenCV</b>, and <b>MediaPipe</b>.
    </div>
""", unsafe_allow_html=True)

# python -m streamlit run fitnessapp1.py  
#.\venv311\Scripts\activate
