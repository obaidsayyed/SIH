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
# Jump Transformer
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
st.set_page_config(page_title="AthliTech 🏃", layout="wide")

# ===== BMI Section with Highlighted Background =====
st.sidebar.header("👤 Athlete Details")
height_unit = st.sidebar.selectbox("Select height unit:", ["cm", "feet & inches"])

if height_unit == "cm":
    height_cm = st.sidebar.number_input("Enter height (cm):", min_value=50, max_value=250, step=1)
else:
    height_feet = st.sidebar.number_input("Feet:", min_value=3, max_value=8, step=1)
    height_inches = st.sidebar.number_input("Inches:", min_value=0, max_value=11, step=1)
    height_cm = round((height_feet * 30.48) + (height_inches * 2.54), 2)

weight_kg = st.sidebar.number_input("Enter weight (kg):", min_value=10, max_value=200, step=1)

if height_cm > 0 and weight_kg > 0:
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)

    # Determine background color and range text
    if bmi < 18.5:
        bg_color = "#EAB308"  # Yellow
        bmi_range_text = "Underweight"
    elif 18.5 <= bmi <= 24.9:
        bg_color = "#16A34A"  # Green
        bmi_range_text = "Normal"
    elif 25 <= bmi <= 29.9:
        bg_color = "#DC2626"  # Red
        bmi_range_text = "Overweight"
    else:
        bg_color = "#000000"  # Black
        bmi_range_text = "Obese"

    # Display BMI with background highlight and white text
    st.sidebar.markdown(f"""
        <div style='background-color:{bg_color}; padding: 10px; border-radius: 8px; text-align:center;'>
            <h3 style='color:white; margin:0'>📊 BMI: {bmi}</h3>
            <p style='color:white; margin:0'>{bmi_range_text}</p>
        </div>
    """, unsafe_allow_html=True)

# ===== Activity Selection =====
st.sidebar.title("AthliTech 🏃")
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
        Made with ❤️ using <b>Streamlit</b>, <b>OpenCV</b>, and <b>MediaPipe</b>, <b> by team Neura-Minds</b>.
    </div>
""", unsafe_allow_html=True)

# Run using:
# python -m streamlit run fitapp.py
