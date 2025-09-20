import streamlit as st
import cv2
import mediapipe as mp
import time

# =====================
# Functions for Situps
# =====================
def run_situps():
    st.title("🏋️ Situps Counter")
    st.write("Perform situps in front of your camera.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("⚠️ Unable to access webcam.")
            break

        # Flip for natural viewing
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Example situp logic (shoulder y position changes)
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y

            if left_shoulder > left_hip:
                stage = "down"
            if left_shoulder < left_hip and stage == "down":
                stage = "up"
                counter += 1

        cv2.putText(frame, f'Situps: {counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()


# =====================
# Functions for Jumps
# =====================
def run_jumps():
    st.title("🤾 Jump Counter")
    st.write("Perform jumps in front of your camera.")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("⚠️ Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Example jump logic (ankle y position changes)
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y

            if left_ankle > 0.8:
                stage = "down"
            if left_ankle < 0.7 and stage == "down":
                stage = "up"
                counter += 1

        cv2.putText(frame, f'Jumps: {counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()


# =====================
# Main App
# =====================
st.sidebar.title("🏃 Fitness Tracker")
choice = st.sidebar.selectbox("Select Activity", ["Situps", "Jumps"])

if choice == "Situps":
    run_situps()
elif choice == "Jumps":
    run_jumps()
