# 🏃 Fitness Tracker: Live Sit-Ups & Jump Counter

This is a **real-time fitness tracker** web app built with **Streamlit**, **OpenCV**, and **MediaPipe**. It uses your webcam to **count sit-ups and jumps** live, making it a perfect tool for home workouts, fitness challenges, or just fun tracking.  

---

## 🎯 Features

- Live Sit-Up counter with adaptive thresholds.  
- Live Jump counter using hip position for stability.  
- Real-time webcam processing using **MediaPipe Pose**.  
- Smooth and interactive UI with **Streamlit**.  
- Sidebar to select between Sit-Ups and Jumps.  
- Clean, modern design with live counters and stats.  

---

## 💻 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fitness-tracker.git
cd fitness-tracker
pip install -r requirements.txt
streamlit run app.py
streamlit==1.28.0
opencv-python==4.8.0
mediapipe==0.10.13
streamlit-webrtc==0.44.0
