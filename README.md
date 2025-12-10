emotion_detector
🧠 Research-Grade AI Emotion Detector

A real-time AI system that detects human emotions using Computer Vision and Deep Learning. Unlike standard detectors, this project features an **Automatic Calibration System** that learns the user's specific facial structure to ensure high accuracy.

## 🚀 Features
* **Adaptive Calibration:** Scans the user's resting face for 5 seconds to establish a baseline.
* **Lighting Correction (CLAHE):** Automatically enhances video in dark environments.
* **Live Confidence Dashboard:** Displays real-time probability percentages.
* **Session Reporting:** Generates a visual graph (`.png`) and data log (`.csv`) after every session.

## 🛠️ Installation
1. Clone this repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
