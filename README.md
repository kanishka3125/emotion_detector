# Emotion Detector Web App

A real-time facial emotion analysis application using DeepFace and OpenCV, converted from a desktop script to a professional web dashboard.

## Features
- **Browser-based Webcam**: Uses HTML5 `getUserMedia` for live camera feed, ensuring no server-side GUI dependencies.
- **Advanced Emotion Analysis**: Utilizes DeepFace with an OpenCV backend to analyze 7 different emotions.
- **Dynamic Calibration**: Automatically computes a baseline neutral expression over a 5-second window.
- **Relative Spike Detection**: Calculates emotion shifts relative to the baseline with higher sensitivity for specific emotions (sad, angry).
- **Stabilization Logic**: Employs an 8-frame history queue to prevent flickering, updating only when consensus is reached.
- **Live Dashboard**: Displays real-time confidence scores and probability distributions.
- **Session Reports**: Allows downloading a CSV log of the session data upon completion.

## Architecture
- **Backend**: Flask API providing session management and handling the intensive DeepFace inference.
- **Frontend**: Clean, dark-mode, responsive UI built with vanilla HTML/CSS/JS. Connects to the backend via REST endpoints.

## Local Setup

### Requirements
- Python 3.8+
- Webcam

### Installation
1. Clone the repository and navigate to the project folder.
2. It's recommended to use a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install the minimal dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`.
3. Allow camera permissions and click **Start Analysis**.

## Deployment Notes
- **Hosting the Backend**: This application requires TensorFlow and OpenCV, meaning lightweight serverless environments (like Vercel) are not suitable due to size and memory limits. Deploy the backend to a VPS, Docker container (e.g., AWS ECS, Google Cloud Run), or a specialized AI hosting platform.
- **Hosting the Frontend**: The HTML/CSS/JS can be hosted anywhere (Vercel, Netlify, S3) if you configure CORS on the Flask backend and point the API calls to your hosted Python server.
- **Production Server**: Use `gunicorn` to run the Flask app in production. Example:
  ```bash
  gunicorn -w 1 -b 0.0.0.0:5000 app:app
  ```
  *(Note: Due to the high memory footprint of DeepFace models, start with 1 worker unless your server has substantial RAM).*
