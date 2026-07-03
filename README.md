# EmotionAI — Adaptive Facial Emotion Intelligence

A real-time facial emotion analysis application using DeepFace and OpenCV, converted from a desktop script to a professional web dashboard.

## Features
- **Browser-based Webcam**: Uses HTML5 `getUserMedia` for live camera feed — no server-side GUI dependencies.
- **Advanced Emotion Analysis**: Utilises DeepFace with an OpenCV backend to analyse 7 different emotions.
- **Personal Baseline Calibration**: Automatically computes a baseline neutral expression over a 5-second window.
- **Relative Spike Detection**: Calculates emotion shifts relative to the baseline, with higher sensitivity for sad and angry.
- **Stabilisation Logic**: 3-frame majority-vote queue prevents flickering.
- **Live Dashboard**: Real-time confidence scores and probability bars.
- **Session Reports**: Download a timestamped CSV log of the full session on completion.

## Architecture
- **Backend**: Flask API providing session management and DeepFace inference.
- **Frontend**: Dark-mode responsive UI in vanilla HTML/CSS/JS, communicating via REST endpoints.
- **WSGI Server**: Gunicorn (production) / Flask dev server (local).

---

## Local Setup

### Requirements
- Python 3.10+ (3.11 recommended)
- Webcam

### Installation

```bash
# 1. Clone and enter the directory
git clone <your-repo-url>
cd emotion_detector

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running Locally

**Option A — Flask dev server (simplest):**
```bash
python app.py
```

**Option B — Gunicorn (mirrors production exactly):**
```bash
gunicorn -w 1 --timeout 120 -b 0.0.0.0:5000 app:app
```

Then open `http://localhost:5000`, allow camera permissions, and click **Start Analysis**.

---

## Deploying to Render

### Prerequisites
- A [Render](https://render.com) account (free tier works)
- This repository pushed to GitHub or GitLab

### Step-by-step

1. **Push your code to GitHub** (make sure `render.yaml` is in the root):
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create a new Web Service on Render:**
   - Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**
   - Connect your GitHub account and select this repository

3. **Configure the service** (Render will auto-detect `render.yaml` — fields below are for manual setup):

   | Setting | Value |
   |---------|-------|
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn -w 1 --timeout 120 -b 0.0.0.0:$PORT app:app` |
   | **Instance Type** | Standard (512 MB RAM minimum; 1 GB+ recommended for DeepFace) |

4. **Environment Variables** (set in Render dashboard → Environment):

   | Key | Value |
   |-----|-------|
   | `SECRET_KEY` | Click **Generate** for a random secure value |
   | `PYTHON_VERSION` | `3.11.0` |

5. **Deploy**: Click **Create Web Service**. Render will install dependencies and start the server. The first deploy takes a few minutes — DeepFace downloads its model weights on first request.

6. **Access your app** at the `.onrender.com` URL shown in the dashboard.

### Important Notes

- **Single worker**: The start command uses `-w 1`. DeepFace + TensorFlow are memory-heavy; adding more workers on a small instance will cause OOM crashes.
- **Cold starts**: Free-tier Render services spin down after 15 minutes of inactivity. The first request after sleep will be slow (30–60 s) while models reload.
- **Webcam privacy**: The browser captures frames and sends them as base64 to your server for analysis. No video is stored server-side.
- **HTTPS**: Render provides TLS automatically. Browsers require HTTPS to access the webcam via `getUserMedia`, so deployment on Render works out of the box.

### Build Command (for Render dashboard)
```
pip install -r requirements.txt
```

### Start Command (for Render dashboard)
```
gunicorn -w 1 --timeout 120 -b 0.0.0.0:$PORT app:app
```
