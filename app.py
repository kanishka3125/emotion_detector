import os
import uuid
import base64
import csv
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from emotion_engine import EmotionAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# In-memory session store: session_id -> EmotionAnalyzer instance
# In a real production deployment with multiple workers, a shared state like Redis would be needed.
# For local/VPS deployment with a single worker, this is sufficient.
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = EmotionAnalyzer()
    return jsonify({"session_id": session_id, "status": "success"})

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if not data or 'session_id' not in data or 'image' not in data:
        return jsonify({"error": "Missing session_id or image data"}), 400

    session_id = data['session_id']
    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 404
        
    image_data = data['image']
    
    # Handle base64 string
    # Data is expected in format "data:image/jpeg;base64,..."
    if "," in image_data:
        image_data = image_data.split(",")[1]
        
    try:
        # Decode base64 to numpy array for OpenCV
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Analyze frame
        analyzer = sessions[session_id]
        result = analyzer.process_frame(frame_bgr)
        
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/end_session/<session_id>', methods=['GET'])
def end_session(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 404
        
    analyzer = sessions[session_id]
    report_data = analyzer.get_report_data()
    
    # Calculate some basic stats for the summary
    if not report_data:
        return jsonify({"message": "No data collected", "frames_analyzed": 0})
        
    emotions = [r[1] for r in report_data]
    from collections import Counter
    emotion_counts = Counter(emotions)
    dominant_emotion = emotion_counts.most_common(1)[0][0]
    
    return jsonify({
        "status": "ended",
        "frames_analyzed": len(report_data),
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": dict(emotion_counts)
    })

@app.route('/api/download_report/<session_id>', methods=['GET'])
def download_report(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id"}), 404
        
    analyzer = sessions[session_id]
    report_data = analyzer.get_report_data()
    
    # Create CSV in memory
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Timestamp", "Emotion", "Confidence", "Status"])
    
    for row in report_data:
        timestamp, emotion, conf = row
        writer.writerow([timestamp, emotion, f"{conf:.1f}%", ""])
        
    output = si.getvalue()
    si.close()
    
    # Clean up the session since we are downloading the final report
    del sessions[session_id]
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=Emotion_Report_{session_id}.csv"}
    )

if __name__ == '__main__':
    # Run locally
    app.run(host='0.0.0.0', port=5000, debug=True)
