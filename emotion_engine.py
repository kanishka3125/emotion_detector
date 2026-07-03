import cv2
import numpy as np
from deepface import DeepFace
from collections import deque, Counter

class EmotionAnalyzer:
    def __init__(self):
        self.DETECTOR_BACKEND = 'opencv'
        self.MAX_CALIBRATION_FRAMES = 30
        
        # State
        self.baseline_emotions = {}
        self.calibrated = False
        self.calibration_frames = 0
        
        self.emotion_history = deque(maxlen=3)
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0

        # Data collection for graphing/report
        self.session_data = [] # List of tuples: (timestamp_str, emotion, confidence_pct)

    def process_frame(self, frame_bgr):
        """
        Takes a BGR frame (from OpenCV), processes it via DeepFace, applies
        calibration and spike logic, and returns a dictionary with current status.
        """
        # PHASE 1: CALIBRATION
        if not self.calibrated:
            try:
                results = DeepFace.analyze(frame_bgr, actions=['emotion'], detector_backend=self.DETECTOR_BACKEND, enforce_detection=True, silent=True)
                if isinstance(results, list): results = results[0]
                
                emotions = results['emotion']
                for key, val in emotions.items():
                    self.baseline_emotions[key] = self.baseline_emotions.get(key, 0) + val
            except Exception as e:
                pass # Ignore if no face detected or deepface error

            self.calibration_frames += 1
            
            if self.calibration_frames >= self.MAX_CALIBRATION_FRAMES:
                for key in self.baseline_emotions:
                    self.baseline_emotions[key] /= self.MAX_CALIBRATION_FRAMES
                self.calibrated = True

            return {
                "status": "calibrating",
                "progress": min(100, int((self.calibration_frames / self.MAX_CALIBRATION_FRAMES) * 100)),
                "emotion": "Neutral",
                "confidence": 0.0,
                "probabilities": {}
            }

        # PHASE 2: ACTIVE DETECTION
        raw_emotions = {}
        try:
            if frame_bgr is None:
                raise ValueError("Empty frame")

            # Lighting Correction (CLAHE)
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            frame_enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

            results = DeepFace.analyze(frame_enhanced, actions=['emotion'], detector_backend=self.DETECTOR_BACKEND, enforce_detection=True, silent=True)
            if isinstance(results, list): results = results[0]
            raw_emotions = results['emotion']
            
            # Compare to Baseline (The Relative Math)
            best_emotion = "neutral"
            highest_spike = 0
            
            for emotion, score in raw_emotions.items():
                baseline = self.baseline_emotions.get(emotion, 0)
                spike = score - baseline
                if emotion in ['sad', 'angry']: 
                    spike *= 1.5 
                
                if spike > highest_spike:
                    highest_spike = spike
                    best_emotion = emotion
            
            final_decision = "neutral" if highest_spike < 10 else best_emotion

            # Stabilize - reduced buffer for faster web response
            self.emotion_history.append(final_decision)
            consistent_emotion = Counter(self.emotion_history).most_common(1)[0]
            if consistent_emotion[1] >= 2:
                self.current_emotion = consistent_emotion[0]
                self.current_confidence = float(raw_emotions.get(self.current_emotion, 0))

        except Exception as e:
            pass # No face detected or processing error

        # Only append data if we are calibrated and have data
        import datetime
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.session_data.append((current_time, self.current_emotion, float(self.current_confidence)))

        return {
            "status": "active",
            "progress": 100,
            "emotion": str(self.current_emotion),
            "confidence": float(self.current_confidence),
            "probabilities": {str(k): float(v) for k, v in raw_emotions.items()}
        }

    def get_report_data(self):
        """Returns the accumulated session data for generating a CSV"""
        return self.session_data
