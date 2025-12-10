import cv2
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from deepface import DeepFace
from collections import deque, Counter

# --- CONFIGURATION ---
DETECTOR_BACKEND = 'opencv' 

# Create Report File
session_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
csv_filename = f"Final_Report_{session_id}.csv"
csv_file = open(csv_filename, mode='w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["Timestamp", "Emotion", "Confidence", "Status"])

# Lists for Graph
graph_times = []
graph_emotions = []

# Camera Setup
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Calibration Variables
baseline_emotions = {}
calibrated = False
calibration_frames = 0
MAX_CALIBRATION_FRAMES = 30 # Approx 5 seconds

# Stability Variables
emotion_history = deque(maxlen=8)
current_emotion = "Neutral"
current_confidence = 0.0
display_color = (255, 255, 0)

print("--------------------------------------------------")
print(" ULTIMATE ACCURACY + PERCENTAGE MODE")
print(" Step 1: Calibration (Sit still for 5 seconds)")
print(" Step 2: Press 'q' to see the Graph")
print("--------------------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1) # Mirror effect
    
    # --- PHASE 1: CALIBRATION ---
    if not calibrated:
        # Draw UI
        cv2.rectangle(frame, (50, 200), (590, 280), (0, 0, 0), -1)
        cv2.putText(frame, f"CALIBRATING... {calibration_frames/MAX_CALIBRATION_FRAMES*100:.0f}%", 
                   (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "LOOK AT CAMERA - RELAX FACE", (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if calibration_frames % 2 == 0:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], detector_backend=DETECTOR_BACKEND, enforce_detection=False, silent=True)
                if isinstance(results, list): results = results[0]
                
                # Learn Baseline
                emotions = results['emotion']
                for key, val in emotions.items():
                    baseline_emotions[key] = baseline_emotions.get(key, 0) + val
            except: pass
        
        calibration_frames += 1
        
        if calibration_frames >= MAX_CALIBRATION_FRAMES:
            # Average the baseline
            for key in baseline_emotions:
                baseline_emotions[key] /= (MAX_CALIBRATION_FRAMES / 2)
            print("Calibration Complete!")
            calibrated = True
            
        cv2.imshow('Ultimate AI', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    # --- PHASE 2: ACTIVE DETECTION ---
    
    # Lighting Correction (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    frame_enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

    if calibration_frames % 2 == 0:
        try:
            results = DeepFace.analyze(frame_enhanced, actions=['emotion'], detector_backend=DETECTOR_BACKEND, enforce_detection=False, silent=True)
            if isinstance(results, list): results = results[0]
            
            raw_emotions = results['emotion']
            
            # Compare to Baseline (The Relative Math)
            best_emotion = "neutral"
            highest_spike = 0
            
            for emotion, score in raw_emotions.items():
                baseline = baseline_emotions.get(emotion, 0)
                spike = score - baseline
                if emotion in ['sad', 'angry']: spike *= 1.5 
                
                if spike > highest_spike:
                    highest_spike = spike
                    best_emotion = emotion
            
            final_decision = "neutral" if highest_spike < 10 else best_emotion

            # Stabilize
            emotion_history.append(final_decision)
            consistent_emotion = Counter(emotion_history).most_common(1)[0]
            if consistent_emotion[1] >= 4:
                current_emotion = consistent_emotion[0]
                # Get the actual percentage for this winning emotion
                current_confidence = raw_emotions.get(current_emotion, 0)

            # SAVE DATA FOR GRAPH
            current_time = datetime.now().strftime("%H:%M:%S")
            writer.writerow([current_time, current_emotion, f"{current_confidence:.1f}%"])
            graph_times.append(current_time)
            graph_emotions.append(current_emotion)

        except: pass

    # --- DRAWING ---
    cv2.rectangle(frame, (0, 420), (640, 480), (20, 20, 20), -1)
    
    if current_emotion in ['happy', 'surprise']: display_color = (0, 255, 0)
    elif current_emotion in ['sad', 'fear', 'angry']: display_color = (0, 0, 255)
    else: display_color = (200, 200, 200)

    # Added Percentage Display Here
    text = f"STATUS: {current_emotion.upper()} ({current_confidence:.1f}%)"
    cv2.putText(frame, text, (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
    
    # Rec dot
    if (calibration_frames // 10) % 2 == 0:
         cv2.circle(frame, (600, 50), 8, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (530, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Ultimate AI', frame)
    calibration_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- PHASE 3: GENERATE GRAPH ---
cap.release()
cv2.destroyAllWindows()
csv_file.close()

if len(graph_emotions) > 0:
    print("Generating Session Graph...")
    plt.figure(figsize=(10, 5))
    
    # Plot data points
    plt.plot(graph_times, graph_emotions, marker='o', linestyle='-', color='blue', markersize=4)
    
    # Styling
    plt.title(f'Emotion Analysis Report - {session_id}', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Detected Emotion', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Rotate timestamps so they don't overlap
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save and Show
    plt.savefig(f"Graph_{session_id}.png")
    plt.show()
else:
    print("Not enough data to graph.")