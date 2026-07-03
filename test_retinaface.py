import cv2
from deepface import DeepFace

print("Testing RetinaFace...")
try:
    # Use a dummy image (black square)
    import numpy as np
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    results = DeepFace.analyze(dummy, actions=['emotion'], detector_backend='retinaface', enforce_detection=False)
    print("RetinaFace success")
except Exception as e:
    print(f"Error: {e}")
