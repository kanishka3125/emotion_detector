import cv2
from deepface import DeepFace

import time
start = time.time()
print("Testing mtcnn...")
try:
    import numpy as np
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    results = DeepFace.analyze(dummy, actions=['emotion'], detector_backend='mtcnn', enforce_detection=False)
    print(f"MTCNN success in {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
