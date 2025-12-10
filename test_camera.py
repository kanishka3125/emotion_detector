import cv2

print("Testing Camera 0...")
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap0.isOpened():
    print("✅ Camera 0 is working!")
    cap0.release()
else:
    print("❌ Camera 0 failed.")

print("Testing Camera 1...")
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if cap1.isOpened():
    print("✅ Camera 1 is working!")
    cap1.release()
else:
    print("❌ Camera 1 failed.")