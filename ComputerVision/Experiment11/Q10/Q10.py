import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

# Simple AR overlay: a transparent hat-like polygon over the face.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Could not load face detector.")

def draw_overlay(frame, x, y, w, h):
    overlay = frame.copy()
    # A simple polygon positioned above the face; resize with face dimensions.
    pts = np.array([
        [x + int(0.15*w), y + int(0.18*h)],
        [x + int(0.28*w), y - int(0.18*h)],
        [x + int(0.72*w), y - int(0.18*h)],
        [x + int(0.85*w), y + int(0.18*h)]
    ], dtype=np.int32)
    cv2.fillPoly(overlay, [pts], (180, 60, 220))
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    cv2.ellipse(frame, (x+w//2, y+int(0.12*h)),
                (max(1, w//2), max(1, h//10)), 0, 0, 180, (255, 255, 255), 2)

print("AR overlay demo. Press Q to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    for x, y, w, h in faces:
        draw_overlay(frame, x, y, w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow("Q10 Simple AR Face Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
