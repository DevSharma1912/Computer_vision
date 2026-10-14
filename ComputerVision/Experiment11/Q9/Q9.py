import cv2
import time
import math

# Uses OpenCV Haar eye detection for a basic classroom demo.
# This is not a medical or safety-grade drowsiness detector.
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Could not load Haar cascade files.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

eyes_missing_since = None
ALARM_SECONDS = 2.0

print("Keep your face visible. Press Q to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    eyes_found = False
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 180, 0), 2)
        roi_gray = gray[y:y+int(h*0.65), x:x+w]
        roi_frame = frame[y:y+int(h*0.65), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(15, 15))
        if len(eyes) >= 1:
            eyes_found = True
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    now = time.time()
    if faces and not eyes_found:
        if eyes_missing_since is None:
            eyes_missing_since = now
        elif now - eyes_missing_since >= ALARM_SECONDS:
            cv2.putText(frame, "WARNING: EYES NOT DETECTED", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("Warning: eyes not detected for a sustained period.")
    else:
        eyes_missing_since = None
        cv2.putText(frame, "Monitoring", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Q9 Blink/Drowsiness Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
