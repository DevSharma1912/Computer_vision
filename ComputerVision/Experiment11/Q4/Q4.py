import cv2
import os
import csv
import numpy as np
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
# In this project, Dev/ and Student2/ are beside Q4.py.
DATASET_PATH = os.path.join(HERE, "dataset")
ATTENDANCE_FILE = os.path.join(HERE, "attendance.csv")
FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 75

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade.")

faces, labels, label_names = [], [], {}
label_id = 0

for person_name in sorted(os.listdir(DATASET_PATH)):
    person_folder = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_folder):
        continue
    # Ignore generated folders and folders not intended as people.
    if person_name.lower() in {"dataset", "__pycache__"}:
        continue

    person_count = 0
    for filename in os.listdir(person_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        path = os.path.join(person_folder, filename)
        image = cv2.imread(path)
        if image is None:
            print("Cannot read image:", path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(found) == 0:
            print("No face detected in:", path)
            continue
        x, y, w, h = max(found, key=lambda b: b[2] * b[3])
        face = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
        faces.append(face)
        labels.append(label_id)
        person_count += 1

    if person_count:
        label_names[label_id] = person_name
        label_id += 1
    else:
        print(f"No usable face photos found for {person_name}.")

if not faces:
    raise RuntimeError(
        "No training faces found. Put clear JPG/PNG face photos inside "
        "Q4\\Dev\\ and Q4\\Student2\\ (folders beside Q4.py)."
    )

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels, dtype=np.int32))
print("Training complete:", label_names)

if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Name", "Date", "Time", "Status"])

today = datetime.now().strftime("%Y-%m-%d")
marked_today = set()
with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("Date") == today:
            marked_today.add(row.get("Name"))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened. Try changing VideoCapture(0) to VideoCapture(1).")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read camera frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        for x, y, w, h in found:
            face = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            predicted_id, distance = recognizer.predict(face)
            if predicted_id in label_names and distance < CONFIDENCE_THRESHOLD:
                name, color = label_names[predicted_id], (0, 200, 0)
                status = "Present"
                if name not in marked_today:
                    now = datetime.now()
                    with open(ATTENDANCE_FILE, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            name, now.strftime("%Y-%m-%d"),
                            now.strftime("%H:%M:%S"), status
                        ])
                    marked_today.add(name)
                    print("Attendance marked:", name)
            else:
                name, status, color = "Unknown", "Not registered", (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, max(y-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, status, (x, y+h+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Q4 Smart Attendance - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
print("Attendance saved to:", ATTENDANCE_FILE)
