import cv2
import numpy as np

# Select target color: "green", "red", or "yellow"
TARGET_COLOR = "green"

RANGES = {
    "green": [(np.array([35, 60, 40]), np.array([85, 255, 255]))],
    "yellow": [(np.array([20, 80, 80]), np.array([35, 255, 255]))],
    "red": [
        (np.array([0, 100, 70]), np.array([10, 255, 255])),
        (np.array([170, 100, 70]), np.array([179, 255, 255]))
    ]
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RANGES[TARGET_COLOR]:
        mask |= cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{TARGET_COLOR} object", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("Q6 Color Inspector", frame)
    cv2.imshow("Color Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        cv2.imwrite("color_inspection.jpg", frame)
        cv2.imwrite("color_mask.jpg", mask)
        print("Saved inspection images.")
cap.release()
cv2.destroyAllWindows()
