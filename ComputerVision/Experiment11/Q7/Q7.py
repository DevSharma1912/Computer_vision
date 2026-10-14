import cv2
import numpy as np

# Track a colored object. Default is blue; change HSV bounds for your object.
LOWER = np.array([95, 100, 50])
UPPER = np.array([130, 255, 255])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

canvas = None
previous = None
kernel = np.ones((5, 5), np.uint8)
print("Move a blue object in front of the camera. Press C to clear; Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            (x, y), radius = cv2.minEnclosingCircle(largest)
            if radius > 5:
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    if center is None:
        previous = None
    else:
        if previous is not None:
            cv2.line(canvas, previous, center, (255, 0, 0), 5)
        previous = center

    output = cv2.addWeighted(frame, 1, canvas, 0.8, 0)
    cv2.imshow("Q7 Virtual Whiteboard", output)
    cv2.imshow("Tracking Mask", mask)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        canvas[:] = 0
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
