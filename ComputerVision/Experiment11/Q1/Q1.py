import cv2
import time

# 0 = laptop webcam
# Replace 0 with your mobile IP camera URL if required
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

previous_frame = None
last_snapshot_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # First frame
    if previous_frame is None:
        previous_frame = gray
        continue

    # Find difference between current and previous frame
    difference = cv2.absdiff(previous_frame, gray)

    # Convert difference into black/white
    _, threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

    # Make detected areas larger
    threshold = cv2.dilate(threshold, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:

        # Ignore very small movements/noise
        if cv2.contourArea(contour) < 500:
            continue

        motion_detected = True

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle around moving object
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )

    # Display alert
    if motion_detected:

        cv2.putText(
            frame,
            "MOTION DETECTED!",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Save snapshot only once every 2 seconds
        current_time = time.time()

        if current_time - last_snapshot_time > 2:
            filename = f"motion_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)

            print("Motion detected!")
            print("Snapshot saved:", filename)

            last_snapshot_time = current_time

    else:
        cv2.putText(
            frame,
            "No Motion",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Show result
    cv2.imshow("Motion Detector", frame)

    # Save current frame for next comparison
    previous_frame = gray

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()