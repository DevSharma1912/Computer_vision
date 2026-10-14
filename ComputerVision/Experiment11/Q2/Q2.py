import cv2
import csv
from datetime import datetime

# 0 = laptop webcam
# Replace 0 with your mobile camera URL if using IP Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()

# Create QR detector
detector = cv2.QRCodeDetector()

# Open CSV file
with open("qr_inventory.csv", "a", newline="") as file:

    writer = csv.writer(file)

    # Add header if file is empty
    if file.tell() == 0:
        writer.writerow(["Item", "Timestamp"])

    last_data = ""

    while True:

        # Read camera frame
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame")
            break

        # Detect and decode QR code
        data, points, _ = detector.detectAndDecode(frame)

        # If QR code is detected
        if data:

            print("QR Code:", data)

            # Draw box around QR code
            if points is not None:

                points = points[0]

                for i in range(4):
                    start = tuple(map(int, points[i]))
                    end = tuple(map(int, points[(i + 1) % 4]))

                    cv2.line(
                        frame,
                        start,
                        end,
                        (0, 255, 0),
                        3
                    )

            # Prevent the same QR from being saved continuously
            if data != last_data:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Save to CSV
                writer.writerow([data, timestamp])

                file.flush()

                print("Saved to CSV")
                print("Time:", timestamp)

                last_data = data

            # Display decoded data
            cv2.putText(
                frame,
                "QR: " + data,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Searching for QR Code...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Show camera
        cv2.imshow("QR Code Scanner", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()