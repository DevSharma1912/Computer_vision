import cv2
import pytesseract
import os

# If Tesseract is not on PATH, uncomment and adjust this line:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

last_text = ""
frame_count = 0
print("Point camera at printed text. Press S to save OCR text; Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # OCR every 10 frames to reduce processing load
    if frame_count % 10 == 0:
        processed = cv2.threshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        try:
            last_text = pytesseract.image_to_string(processed, config="--psm 6").strip()
        except pytesseract.TesseractNotFoundError:
            last_text = "Tesseract not installed or not configured."
    preview = last_text[:90].replace("\n", " ")
    cv2.putText(frame, "OCR updates every 10 frames", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if preview:
        cv2.putText(frame, preview, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)
    cv2.imshow("Q8 Live OCR", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        with open("recognized_text.txt", "w", encoding="utf-8") as f:
            f.write(last_text)
        print("Saved recognized_text.txt")
        print(last_text)
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
