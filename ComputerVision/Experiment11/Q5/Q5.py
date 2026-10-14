
import cv2
import numpy as np

def order_points(points):
    points = np.array(points, dtype="float32")
    result = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    result[0] = points[np.argmin(sums)]   # Top-left
    result[1] = points[np.argmin(diffs)]  # Top-right
    result[2] = points[np.argmax(sums)]   # Bottom-right
    result[3] = points[np.argmax(diffs)]  # Bottom-left

    return result


def scan_document(frame):
    height, width = frame.shape[:2]

    # Resize to make processing faster
    scale = 700 / height
    small = cv2.resize(
        frame, (int(width * scale), 700)
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve edge detection
    edges = cv2.Canny(blur, 30, 120)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(
        contours, key=cv2.contourArea, reverse=True
    )[:30]

    corners = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if (
            len(approx) == 4
            and cv2.contourArea(approx) > small.shape[0] * small.shape[1] * 0.05
        ):
            corners = approx.reshape(4, 2).astype("float32")
            break

    display = small.copy()

    if corners is None:
        cv2.putText(
            display, "Looking for document...",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )
        return display, edges, None

    # Show detected corners
    cv2.polylines(
        display, [corners.astype("int32")],
        True, (0, 255, 0), 3
    )

    # Convert corners back to original frame coordinates
    corners = order_points(corners / scale)
    tl, tr, br, bl = corners

    out_width = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))
    out_height = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    if out_width < 2 or out_height < 2:
        return display, edges, None

    destination = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        corners, destination
    )

    warped = cv2.warpPerspective(
        frame, matrix, (out_width, out_height)
    )

    gray_scan = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray_scan, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return display, edges, scanned


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening. Try cv2.VideoCapture(1).")
    raise SystemExit

latest_scan = None

print("Place the whole paper against a contrasting background.")
print("Green outline = document detected.")
print("S = save scan | Q = quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera.")
        break

    display, edges, latest_scan = scan_document(frame)

    cv2.imshow("Document Detection", display)
    cv2.imshow("Edge Detection", edges)

    if latest_scan is not None:
        preview_height = int(
            500 * latest_scan.shape[0] / latest_scan.shape[1]
        )
        preview = cv2.resize(latest_scan, (500, preview_height))
        cv2.imshow("Scanned Document", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if latest_scan is not None:
            cv2.imwrite("scanned_document.jpg", latest_scan)
            print("Saved scanned_document.jpg")
        else:
            print("No document detected yet.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
