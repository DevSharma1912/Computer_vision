import cv2
import mediapipe as mp
import pyautogui
import time

# ==========================================
# MediaPipe Hand Landmarker
# ==========================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

# Create MediaPipe options
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# ==========================================
# Camera
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    exit()


# ==========================================
# Finger Counting Function
# ==========================================

def count_fingers(landmarks, hand_type):

    count = 0

    # --------------------------
    # Thumb
    # --------------------------

    if hand_type == "Right":
        if landmarks[4].x < landmarks[3].x:
            count += 1

    else:
        if landmarks[4].x > landmarks[3].x:
            count += 1


    # --------------------------
    # Other four fingers
    # --------------------------

    fingers = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18)   # Little
    ]

    for tip, pip in fingers:

        if landmarks[tip].y < landmarks[pip].y:
            count += 1

    return count


# ==========================================
# Gesture Control
# ==========================================

last_action = 0
cooldown = 1.0

timestamp_ms = 0


# ==========================================
# Main Loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame")
        break


    # Mirror the camera
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Increase timestamp
    timestamp_ms += 33


    # Detect hand
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ======================================
    # If a hand is detected
    # ======================================

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        hand_type = "Right"

        if result.handedness:
            hand_type = result.handedness[0][0].category_name


        # Count fingers
        fingers = count_fingers(
            landmarks,
            hand_type
        )


        # ==================================
        # Draw landmarks
        # ==================================

        h, w, _ = frame.shape

        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # ==================================
        # Display finger count
        # ==================================

        cv2.putText(
            frame,
            f"Fingers: {fingers}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Hand: {hand_type}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        # ==================================
        # Gesture Actions
        # ==================================

        current_time = time.time()


        if current_time - last_action > cooldown:

            # 0 fingers → Volume Down
            if fingers == 0:

                pyautogui.press("volumedown")

                print("Gesture: Volume Down")

                last_action = current_time


            # 1 finger → Space
            elif fingers == 1:

                pyautogui.press("space")

                print("Gesture: SPACE")

                last_action = current_time


            # 2 fingers → Right Arrow
            elif fingers == 2:

                pyautogui.press("right")

                print("Gesture: Next / Right")

                last_action = current_time


            # 5 fingers → Volume Up
            elif fingers == 5:

                pyautogui.press("volumeup")

                print("Gesture: Volume Up")

                last_action = current_time


    else:

        cv2.putText(
            frame,
            "No Hand Detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ==========================================
    # Show Camera
    # ==========================================

    cv2.imshow(
        "Hand Gesture Desktop Controller",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()