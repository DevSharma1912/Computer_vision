# ================================================================
# EXPERIMENT NO. 7
# MOTION ESTIMATION USING OPTICAL FLOW ALGORITHMS
# IN VIDEO SEQUENCES
# ================================================================

# Name       : Dev Sharma
# UID        : CU24250269
# Course     : B.Tech CSE
# Section    : CSE "A"
# Roll No.   : 17


# ================================================================
# AIM
# ================================================================

# To implement optical flow algorithms for estimating the motion
# of objects between consecutive video frames and analyze motion
# patterns for dynamic scene understanding using Python and OpenCV.


# ================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# ================================================================
# STEP 2: UPLOAD VIDEO FROM COMPUTER
# ================================================================

root = Tk()
root.withdraw()

file_path = askopenfilename(
    title="Select a Video",
    filetypes=[
        ("Video Files", "*.mp4 *.avi *.mov *.mkv")
    ]
)

if not file_path:
    print("No video selected.")
    exit()

video = cv2.VideoCapture(file_path)

if not video.isOpened():
    print("Unable to open video.")
    exit()


# ================================================================
# STEP 3: READ TWO CONSECUTIVE VIDEO FRAMES
# ================================================================

ret1, frame1 = video.read()
ret2, frame2 = video.read()

video.release()

if not ret1 or not ret2:
    print("Unable to read video frames.")
    exit()

# Convert frames to grayscale

gray1 = cv2.cvtColor(
    frame1,
    cv2.COLOR_BGR2GRAY
)

gray2 = cv2.cvtColor(
    frame2,
    cv2.COLOR_BGR2GRAY
)


# ================================================================
# STEP 4: DISPLAY CONSECUTIVE FRAMES
# ================================================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(
    cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
)
plt.title("First Frame")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
)
plt.title("Second Frame")
plt.axis("off")

plt.tight_layout()
plt.show()


# ================================================================
# STEP 5: LUCAS-KANADE SPARSE OPTICAL FLOW
# ================================================================

# Detect good feature points in the first frame

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

points_old = cv2.goodFeaturesToTrack(
    gray1,
    mask=None,
    **feature_params
)

if points_old is None:
    print("No feature points detected.")
    exit()


# Lucas-Kanade parameters

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)


# Calculate optical flow

points_new, status, error = cv2.calcOpticalFlowPyrLK(
    gray1,
    gray2,
    points_old,
    None,
    **lk_params
)


# Select successfully tracked points

good_new = points_new[status == 1]
good_old = points_old[status == 1]


# ================================================================
# STEP 6: VISUALIZE LUCAS-KANADE MOTION VECTORS
# ================================================================

lucas_kanade = frame2.copy()

for new, old in zip(good_new, good_old):

    x_new, y_new = new.ravel()
    x_old, y_old = old.ravel()

    x_new = int(x_new)
    y_new = int(y_new)

    x_old = int(x_old)
    y_old = int(y_old)

    # Draw motion arrow

    cv2.arrowedLine(
        lucas_kanade,
        (x_old, y_old),
        (x_new, y_new),
        (255, 0, 0),
        2,
        tipLength=0.3
    )

    # Draw feature point

    cv2.circle(
        lucas_kanade,
        (x_new, y_new),
        4,
        (0, 255, 0),
        -1
    )


# Display Lucas-Kanade result

plt.figure(figsize=(8, 6))

plt.imshow(
    cv2.cvtColor(
        lucas_kanade,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Lucas-Kanade Sparse Optical Flow")
plt.axis("off")
plt.show()


# ================================================================
# STEP 7: FARNEBACK DENSE OPTICAL FLOW
# ================================================================

# Calculate dense optical flow

flow = cv2.calcOpticalFlowFarneback(
    gray1,
    gray2,
    None,
    0.5,
    3,
    15,
    3,
    5,
    1.2,
    0
)


# ================================================================
# STEP 8: CALCULATE MOTION MAGNITUDE AND DIRECTION
# ================================================================

magnitude, angle = cv2.cartToPolar(
    flow[..., 0],
    flow[..., 1]
)


# ================================================================
# STEP 9: CREATE COLOR-CODED FARNEBACK OPTICAL FLOW
# ================================================================

hsv = np.zeros_like(frame1)

hsv[..., 1] = 255

# Direction of motion

hsv[..., 0] = (
    angle * 180 / np.pi / 2
)

# Magnitude of motion

hsv[..., 2] = cv2.normalize(
    magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


# Convert HSV to BGR

farneback = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2BGR
)


# ================================================================
# STEP 10: DISPLAY FARNEBACK DENSE OPTICAL FLOW
# ================================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    cv2.cvtColor(
        farneback,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Farneback Dense Optical Flow")
plt.axis("off")
plt.show()


# ================================================================
# STEP 11: COMPARISON OF OPTICAL FLOW METHODS
# ================================================================

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.imshow(
    cv2.cvtColor(
        lucas_kanade,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Lucas-Kanade Sparse Optical Flow")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    cv2.cvtColor(
        farneback,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Farneback Dense Optical Flow")
plt.axis("off")

plt.tight_layout()
plt.show()


# ================================================================
# STEP 12: OBSERVATIONS
# ================================================================

# Lucas-Kanade Sparse Optical Flow:
#
# Lucas-Kanade tracks selected feature points between consecutive
# frames.
#
# It produces motion vectors only for important feature points.
#
# It is computationally simpler and generally faster.


# Farneback Dense Optical Flow:
#
# Farneback calculates motion for almost every pixel in the image.
#
# It provides detailed information about the motion present in
# different regions of the frame.
#
# It generally requires more computational resources than
# Lucas-Kanade.


# Comparison:
#
# Lucas-Kanade is suitable for feature tracking and situations
# where only selected points need to be tracked.
#
# Farneback is suitable when complete motion information across
# the image is required.


# ================================================================
# QUESTIONS AND ANSWERS
# ================================================================

# Q1. What is Optical Flow? Mention its applications.
#
# Optical Flow is the apparent motion of pixels or objects between
# consecutive video frames.
#
# Applications include:
# 1. Object tracking
# 2. Video surveillance
# 3. Autonomous navigation
# 4. Motion analysis
# 5. Human action recognition


# Q2. Explain the basic principle of the Lucas-Kanade method.
#
# Lucas-Kanade estimates the motion of selected feature points
# between two consecutive frames.
#
# It assumes that the motion of pixels inside a small local window
# is approximately similar.


# Q3. Differentiate between Dense and Sparse Optical Flow.
#
# Sparse Optical Flow calculates motion only for selected feature
# points.
#
# Dense Optical Flow calculates motion for almost every pixel.
#
# Therefore, Sparse Optical Flow is generally faster while Dense
# Optical Flow provides more complete motion information.


# Q4. Compare Lucas-Kanade and Farneback Optical Flow.
#
# Lucas-Kanade is a Sparse Optical Flow method that tracks selected
# feature points.
#
# Farneback is a Dense Optical Flow method that estimates motion
# over the entire image.
#
# Lucas-Kanade is generally simpler and faster, while Farneback
# provides more detailed motion information.


# Q5. What assumptions are made in Optical Flow estimation?
#
# The main assumptions are:
#
# 1. Brightness of a moving point remains approximately constant.
# 2. Motion between consecutive frames is relatively small.
# 3. Motion is locally smooth or consistent.


# Q6. What factors affect the accuracy of Optical Flow?
#
# The accuracy can be affected by:
#
# 1. Lighting changes
# 2. Camera movement
# 3. Object speed
# 4. Image noise
# 5. Motion blur
# 6. Occlusion
# 7. Texture of the object
# 8. Video frame rate


# Q7. List five real-world applications of Optical Flow.
#
# 1. Autonomous driving
# 2. Video surveillance
# 3. Object tracking
# 4. Human action recognition
# 5. Motion detection and analysis


# Q8. Why are grayscale images commonly used for Optical Flow?
#
# Grayscale images contain intensity information that is sufficient
# for most optical flow calculations.
#
# Using grayscale also reduces computational complexity because
# three color channels do not need to be processed separately.


# Q9. What are the limitations of Optical Flow methods?
#
# Limitations include:
#
# 1. Sensitivity to lighting changes
# 2. Difficulty with very large motion
# 3. Problems caused by occlusion
# 4. Motion blur can reduce accuracy
# 5. Camera motion can affect estimation
# 6. Textureless regions are difficult to track


# Q10. How does Optical Flow contribute to autonomous driving,
# surveillance, and action recognition?
#
# In autonomous driving, Optical Flow helps estimate the movement
# of vehicles, pedestrians and other objects.
#
# In surveillance, it helps detect and track moving objects.
#
# In action recognition, motion patterns can be used to identify
# activities and human movements.


# ================================================================
# RESULT
# ================================================================

# Thus, Lucas-Kanade Sparse Optical Flow and Farneback Dense
# Optical Flow were successfully implemented using Python and
# OpenCV.
#
# Motion between consecutive video frames was estimated and
# visualized using motion vectors and color-coded optical flow.
#
# The two methods were compared based on their motion estimation
# approach, computational complexity and applications.