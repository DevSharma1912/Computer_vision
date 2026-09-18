# ================================================================
# EXPERIMENT NO. 9
# OBJECT DETECTION AND RECOGNITION USING DEEP LEARNING MODELS
# ON STANDARD IMAGE DATASETS
# ================================================================

# Name       : Dev Sharma
# UID        : CU24250269
# Course     : B.Tech CSE
# Section    : CSE "B"
# Roll No.   : 17


# ================================================================
# AIM
# ================================================================

# To implement a deep learning-based object detection and
# recognition system using a pre-trained model and evaluate
# its performance for identifying and localizing multiple
# objects in an image.


# ================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog


# ================================================================
# STEP 2: LOAD PRE-TRAINED YOLO MODEL
# ================================================================

# Load the pre-trained YOLOv8 model.
# YOLOv8 is trained on the COCO dataset.

model = YOLO("yolov8n.pt")


# ================================================================
# STEP 3: SELECT IMAGE FROM COMPUTER
# ================================================================

root = tk.Tk()
root.withdraw()

filename = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
    ]
)

if not filename:
    raise ValueError("No image selected.")

image = cv2.imread(filename)

if image is None:
    raise ValueError("Unable to read the selected image.")

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
# ================================================================
# STEP 4: DISPLAY ORIGINAL IMAGE
# ================================================================

plt.figure(figsize=(8, 6))

plt.imshow(image_rgb)

plt.title("Original Image")

plt.axis("off")

plt.show()


# ================================================================
# STEP 5: IMAGE PREPROCESSING
# ================================================================

# Resize the image for processing.

resized_image = cv2.resize(
    image_rgb,
    (640, 640)
)


# ================================================================
# STEP 6: DISPLAY PREPROCESSED IMAGE
# ================================================================

plt.figure(figsize=(8, 6))

plt.imshow(resized_image)

plt.title("Preprocessed Image")

plt.axis("off")

plt.show()


# ================================================================
# STEP 7: PERFORM OBJECT DETECTION
# ================================================================

# Apply the pre-trained YOLO model to the image.

results = model(image_rgb)

result = results[0]


# ================================================================
# STEP 8: EXTRACT DETECTED OBJECTS
# ================================================================

boxes = result.boxes

detected_objects = []

for box in boxes:

    class_id = int(box.cls[0])

    confidence = float(box.conf[0])

    class_name = model.names[class_id]

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    detected_objects.append(
        [
            class_name,
            confidence,
            x1,
            y1,
            x2,
            y2
        ]
    )


# ================================================================
# STEP 9: DISPLAY DETECTED OBJECTS
# ================================================================

# Display bounding boxes, class labels and confidence scores.

annotated_image = result.plot()

annotated_image = cv2.cvtColor(
    annotated_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(10, 8))

plt.imshow(annotated_image)

plt.title(
    "YOLO Object Detection"
)

plt.axis("off")

plt.show()


# ================================================================
# STEP 10: COUNT DETECTED OBJECTS
# ================================================================

object_counts = {}

for obj in detected_objects:

    class_name = obj[0]

    if class_name in object_counts:

        object_counts[class_name] += 1

    else:

        object_counts[class_name] = 1


# ================================================================
# STEP 11: DISPLAY OBJECT COUNT
# ================================================================

plt.figure(figsize=(10, 5))

if len(object_counts) > 0:

    plt.bar(
        object_counts.keys(),
        object_counts.values()
    )

    plt.xlabel("Object Class")

    plt.ylabel("Number of Objects")

    plt.title(
        "Detected Object Count"
    )

    plt.xticks(rotation=45)

else:

    plt.text(
        0.5,
        0.5,
        "No Objects Detected",
        ha="center",
        va="center"
    )

    plt.axis("off")

plt.show()


# ================================================================
# STEP 12: CALCULATE DETECTION STATISTICS
# ================================================================

total_objects = len(detected_objects)

if total_objects > 0:

    average_confidence = np.mean(
        [
            obj[1]
            for obj in detected_objects
        ]
    )

else:

    average_confidence = 0

inference_time = result.speed["inference"]


# ================================================================
# STEP 13: DISPLAY DETECTION SUMMARY
# ================================================================

# Detection results include:
# - Total number of detected objects
# - Average confidence score
# - Inference time

summary = [
    ["Total Objects Detected", total_objects],
    ["Average Confidence", round(average_confidence, 3)],
    ["Inference Time (ms)", round(inference_time, 2)]
]

fig, ax = plt.subplots(figsize=(7, 2.5))

ax.axis("off")

table = ax.table(
    cellText=summary,
    colLabels=["Parameter", "Value"],
    loc="center"
)

table.auto_set_font_size(False)

table.set_fontsize(11)

table.scale(1, 1.8)

plt.title(
    "Object Detection Summary"
)

plt.show()


# ================================================================
# STEP 14: EVALUATE DETECTION RESULTS
# ================================================================

# Correct Detection:
# An object is correctly detected when its class and location
# are identified correctly.

# False Detection:
# A false detection occurs when the model identifies an object
# incorrectly or detects an object where it does not exist.

# Missed Detection:
# A missed detection occurs when an object present in the image
# is not detected by the model.


# ================================================================
# STEP 15: ANALYZE MODEL PERFORMANCE
# ================================================================

# YOLO is a single-stage object detection model.
#
# It processes the image and predicts bounding boxes,
# class labels and confidence scores.
#
# YOLO provides fast inference and can be used for
# applications requiring quick object detection.
#
# Detection performance can be affected by:
# - Lighting conditions
# - Object size
# - Occlusion
# - Complex backgrounds
# - Image quality
# - Overlapping objects


# ================================================================
# STEP 16: REAL-WORLD APPLICATIONS
# ================================================================

# Object detection can be used in:
#
# 1. Autonomous vehicles
# 2. Surveillance systems
# 3. Healthcare
# 4. Retail analytics
# 5. Industrial inspection
#
# These applications are listed in the experiment description.
# :contentReference[oaicite:1]{index=1}


# ================================================================
# STEP 17: OBSERVATIONS
# ================================================================

# Observation 1:
# The pre-trained YOLO model successfully detects objects
# present in the uploaded image.
#
# Observation 2:
# Bounding boxes identify the location of detected objects.
#
# Observation 3:
# Class labels identify the category of each detected object.
#
# Observation 4:
# Confidence scores indicate the model's confidence
# for each detection.
#
# Observation 5:
# Multiple objects can be detected from a single image.
#
# Observation 6:
# Detection performance can be affected by lighting,
# object size, occlusion and complex backgrounds.
#
# Observation 7:
# Pre-trained deep learning models can detect complex
# visual patterns without manually designing image features.
#
# Observation 8:
# Object detection combines object localization and
# object classification.


# ================================================================
# QUESTIONS AND ANSWERS
# ================================================================

# Q1. Differentiate between image classification,
#     object detection and image segmentation.
#
# Answer:
# Image classification assigns a class label to the complete image.
#
# Object detection identifies objects and provides their
# class labels and bounding boxes.
#
# Image segmentation assigns labels to individual pixels
# or regions of an image.


# ---------------------------------------------------------------

# Q2. Explain the working principle of YOLO.
#
# Answer:
# YOLO stands for You Only Look Once.
#
# It processes the complete image using a deep learning
# neural network and predicts bounding boxes, class labels
# and confidence scores.
#
# YOLO performs detection in a single main detection process,
# which makes it suitable for fast object detection.


# ---------------------------------------------------------------

# Q3. Compare YOLO, SSD and Faster R-CNN based on speed,
#     accuracy and practical applications.
#
# Answer:
# YOLO and SSD are single-stage object detection methods
# designed for fast detection.
#
# Faster R-CNN is a two-stage object detection method that
# first generates region proposals and then performs
# classification and localization.
#
# YOLO and SSD are commonly suitable for applications where
# fast inference is important, while Faster R-CNN uses a
# more computationally intensive two-stage approach.


# ---------------------------------------------------------------

# Q4. What is a bounding box? Why is it important?
#
# Answer:
# A bounding box is a rectangular box surrounding a detected
# object in an image.
#
# It is important because it provides the location and
# approximate size of the detected object.


# ---------------------------------------------------------------

# Q5. Explain the significance of confidence score and IoU.
#
# Answer:
# A confidence score represents the model's confidence
# in a particular object detection.
#
# IoU stands for Intersection over Union.
#
# It measures the overlap between the predicted bounding
# box and the ground-truth bounding box.
#
# A higher IoU indicates greater overlap between the
# predicted and actual bounding boxes.


# ---------------------------------------------------------------

# Q6. What are the advantages of using pre-trained deep
#     learning models over traditional image processing?
#
# Answer:
# Pre-trained deep learning models have already learned
# useful visual features from large datasets.
#
# They reduce the need to train a complete model from
# the beginning and can identify complex patterns.
#
# Traditional image processing generally depends more
# on manually designed features and rules.


# ---------------------------------------------------------------

# Q7. Why are datasets such as MS COCO and Pascal VOC
#     widely used?
#
# Answer:
# MS COCO and Pascal VOC contain labeled images with
# different object categories and localization information.
#
# They are widely used for training, testing and
# evaluating object detection algorithms.


# ---------------------------------------------------------------

# Q8. Mention five real-world applications of object detection.
#
# Answer:
#
# 1. Autonomous vehicles
# 2. Surveillance systems
# 3. Healthcare
# 4. Retail analytics
# 5. Industrial inspection


# ---------------------------------------------------------------

# Q9. What challenges are commonly encountered while
#     detecting objects in complex real-world environments?
#
# Answer:
#
# Common challenges include:
#
# 1. Poor lighting
# 2. Object occlusion
# 3. Overlapping objects
# 4. Small objects
# 5. Complex backgrounds
# 6. Different object sizes
# 7. Camera movement
# 8. Low-quality images


# ---------------------------------------------------------------

# Q10. How can object detection systems be further improved?
#
# Answer:
#
# Object detection systems can be improved using:
#
# 1. Larger and more diverse datasets
# 2. Data augmentation
# 3. Improved deep learning architectures
# 4. Better feature extraction
# 5. Improved training techniques
# 6. Model optimization
# 7. Better hardware
# 8. Recent computer vision techniques


# ================================================================
# RESULT
# ================================================================

# The object detection and recognition system was successfully
# implemented using a pre-trained YOLO deep learning model.
#
# Objects were identified and localized using bounding boxes,
# class labels and confidence scores.
#
# The experiment demonstrated the practical application of
# deep learning-based object detection for identifying and
# localizing multiple objects in images.
