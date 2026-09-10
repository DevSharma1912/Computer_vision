# ================================================================
# EXPERIMENT NO. 6
# IMPLEMENTATION AND COMPARATIVE ANALYSIS OF IMAGE SEGMENTATION
# TECHNIQUES USING PYTHON AND OPENCV
# ================================================================

# Name       : Dev Sharma
# UID        : CU24250269
# Course     : B.Tech CSE
# Section    : CSE "B"
# Roll No.   : 17

# ================================================================
# AIM
# ================================================================
# To implement and evaluate various image segmentation techniques
# for partitioning digital images into meaningful regions, thereby
# facilitating object localization and scene understanding.

# ================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.cluster import KMeans

# ================================================================
# STEP 2: UPLOAD IMAGE FROM COMPUTER
# ================================================================

root = Tk()
root.withdraw()

file_path = askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")
    ]
)

if not file_path:
    print("No image selected.")
    exit()

image = cv2.imread(file_path)

if image is None:
    print("Unable to load image.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded successfully!")
print("Image size:", image.shape)

# ================================================================
# STEP 3: CONVERT IMAGE TO GRAYSCALE AND APPLY GAUSSIAN BLUR
# ================================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blurred Image")
plt.axis("off")

plt.tight_layout()
plt.show()

# ================================================================
# STEP 4: GLOBAL THRESHOLDING
# ================================================================

_, global_threshold = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

plt.figure(figsize=(6, 5))
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")
plt.show()

# ================================================================
# STEP 5: OTSU'S THRESHOLDING
# ================================================================

otsu_value, otsu_threshold = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Otsu threshold value:", round(otsu_value, 2))

plt.figure(figsize=(6, 5))
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu's Thresholding")
plt.axis("off")
plt.show()

# ================================================================
# STEP 6: ADAPTIVE THRESHOLDING
# ================================================================

adaptive_threshold = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

plt.figure(figsize=(6, 5))
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")
plt.show()

# ================================================================
# STEP 7: WATERSHED SEGMENTATION
# ================================================================

# Convert image to binary using Otsu
_, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Remove noise using morphological opening
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Find sure background
sure_bg = cv2.dilate(
    opening,
    kernel,
    iterations=3
)

# Find distance from background
dist_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

# Find sure foreground
_, sure_fg = cv2.threshold(
    dist_transform,
    0.7 * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

# Find unknown region
unknown = cv2.subtract(
    sure_bg,
    sure_fg
)

# Create markers
_, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

# Apply Watershed
watershed_image = image.copy()

markers = cv2.watershed(
    watershed_image,
    markers
)

watershed_image[markers == -1] = [255, 0, 0]

watershed_rgb = cv2.cvtColor(
    watershed_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(8, 6))
plt.imshow(watershed_rgb)
plt.title("Watershed Segmentation")
plt.axis("off")
plt.show()

# ================================================================
# STEP 8: K-MEANS COLOR IMAGE SEGMENTATION
# ================================================================

# Resize image for faster processing
small_image = cv2.resize(
    image_rgb,
    (300, 300)
)

pixels = small_image.reshape(
    (-1, 3)
)

pixels = np.float32(pixels)

# Apply K-Means
k = 3

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(pixels)

centers = np.uint8(
    kmeans.cluster_centers_
)

segmented_pixels = centers[labels]

segmented_image = segmented_pixels.reshape(
    small_image.shape
)

plt.figure(figsize=(6, 5))
plt.imshow(segmented_image)
plt.title("K-Means Segmentation")
plt.axis("off")
plt.show()

# ================================================================
# STEP 9: COMPARISON OF ALL SEGMENTATION TECHNIQUES
# ================================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu's Thresholding")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(watershed_rgb)
plt.title("Watershed")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(segmented_image)
plt.title("K-Means")
plt.axis("off")

plt.tight_layout()
plt.show()

# ================================================================
# STEP 10: DISPLAY IMAGE HISTOGRAM
# ================================================================

plt.figure(figsize=(8, 5))

plt.hist(
    gray.ravel(),
    bins=256,
    range=(0, 256)
)

plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# ================================================================
# STEP 11: OBSERVATIONS
# ================================================================

# Global Thresholding:
# Uses a single threshold value for the entire image.
# It works well when foreground and background have clear intensity
# differences.

# Otsu's Thresholding:
# Automatically selects the threshold value from the image histogram.
# It is useful when the image has two dominant intensity regions.

# Adaptive Thresholding:
# Calculates different threshold values for different local regions.
# It performs better when illumination is uneven.

# Watershed:
# Treats the image as a topographic surface and separates touching
# or overlapping objects using marker-based regions.

# K-Means:
# Groups pixels into clusters according to their color or intensity.
# It is useful for color-based image segmentation.

# ================================================================
# QUESTIONS AND ANSWERS
# ================================================================

# Q1. What is image segmentation? Why is it considered a fundamental
# step in computer vision?
#
# Image segmentation is the process of dividing an image into
# meaningful regions or objects based on pixel intensity, color,
# texture, or boundaries.
#
# It is fundamental because it separates regions of interest from
# the background and makes later computer vision tasks easier.

# Q2. Differentiate between Image Segmentation and Image
# Classification.
#
# Image Segmentation divides an image into different regions or
# individual objects at pixel level.
#
# Image Classification assigns a complete image to a particular
# category or class.
#
# Example:
# Segmentation identifies the pixels belonging to a car.
# Classification identifies the image as a car image.

# Q3. Explain the working principle of Global Thresholding,
# Otsu's Thresholding, and Adaptive Thresholding.
#
# Global Thresholding uses one fixed threshold value for the complete
# image to separate foreground and background.
#
# Otsu's Thresholding automatically calculates an optimal threshold
# from the image histogram.
#
# Adaptive Thresholding calculates different threshold values for
# different local regions of the image, making it useful for images
# with varying illumination.

# Q4. What is the Watershed Algorithm? Why is it useful for separating
# overlapping objects?
#
# Watershed is a region-based segmentation algorithm that treats an
# image like a topographic surface.
#
# It is useful for separating touching or overlapping objects because
# marker regions can be used to identify individual objects.

# Q5. How is K-Means Clustering applied to image segmentation?
#
# Image pixels are treated as data points.
# K-Means groups pixels into K clusters according to their color or
# intensity.
#
# Each pixel is then replaced by the color of its cluster centroid,
# producing the segmented image.

# Q6. Compare threshold-based segmentation and clustering-based
# segmentation techniques.
#
# Threshold-based segmentation separates pixels using threshold
# values. It is simple and computationally efficient.
#
# Clustering-based segmentation groups pixels according to similarity
# in color or intensity and can handle multiple regions more flexibly.
#
# Thresholding is generally faster, while clustering can provide
# better segmentation for complex color images.

# Q7. What challenges are encountered while segmenting images with
# complex backgrounds or varying illumination?
#
# Major challenges include:
# - Uneven illumination
# - Noise
# - Similar foreground and background colors
# - Complex backgrounds
# - Object overlap
# - Shadows and reflections
#
# Adaptive thresholding, preprocessing, Watershed and clustering can
# help overcome some of these challenges.

# Q8. Mention five real-world applications where image segmentation
# plays a critical role.
#
# 1. Medical image analysis
# 2. Autonomous vehicle navigation
# 3. Satellite image analysis
# 4. Industrial inspection
# 5. Object detection and recognition

# Q9. How does image segmentation improve the performance of object
# detection and image recognition systems?
#
# Segmentation isolates important objects or regions from the
# background.
#
# This reduces irrelevant information and allows detection and
# recognition systems to focus on the region of interest, improving
# accuracy and efficiency.

# Q10. Compare traditional image segmentation techniques with
# deep learning-based segmentation methods such as U-Net and
# Mask R-CNN.
#
# Traditional techniques such as thresholding, Watershed and K-Means
# are generally simple, fast and do not require large training
# datasets.
#
# Deep learning methods such as U-Net and Mask R-CNN can learn complex
# features and usually provide better segmentation for difficult
# real-world images.
#
# However, deep learning methods require large labelled datasets,
# greater computational resources and more training time.

# ================================================================
# END OF EXPERIMENT
# ================================================================