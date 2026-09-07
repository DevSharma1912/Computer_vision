# Name :- Dev Sharma
# CU24250269
# BTech CSE 3rd Year SECTION - "A"
# Roll No. :- 17

# ============================================================
# Experiment No. 2
# Experiment Name
# Contrast Enhancement and Histogram-Based Image Processing
# using Python and OpenCV
# ============================================================

# Aim:
# To analyze and implement contrast enhancement techniques
# using histogram analysis and histogram equalization in
# Python and OpenCV for improving image quality and enhancing
# visual information in low-contrast images.


# ============================================================
# Step 1: Import Required Libraries
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog


# ============================================================
# Step 2: Upload / Select Image from Computer
# ============================================================

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Low-Contrast Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
        ("All Files", "*.*")
    ]
)

if not image_path:
    print("No image selected.")
    exit()

# Load image
image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("Error: Unable to load the selected image.")
    exit()

print("Image loaded successfully.")
print("Selected Image:", image_path)
print("Image Shape:", image.shape)


# ============================================================
# Step 3: Display Original Image
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    image,
    cmap="gray"
)

plt.title("Original Grayscale Image")
plt.axis("off")

plt.show()

print("Original image displayed successfully.")


# ============================================================
# Step 4: Generate Histogram of Original Image
# ============================================================

original_histogram = cv2.calcHist(
    [image],
    [0],
    None,
    [256],
    [0, 256]
)

plt.figure(figsize=(8, 5))

plt.plot(
    original_histogram
)

plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.show()

print("Original image histogram generated successfully.")


# ============================================================
# Step 5: Perform Contrast Stretching
# ============================================================

min_value = np.min(image)
max_value = np.max(image)

if max_value == min_value:
    contrast_stretched = image.copy()
else:
    contrast_stretched = (
        (image - min_value)
        * 255.0
        / (max_value - min_value)
    )

contrast_stretched = np.uint8(
    contrast_stretched
)

print("\n========== CONTRAST STRETCHING ==========")

print("Minimum Pixel Intensity:", min_value)
print("Maximum Pixel Intensity:", max_value)
print("Contrast Stretching: Completed")


# ============================================================
# Step 6: Apply Histogram Equalization
# ============================================================

histogram_equalized = cv2.equalizeHist(
    image
)

print("Histogram Equalization: Completed")


# ============================================================
# Step 7: Apply CLAHE
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(
    image
)

print("CLAHE Enhancement: Completed")


# ============================================================
# Step 8: Generate Histograms of Enhanced Images
# ============================================================

equalized_histogram = cv2.calcHist(
    [histogram_equalized],
    [0],
    None,
    [256],
    [0, 256]
)

clahe_histogram = cv2.calcHist(
    [clahe_image],
    [0],
    None,
    [256],
    [0, 256]
)

print("Enhanced image histograms generated successfully.")


# ============================================================
# Step 9: Display Original and Enhanced Images
# ============================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(
    image,
    cmap="gray"
)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(
    contrast_stretched,
    cmap="gray"
)
plt.title("Contrast Stretched")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(
    histogram_equalized,
    cmap="gray"
)
plt.title("Histogram Equalized")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(
    clahe_image,
    cmap="gray"
)
plt.title("CLAHE Enhanced")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Original and enhanced images displayed successfully.")


# ============================================================
# Step 10: Compare Histograms
# ============================================================

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)

plt.plot(
    original_histogram
)

plt.title("Original Image Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])


plt.subplot(3, 1, 2)

plt.plot(
    equalized_histogram
)

plt.title("Histogram Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])


plt.subplot(3, 1, 3)

plt.plot(
    clahe_histogram
)

plt.title("CLAHE Enhanced Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

print("Histogram comparison completed successfully.")


# ============================================================
# Step 11: Calculate Image Statistics
# ============================================================

print("\n========== IMAGE STATISTICS ==========")

print(
    "Original Mean Intensity:",
    round(np.mean(image), 2)
)

print(
    "Contrast Stretched Mean Intensity:",
    round(np.mean(contrast_stretched), 2)
)

print(
    "Histogram Equalized Mean Intensity:",
    round(np.mean(histogram_equalized), 2)
)

print(
    "CLAHE Mean Intensity:",
    round(np.mean(clahe_image), 2)
)

print(
    "\nOriginal Intensity Range:",
    np.min(image),
    "to",
    np.max(image)
)

print(
    "Contrast Stretched Range:",
    np.min(contrast_stretched),
    "to",
    np.max(contrast_stretched)
)

print(
    "Histogram Equalized Range:",
    np.min(histogram_equalized),
    "to",
    np.max(histogram_equalized)
)

print(
    "CLAHE Range:",
    np.min(clahe_image),
    "to",
    np.max(clahe_image)
)


# ============================================================
# Step 12: Final Output
# ============================================================

print("\n========== CONTRAST ENHANCEMENT ==========")

print("Original Histogram: Generated")
print("Contrast Stretching: Completed")
print("Histogram Equalization: Completed")
print("CLAHE Enhancement: Completed")
print("Histogram Comparison: Completed")

print("\nObservations:")

print("Contrast stretching expands the dynamic range of pixel intensities.")

print("Histogram Equalization improves the overall contrast by")
print("redistributing pixel intensity values.")

print("CLAHE improves local contrast and is useful for images")
print("with varying illumination.")

print("CLAHE also limits excessive contrast enhancement by")
print("clipping the histogram in local regions.")

print("\nExperiment Completed Successfully.")


# ============================================================
# EXPERIMENT 2 - ANSWERS TO QUESTIONS
# ============================================================


# ------------------------------------------------------------
# QUESTION 1:
# What is image contrast, and why is it important in image
# processing?
# ------------------------------------------------------------

# Image contrast refers to the difference in intensity between
# different regions or objects in an image.
#
# An image with high contrast has clearly distinguishable
# bright and dark regions.
#
# A low-contrast image has pixel intensities concentrated
# within a small range, making details difficult to observe.
#
# Contrast is important because it improves:
#
# 1. Image visibility.
# 2. Feature visibility.
# 3. Object boundaries.
# 4. Feature extraction.
# 5. Object recognition.
#
# Contrast enhancement can therefore make important image
# information easier to analyze.


# ------------------------------------------------------------
# QUESTION 2:
# Explain the concept of an image histogram. What information
# does it provide?
# ------------------------------------------------------------

# An image histogram is a graphical representation of the
# distribution of pixel intensity values in an image.
#
# For an 8-bit grayscale image, intensity values range from
# 0 to 255.
#
# The horizontal axis represents pixel intensity.
#
# The vertical axis represents the number of pixels having
# each intensity value.
#
# A histogram provides information about:
#
# 1. Brightness distribution.
# 2. Contrast.
# 3. Dark and bright regions.
# 4. Distribution of pixel intensities.
# 5. Possible underexposure or overexposure.
#
# Histogram analysis is useful for selecting appropriate
# contrast enhancement techniques.


# ------------------------------------------------------------
# QUESTION 3:
# Differentiate between Histogram Stretching and Histogram
# Equalization.
# ------------------------------------------------------------

# Histogram Stretching:
#
# Histogram stretching expands the existing range of pixel
# intensities to use a larger portion of the available
# intensity range.
#
# It improves contrast by mapping the minimum and maximum
# intensity values to a wider range.
#
#
# Histogram Equalization:
#
# Histogram equalization redistributes pixel intensities
# using the cumulative distribution of the histogram.
#
# It attempts to spread intensity values more evenly over
# the available range.
#
#
# Main Difference:
#
# Histogram Stretching:
# -> Expands the existing intensity range.
#
# Histogram Equalization:
# -> Redistributes pixel intensities to improve contrast.
#
# Stretching is generally simpler, while equalization can
# produce stronger contrast enhancement.


# ------------------------------------------------------------
# QUESTION 4:
# What is Contrast Limited Adaptive Histogram Equalization
# (CLAHE)? How does it differ from standard Histogram
# Equalization?
# ------------------------------------------------------------

# CLAHE stands for Contrast Limited Adaptive Histogram
# Equalization.
#
# CLAHE divides an image into small local regions called tiles
# and performs histogram equalization separately in each region.
#
# It also uses a clip limit to prevent excessive contrast
# enhancement.
#
#
# Standard Histogram Equalization:
#
# Standard Histogram Equalization works on the entire image
# using a global histogram.
#
#
# CLAHE:
#
# CLAHE performs local contrast enhancement and limits
# excessive amplification.
#
# Therefore:
#
# Standard HE -> Global contrast enhancement.
# CLAHE       -> Local and contrast-limited enhancement.
#
# CLAHE is particularly useful when illumination varies
# across different regions of an image.


# ------------------------------------------------------------
# QUESTION 5:
# Why is histogram equalization commonly applied before
# feature extraction and image segmentation?
# ------------------------------------------------------------

# Histogram equalization can improve image contrast and make
# important structures more visible.
#
# Better contrast can make boundaries and intensity differences
# easier to identify.
#
# This can help feature extraction algorithms detect useful
# patterns and can make segmentation more effective.
#
# It can therefore be useful as a preprocessing step before:
#
# 1. Feature extraction.
# 2. Image segmentation.
# 3. Object detection.
# 4. Pattern recognition.
#
# However, the enhancement method should be selected according
# to the characteristics of the image.


# ------------------------------------------------------------
# QUESTION 6:
# Mention three real-world applications where histogram
# equalization is widely used.
# ------------------------------------------------------------

# Three real-world applications are:
#
# 1. Medical Imaging:
#    Histogram-based enhancement can improve visibility of
#    structures in medical images.
#
# 2. Satellite and Remote Sensing Images:
#    Contrast enhancement can improve the visibility of
#    features in satellite imagery.
#
# 3. Low-Light Photography:
#    Histogram enhancement can improve the visibility of
#    details in dark or poorly illuminated photographs.
#
# These applications use contrast enhancement to make
# important visual information easier to analyze.


# ------------------------------------------------------------
# QUESTION 7:
# What are the limitations of global histogram equalization?
# ------------------------------------------------------------

# Global Histogram Equalization uses a single histogram for
# the entire image.
#
# Its limitations include:
#
# 1. It may over-enhance noise.
# 2. It may produce unnatural contrast.
# 3. It may lose local details.
# 4. It does not handle varying illumination very well.
# 5. It can produce excessive enhancement in some regions.
#
# CLAHE can address some of these limitations by performing
# enhancement locally and limiting excessive contrast.


# ------------------------------------------------------------
# QUESTION 8:
# How does contrast enhancement improve the performance of
# object detection and recognition systems?
# ------------------------------------------------------------

# Contrast enhancement improves the visibility of important
# structures and object boundaries.
#
# It can make features easier to distinguish from the
# background.
#
# Improved contrast can help:
#
# 1. Highlight object boundaries.
# 2. Improve feature visibility.
# 3. Reduce the effect of poor illumination.
# 4. Improve segmentation.
# 5. Provide better input for feature extraction.
#
# As a result, contrast enhancement can improve the quality
# of input images used by object detection and recognition
# systems.


# ------------------------------------------------------------
# QUESTION 9:
# Compare histogram-based enhancement techniques with
# brightness adjustment methods.
# ------------------------------------------------------------

# Histogram-Based Enhancement:
#
# Histogram techniques analyze and modify the distribution
# of pixel intensities.
#
# Examples:
# 1. Contrast Stretching
# 2. Histogram Equalization
# 3. CLAHE
#
# They primarily aim to improve contrast and the distribution
# of intensity values.
#
#
# Brightness Adjustment:
#
# Brightness adjustment generally shifts pixel intensity
# values upward or downward.
#
# Increasing brightness makes an image lighter.
# Decreasing brightness makes an image darker.
#
#
# Main Difference:
#
# Brightness Adjustment:
# -> Changes overall brightness.
#
# Histogram-Based Enhancement:
# -> Analyzes and modifies intensity distribution to improve
#    contrast.
#
# Therefore, histogram-based methods can provide more
# controlled contrast enhancement than simply changing
# brightness.


# ------------------------------------------------------------
# QUESTION 10:
# Why is CLAHE preferred for medical imaging and low-light
# image enhancement?
# ------------------------------------------------------------

# CLAHE is useful for medical imaging and low-light images
# because it enhances contrast locally.
#
# Different areas of an image may have different illumination
# levels.
#
# CLAHE processes local regions independently and uses a clip
# limit to prevent excessive contrast enhancement.
#
# Advantages include:
#
# 1. Improved local contrast.
# 2. Better visibility of local details.
# 3. Improved handling of varying illumination.
# 4. Reduced risk of excessive contrast amplification.
# 5. Better visibility of important structures.
#
# Therefore, CLAHE is particularly suitable for medical images
# and low-light images where local details are important.