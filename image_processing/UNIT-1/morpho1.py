import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1. Load image
# -----------------------------------------------------------
# Read image in grayscale (0 means grayscale mode)
img = cv2.imread("sample_image.png", 0)

if img is None:
    raise FileNotFoundError("⚠️ Image not found. Place 'sample_image.png' in the same folder.")

# -----------------------------------------------------------
# 2. Morphological Opening & Closing
# -----------------------------------------------------------
# Create a structuring element (kernel)
kernel = np.ones((5, 5), np.uint8)

# Opening: erosion followed by dilation (removes small white noise)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing: dilation followed by erosion (fills small black holes)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# -----------------------------------------------------------
# 3. Defect Segmentation (Thresholding)
# -----------------------------------------------------------
# Otsu's Thresholding (automatic global threshold)
_, otsu_thresh = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Adaptive Thresholding (better for uneven lighting)
adaptive_thresh = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# -----------------------------------------------------------
# 4. Feature Extraction (SIFT & ORB)
# -----------------------------------------------------------
# ---- SIFT (Scale-Invariant Feature Transform) ----
# Detects keypoints & descriptors for image matching/alignment
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, keypoints_sift, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ---- ORB (Oriented FAST and Rotated BRIEF) ----
# A free alternative to SIFT (since SIFT was patented earlier)
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0,255,0),
                            flags=0)

# -----------------------------------------------------------
# 5. Display Results
# -----------------------------------------------------------
plt.figure(figsize=(12,8))

# Original
plt.subplot(2,3,1), plt.imshow(img, cmap='gray')
plt.title("Original"), plt.axis("off")

# Morphological Opening
plt.subplot(2,3,2), plt.imshow(opening, cmap='gray')
plt.title("Opening (remove noise)"), plt.axis("off")

# Morphological Closing
plt.subplot(2,3,3), plt.imshow(closing, cmap='gray')
plt.title("Closing (fill holes)"), plt.axis("off")

# Otsu Thresholding
plt.subplot(2,3,4), plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu Thresholding"), plt.axis("off")

# Adaptive Thresholding
plt.subplot(2,3,5), plt.imshow(adaptive_thresh, cmap='gray')
plt.title("Adaptive Thresholding"), plt.axis("off")

# SIFT Features
plt.subplot(2,3,6), plt.imshow(img_sift)
plt.title(f"SIFT Features ({len(keypoints_sift)} points)"), plt.axis("off")

plt.tight_layout()
plt.show()

# Also show ORB features separately
plt.figure(figsize=(6,6))
plt.imshow(img_orb)
plt.title(f"ORB Features ({len(keypoints_orb)} points)")
plt.axis("off")
plt.show()
