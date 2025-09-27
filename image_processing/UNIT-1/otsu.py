import cv2
import matplotlib.pyplot as plt

# Load grayscale image (example: defected part)
img = cv2.imread("D:\\HCL\\HCL-Training\\image_processing\\003.png", 0)

if img is None:
    raise FileNotFoundError("⚠️ Please put 'defect_sample.png' in the same folder.")

# Otsu’s Thresholding (automatic global threshold)
_, otsu_thresh = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Adaptive Thresholding (works under uneven lighting)
adaptive_thresh = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# Display results
plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1, 3, 2), plt.imshow(otsu_thresh, cmap="gray"), plt.title("Otsu")
plt.subplot(1, 3, 3), plt.imshow(adaptive_thresh, cmap="gray"), plt.title("Adaptive")
plt.show()

