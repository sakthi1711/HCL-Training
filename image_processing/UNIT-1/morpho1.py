import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a binary mask (use any black/white image)
img = cv2.imread("bmask.png", 0)

if img is None:
    raise FileNotFoundError("Please put in the same folder.")

# Create a kernel (structuring element)
kernel = np.ones((15,15), np.uint8)

# Opening: removes small white noise
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing: fills small black holes
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Display results
plt.subplot(1, 3, 1), plt.imshow(img, cmap="gray"), plt.title("Original")
plt.subplot(1, 3, 2), plt.imshow(opening, cmap="gray"), plt.title("Opening")
plt.subplot(1, 3, 3), plt.imshow(closing, cmap="gray"), plt.title("Closing")
plt.show()