import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('D:\HCL_Git\Images\colors.png')  # make sure your image name matches
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to HSV and LAB
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)



# Example segmentation for red color in HSV
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 | mask2
seg_hsv = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# Display results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title('Original RGB')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title('HSV Image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(seg_hsv)
plt.title('Segmented (Red Color)')
plt.axis('off')

plt.show()
