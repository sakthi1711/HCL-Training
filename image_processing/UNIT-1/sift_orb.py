import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("Imaging\\UNIT1\\HCL-Training\\image_processing\\UNIT-1\\chessboard.jpg", 0)

if img is None:
    raise FileNotFoundError("Please in the same folder.")

# ---- SIFT ----
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, keypoints_sift, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ---- ORB ----
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0,255,0),
                            flags=0)

# Display results
plt.subplot(1, 2, 1), plt.imshow(img_sift), plt.title(f"SIFT ({len(keypoints_sift)} features)")
plt.subplot(1, 2, 2), plt.imshow(img_orb), plt.title(f"ORB ({len(keypoints_orb)} features)")
plt.show()