import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\\HCL\\Dataset\\Panda.png', cv2.IMREAD_GRAYSCALE)

noisy_img = img.copy()
salt_pepper_amount = 0.05
rows, cols = noisy_img.shape
num_pixels = rows * cols
num_salt = int(num_pixels * salt_pepper_amount / 2)
num_pepper = int(num_pixels * salt_pepper_amount / 2)

# Add salt noise (white pixels)
salt_coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape]
noisy_img[salt_coords[0], salt_coords[1]] = 255

# Add pepper noise (black pixels)
pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_img.shape]
noisy_img[pepper_coords[0], pepper_coords[1]] = 0

# The kernel size must be a positive, odd integer (e.g., 3, 5, 7)
kernel_size = 3
filtered_img = cv2.medianBlur(noisy_img, kernel_size)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(noisy_img, cmap='gray')
axes[0].set_title('Noisy Image')
axes[0].axis('off')

axes[1].imshow(filtered_img, cmap='gray')
axes[1].set_title(f'Median Filtered ({kernel_size}x{kernel_size})')
axes[1].axis('off')

plt.show()
