import cv2
import matplotlib.pyplot as plt

img = cv2.imread('D:\\HCL\\Dataset\\bottle\\test\\good\\015.png', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (5, 5), 1.4)

edges = cv2.Canny(blurred, 100, 200)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.show()
