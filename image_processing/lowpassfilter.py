import cv2
import numpy as np

img = cv2.imread('D:\\HCL\\Dataset\\wood\\test\\good\\003.png' , cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), np.float32) / 25

filtered_img = cv2.filter2D(img, -1, kernel)

cv2.imshow('orignal image', img)
cv2.imshow('filtered image', filtered_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
