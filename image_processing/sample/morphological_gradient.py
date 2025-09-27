import cv2
import numpy as np

img = cv2.imread('morpho.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Could not read the image. Please check the file path.")
else:
    kernel = np.ones((5, 5), np.uint8)

    # Calculate the dilation and erosion versions
    dilated_img = cv2.dilate(img, kernel, iterations=1)
    eroded_img = cv2.erode(img, kernel, iterations=1)
    
    # Calculate the morphological gradient
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    cv2.imshow('Original', img)
    cv2.imshow('Dilated Image', dilated_img)
    cv2.imshow('Eroded Image', eroded_img)
    cv2.imshow('Morphological Gradient', gradient)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()





































