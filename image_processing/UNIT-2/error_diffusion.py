import cv2
import numpy as np

def ordered_dithering(img):
    """Apply ordered dithering (Bayer matrix)"""
    # 4x4 Bayer matrix
    bayer_matrix = (1/16) * np.array([[0, 8, 2, 10],
                                      [12, 4, 14, 6],
                                      [3, 11, 1, 9],
                                      [15, 7, 13, 5]])
    h, w = img.shape
    tile_h = int(np.ceil(h / 4))
    tile_w = int(np.ceil(w / 4))
    threshold_map = np.tile(bayer_matrix, (tile_h, tile_w))
    threshold_map = threshold_map[:h, :w] * 255
    halftoned = (img > threshold_map).astype(np.uint8) * 255
    return halftoned

def error_diffusion(img):
    """Apply Floyd–Steinberg error diffusion halftoning"""
    img = img.astype(float)
    h, w = img.shape
    halftoned = np.copy(img)
    for y in range(h):
        for x in range(w):
            old_pixel = halftoned[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            halftoned[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x+1 < w:
                halftoned[y, x+1] += quant_error * 7/16
            if y+1 < h and x > 0:
                halftoned[y+1, x-1] += quant_error * 3/16
            if y+1 < h:
                halftoned[y+1, x] += quant_error * 5/16
            if y+1 < h and x+1 < w:
                halftoned[y+1, x+1] += quant_error * 1/16
    halftoned = np.clip(halftoned, 0, 255).astype(np.uint8)
    return halftoned

if __name__ == "__main__":
    # Load your grayscale industrial image
    img_path = "D:\\HCL_Git\\Images\\x-ray.jpg"  # change this path
    img = cv2.imread(img_path, 0)  # 0 = grayscale
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Choose method
    print("Choose halftoning method:")
    print("1 - Ordered Dithering")
    print("2 - Error Diffusion (Floyd–Steinberg)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        halftoned = ordered_dithering(img)
        output_path = "ordered_halftone.png"
    elif choice == "2":
        halftoned = error_diffusion(img)
        output_path = "floyd_steinberg_halftone.png"
    else:
        raise ValueError("Invalid choice! Enter 1 or 2.")

    # Save and display1
    cv2.imshow("Halftone", halftoned)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
