import cv2
import math
import matplotlib.pyplot as plt

# ---------- USER INPUT ----------
images_info = [
    ("D://HCL_Git//HCL-Training//image_processing//UNIT-2//Images//4mm.png", 4.0),
    ("D://HCL_Git//HCL-Training//image_processing//UNIT-2//Images//8mm.png", 8.0),
    ("D://HCL_Git//HCL-Training//image_processing//UNIT-2//Images//16mm.png", 16.0)
]

sensor_width_mm = 6.3        # Example smartphone sensor
aperture_N = 2.8
circle_of_confusion = 0.005  # mm
subject_distance_mm = 1000   # 1 meter

real_object_width_mm = 100.0  # known width (e.g., 10 cm ruler)
pixel_x1, pixel_x2 = 200, 600  # known pixel positions
# --------------------------------

def compute_fov(sensor_width_mm, focal_length_mm):
    """Compute Field of View in degrees."""
    return 2 * math.degrees(math.atan(sensor_width_mm / (2 * focal_length_mm)))

def compute_dof(f, N, c, s):
    """Approximate DOF (Depth of Field) in mm."""
    return (2 * N * c * s**2) / (f**2)

focal_lengths = []
fovs = []
dofs = []

# ---------- Analysis ----------
for image_path, focal_length in images_info:
    img = cv2.imread(image_path)
    if img is None:
        print(f" Could not load {image_path}")
        continue

    fov = compute_fov(sensor_width_mm, focal_length)
    pixel_width = pixel_x2 - pixel_x1
    pixel_resolution = real_object_width_mm / pixel_width  # mm/pixel
    dof = compute_dof(focal_length, aperture_N, circle_of_confusion, subject_distance_mm)

    focal_lengths.append(focal_length)
    fovs.append(fov)
    dofs.append(dof)

    print("\n Image:", image_path)
    print(f" Focal Length: {focal_length} mm")
    print(f" Field of View: {fov:.2f}°")
    print(f" Pixel Resolution: {pixel_resolution:.4f} mm/pixel")
    print(f" Estimated DOF: {dof:.2f} mm")

# ---------- Plot Comparison ----------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(focal_lengths, fovs, 'o-', label="Field of View")
plt.xlabel("Focal Length (mm)")
plt.ylabel("FOV (degrees)")
plt.title("Field of View vs Focal Length")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(focal_lengths, dofs, 's--', label="Depth of Field", color='orange')
plt.xlabel("Focal Length (mm)")
plt.ylabel("DOF (mm)")
plt.title("Depth of Field vs Focal Length")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
