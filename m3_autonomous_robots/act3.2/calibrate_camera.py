import cv2
import numpy as np
import glob
import os

""" CHECKBOARD PARAMETERS """

num_cols = 5 # Internal Corners
num_rows = 7 
square_size = 30 # mm

""" STEP 1. PREPARE 3D CHECKBOARD POINTS """

objp = np.zeros((num_cols * num_rows, 3), np.float32)
# Creates grid of coordinates and flattens it to a list of points
objp[:,:2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1,2)  # (0, 1), (1, 0), (2, 0) ...
objp = objp * square_size # Scale the grid to real life units

objpoints = [] # Real Coordinates
imgpoints = [] # Pixel Coordinates detected in each photo

""" STEP 2. DETECT CORNERS IN EACH CALIBRATION IMAGE """

images = glob.glob('fotografias/checkboard/*.jpg')
print(f"Found {len(images)} calibration images\n")

valid_images = 0

for image_path in images:

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # findChessboardCorners SEARCHES FOR THE CORNER PATTERN IN THE GRAYSCALE IMAGE
    found, corners = cv2.findChessboardCorners(gray, (num_cols, num_rows), None)

    if found:
        valid_images += 1
        
        # cornerSubPix REFINES THE CORNERS TO SUB-PIXEL LEVEL FOR BETTER ACCURACY
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Stop when error is small enough or after max iterations
        refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints.append(refined_corners)


""" STEP 3. CAMERA PARAMETERS """

h, w = cv2.imread(images[0]).shape[:2]

reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # Real 3D points
    imgpoints,   # Detected 2D points
    (w, h),      # Image size
    None,        # Initial matrix
    None         # Initial coefficients
)

print(f"\n Calibration completed!")
print(f"   Reprojection error (RMS): {reprojection_error:.4f} px") # Measures how well the model fits (lower = better)
print(f" Intrinsic camera matrix K:\n{camera_matrix}\n") # Focal lengths & Optical Center
print(f" Distortion coefficients:\n{dist_coeffs}\n") # Lens Distortion Parameter


np.savez("camera_parameters.npz",
         camera_matrix = camera_matrix,
         dist_coeffs = dist_coeffs)

print(" Parameters saved in 'camera_parameters.npz'")
