import cv2
import numpy as np
import glob
import os

""" STEP 1. LOAD CAMERA PARAMETERS """
with np.load("camera_parameters.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]


""" STEP 2. LOAD AND UNDISTORT IMAGES """
image_paths = sorted(glob.glob("fotografias/panorama/*.jpg")) 
print(f"Found {len(image_paths)} images. Removing lens distortion...")

images_to_stitch = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
        
    h, w = img.shape[:2]    # Extract height and width
    
    # Calculate the optimal camera matrix for this specific image size
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    
    # Remove distortion
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    images_to_stitch.append(img_undistorted)

print("✅ All images corrected.")


""" STEP 3. STITCH IMAGES TOGETHER """

stitcher = cv2.Stitcher_create() # Built-in OpenCV stitcher object
status, panorama = stitcher.stitch(images_to_stitch) # Returns final stitched image

if status == cv2.Stitcher_OK:    
    cv2.imwrite("massive_panorama.jpg", panorama)
    print("Saved as 'massive_panorama.jpg'")

    # Display a resized version of the full panorama
    display_img = cv2.resize(panorama, (int(panorama.shape[1]*0.2), int(panorama.shape[0]*0.2)))
    cv2.imshow("Massive Panorama", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
