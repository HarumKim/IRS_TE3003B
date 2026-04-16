import cv2
import numpy as np
import glob


""" STEP 1. LOAD CAMERA PARAMETERS """

with np.load("/home/kim/irs_tec/m3_autonomous_robots/act3.3/camera_parameters.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]


""" STEP 2. LOAD AND UNDISTORT IMAGES """

image_paths = sorted(glob.glob("/home/kim/irs_tec/fotografias/map/*.jpeg"))
print(f"Found {len(image_paths)} images. Removing lens distortion...")

images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    h, w = img.shape[:2]
    new_cam, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    images.append(cv2.undistort(img, camera_matrix, dist_coeffs, None, new_cam))

print(f"✅ {len(images)} images corrected.")


def get_drawing_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, paper = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
    k = np.ones((20, 20), np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k)
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, k)

    _, dark = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.bitwise_and(dark, paper)

    dk = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(lines, dk, iterations=3)
    return mask, gray


def find_translation(img1, img2, min_matches=4):

    mask1, gray1 = get_drawing_mask(img1)
    mask2, gray2 = get_drawing_mask(img2)

    n1 = int(mask1.sum() // 255)
    n2 = int(mask2.sum() // 255)
    print(f" Drawing pixels: {n1}, {n2}")

    if n1 < 500 or n2 < 500:
        return None

    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01, edgeThreshold=20)
    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)
    print(f" Keypoints: {len(kp1)}, {len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < min_matches or len(kp2) < min_matches:
        print(" Not enough keypoints.")
        return None

    bf = cv2.BFMatcher()
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f" Good matches: {len(good)}")

    if len(good) < min_matches:
        print(" Not enough matches.")
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    dx = pts1[:, 0] - pts2[:, 0]
    dy = pts1[:, 1] - pts2[:, 1]
    tx, ty = float(np.median(dx)), float(np.median(dy))

    inliers = int(np.sum((np.abs(dx - tx) < 30) & (np.abs(dy - ty) < 30)))
    print(f" Translation: ({tx:+.1f}, {ty:+.1f}) px | inliers: {inliers}/{len(good)}")

    if inliers < min_matches:
        print(" Too few inliers.")
        return None

    return tx, ty


""" STEP 3. COMPUTE ABSOLUTE POSITIONS """

positions = [(0.0, 0.0)]
for i in range(1, len(images)):
    print(f"\nPair {i} → {i+1}:")
    result = find_translation(images[i - 1], images[i])
    if result is not None:
        tx, ty = result
        positions.append((positions[-1][0] + tx, positions[-1][1] + ty))
        print(f" ✅ Accumulated position: {positions[-1]}")
    else:
        positions.append(positions[-1])

print(f"\nFinal positions: {[(round(x,1), round(y,1)) for x,y in positions]}")


""" STEP 4. ASSEMBLE CANVAS """

img_h, img_w = images[0].shape[:2]
xs = [p[0] for p in positions]
ys = [p[1] for p in positions]
x_min, y_min = min(xs), min(ys)

canvas_w = int(np.ceil(max(xs) + img_w - x_min))
canvas_h = int(np.ceil(max(ys) + img_h - y_min))

canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32) * 255
weight_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

for idx, (img, (ox, oy)) in enumerate(zip(images, positions)):
    x = int(round(ox - x_min))
    y = int(round(oy - y_min))
    cx1 = max(0, x); cy1 = max(0, y)
    cx2 = min(canvas_w, x + img_w); cy2 = min(canvas_h, y + img_h)
    sx1, sy1 = cx1 - x, cy1 - y
    h_roi, w_roi = cy2 - cy1, cx2 - cx1

    patch = img[sy1:sy1+h_roi, sx1:sx1+w_roi]

    gray_p = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, pmask = cv2.threshold(gray_p, 150, 255, cv2.THRESH_BINARY)
    k = np.ones((15, 15), np.uint8)
    pmask = cv2.morphologyEx(pmask, cv2.MORPH_CLOSE, k)
    pmask = cv2.morphologyEx(pmask, cv2.MORPH_OPEN, k)

    dist = cv2.distanceTransform(pmask, cv2.DIST_L2, 5).astype(np.float32)
    # Normalize to [0, 1]
    d_max = dist.max()
    if d_max > 0:
        dist /= d_max

    w_roi_map = weight_map[cy1:cy2, cx1:cx2]
    c_roi = canvas[cy1:cy2, cx1:cx2]

    for ch in range(3):
        c_roi[:, :, ch] = (
            c_roi[:, :, ch] * w_roi_map + patch[:, :, ch].astype(np.float32) * dist
        ) / np.maximum(w_roi_map + dist, 1e-6)

    weight_map[cy1:cy2, cx1:cx2] = w_roi_map + dist
    canvas[cy1:cy2, cx1:cx2] = c_roi
    print(f"Placed image {idx+1} at canvas ({cx1}, {cy1})")

canvas = np.clip(canvas, 0, 255).astype(np.uint8)


""" STEP 5. SAVE AND DISPLAY """

cv2.imwrite("/home/kim/irs_tec/m3_autonomous_robots/act3.3/photogramettry.jpg", canvas)

scale = min(1.0, 1400 / canvas.shape[1])
display = cv2.resize(canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)))
cv2.imshow("Photogrammetry", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
