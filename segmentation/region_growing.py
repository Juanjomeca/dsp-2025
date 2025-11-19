import cv2
import numpy as np
from collections import deque

def region_growing(img, seed_coordinates, threshold=5):
    h, w = img.shape
    output = np.zeros_like(img, dtype=np.uint8)
    visited = np.zeros_like(img, dtype=np.bool_)

    queue = deque([seed_coordinates])
    seed_intensity = int(img[seed_coordinates])

    neighbors = [(-1,-1), (-1,0), (-1,1),
                 (0,-1),         (0,1),
                 (1,-1),  (1,0), (1,1)]

    while queue:
        y, x = queue.popleft()
        if visited[y, x]:
            continue
        visited[y, x] = True

        if abs(int(img[y,x]) - seed_intensity) <= threshold:
            output[y,x] = 255

            for dy, dx in neighbors:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny,nx]:
                    queue.append((ny,nx))
    return output

img = cv2.imread('lab_images/frutas.jpg', 0)

blur = cv2.GaussianBlur(img, (3,3), 0)
color_img = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

window_name = "Region Growing"
cv2.namedWindow(window_name)

threshold_val = 20
seed = None
region_mask = None

def on_trackbar(val):
    global threshold_val, seed, region_mask
    threshold_val = max(val, 1)
    if seed is not None:
        region_mask = region_growing(blur, seed, threshold_val)
        overlay = color_img.copy()
        overlay[region_mask==255] = [0,0,255]
        cv2.imshow(window_name, overlay)

def on_mouse(event, x, y, flags, param):
    global seed, region_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (y, x)
        print(f"Seed selected: {seed}")
        region_mask = region_growing(blur, seed, threshold_val)
        overlay = color_img.copy()
        overlay[region_mask==255] = [0,0,255]
        cv2.imshow(window_name, overlay)

cv2.createTrackbar("Threshold", window_name, threshold_val, 100, on_trackbar)
cv2.setMouseCallback(window_name, on_mouse)

cv2.imshow(window_name, color_img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        seed = None
        region_mask = None
        cv2.imshow(window_name, color_img)

cv2.destroyAllWindows()
