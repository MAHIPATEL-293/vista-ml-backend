from ultralytics import YOLO
import numpy as np
import cv2
import os

# Load models
road_model = YOLO("runs/segment/train-3/weights/best.pt")
defect_model = YOLO("runs/detect/train-2/weights/best.pt")

img_path = "test.jpg"

# Check image exists
if not os.path.exists(img_path):
    print("Image not found ❌")
    exit()

# Run models
road_results = road_model(img_path)
defect_results = defect_model(img_path)

# Load original image
img = cv2.imread(img_path)

if img is None:
    print("Image cannot be opened ❌")
    exit()

# --- Get ROAD MASK ---
road_mask = None
if road_results[0].masks is not None:
    road_mask = road_results[0].masks.data[0].cpu().numpy()
    road_mask = cv2.resize(road_mask, (img.shape[1], img.shape[0]))
else:
    print("No road detected ❌")
    exit()

# --- Process defects ---
boxes = defect_results[0].boxes

valid_detections = []

if boxes is not None and road_mask is not None:
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Check if inside road
        if road_mask[cy][cx] > 0.5:
            valid_detections.append((x1, y1, x2, y2))

# --- Draw only valid detections ---
for (x1, y1, x2, y2) in valid_detections:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("final_output.jpg", img)

print("Final output saved ✅")