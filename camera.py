from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    # Save image instead of showing
    cv2.imwrite(f"output_{frame_count}.jpg", annotated)
    print(f"Saved output_{frame_count}.jpg")

    frame_count += 1

    if frame_count == 5:  # stop after 5 frames
        break

cap.release()