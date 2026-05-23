from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train-2/weights/best.pt")

# Run detection (save result instead of showing)
results = model("test.jpg", save=True)

print("Done ✅")