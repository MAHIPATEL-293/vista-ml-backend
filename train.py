from ultralytics import YOLO

# Load YOLOv8 model (auto downloads)
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="dataset/road_defects_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)