from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

model.train(
    data="E:/data/data.yaml",
    epochs=10,
    imgsz=320,
    batch=2,
    workers=0
)            