from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "VISTA ML Backend is running",
        "status": "success"
    })

# Load trained models
road_model = YOLO("runs/segment/train-3/weights/best.pt")
defect_model = YOLO("runs/detect/train-2/weights/best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "isRoad": False,
            "message": "No image uploaded"
        }), 400

    file = request.files["image"]
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(filename)

    try:
        img = cv2.imread(filename)

        if img is None:
            return jsonify({
                "isRoad": False,
                "message": "Invalid image"
            }), 400

        road_results = road_model(filename)

        if road_results[0].masks is None:
            return jsonify({
                "isRoad": False,
                "message": "No road detected. Please capture road image only."
            })

        defect_results = defect_model(filename)

        detections = []

        if defect_results[0].boxes is not None:
            for box in defect_results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = defect_model.names[cls_id]

                detections.append({
                    "class": name,
                    "confidence": round(conf, 2)
                })

        return jsonify({
            "isRoad": True,
            "message": "Road image accepted",
            "detections": detections
        })

    except Exception as e:
        return jsonify({
            "isRoad": False,
            "message": str(e)
        }), 500

    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)