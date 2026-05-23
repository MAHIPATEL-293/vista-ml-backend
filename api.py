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
    return {"message": "VISTA ML Backend is running"}

road_model = YOLO("runs/segment/train-3/weights/best.pt")
defect_model = YOLO("runs/detect/train-2/weights/best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"isRoad": False})

    file = request.files["image"]
    filename = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(filename)

    img = cv2.imread(filename)

    road_results = road_model(filename)

    if road_results[0].masks is None:
        os.remove(filename)
        return jsonify({"isRoad": False})

    os.remove(filename)

    return jsonify({"isRoad": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)