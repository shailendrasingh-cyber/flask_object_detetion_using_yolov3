from flask import Flask, request, jsonify
from flask_cors import CORS  
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    bbox, label, conf = cv.detect_common_objects(img, confidence=0.25, model='yolov3-tiny')

    # Draw bounding boxes on the image
    output_image = draw_bbox(img, bbox, label, conf)

    # Encode image as base64
    _, encoded_image = cv2.imencode('.jpg', output_image)
    base64_image = base64.b64encode(encoded_image).decode('utf-8')

    detections = [{"label": l, "confidence": c, "bbox": b} for l, c, b in zip(label, conf, bbox)]

    return jsonify({
        "image_path": "data:image/jpeg;base64," + base64_image,
        "detections": detections
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9898, debug=True)
