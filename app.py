from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# ---- Load YOLO Model ----
MODEL_CFG = os.path.join(os.getcwd(), "yolov3-tiny.cfg")
MODEL_WEIGHTS = os.path.join(os.getcwd(), "yolov3-tiny.weights")
MODEL_CLASSES = os.path.join(os.getcwd(), "coco.names")

# Load class labels
with open(MODEL_CLASSES, "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

@app.route('/predict', methods=['POST'])
def predict():
    image_bytes = None

    # Case 1: multipart/form-data (HTML form upload)
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        image_bytes = file.read()

    # Case 2: JSON with base64
    elif request.is_json:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        try:
            image_bytes = base64.b64decode(data['image'])
        except Exception:
            return jsonify({"error": "Invalid base64 string"}), 400

    else:
        return jsonify({"error": "Unsupported request format"}), 400

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    # YOLO forward pass
    (H, W) = img.shape[:2]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.3)

    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            label = CLASSES[class_ids[i]]
            conf = confidences[i]

            # Draw bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [int(x), int(y), int(w), int(h)]
            })

    # Encode result image
    _, encoded_img = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(encoded_img).decode("utf-8")

    return jsonify({
        "image_path": "data:image/jpeg;base64," + base64_img,
        "detections": detections
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
