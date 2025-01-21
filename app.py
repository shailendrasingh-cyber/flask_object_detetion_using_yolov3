# from flask import Flask, request, jsonify
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# import numpy as np

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # SERVER-SIDE VALIDATION
#     if 'image' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['image']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in {'jpeg', 'jpg', 'png', 'webp'}):
#         return jsonify({'error': 'Allowed image types are -> jpeg, jpg, png, webp'}), 400

#     # Read image file
#     image_file = file.read()
#     nparr = np.frombuffer(image_file, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Perform object detection
#     bbox, label, conf = cv.detect_common_objects(
#         img, confidence=0.2, model='yolov3-tiny')

#     # Draw bounding boxes and labels on the image
#     output_image = draw_bbox(img, bbox, label, conf)

#     # Save the output image to static directory
#     output_path = 'static/outputs/output.jpg'
#     cv2.imwrite(output_path, output_image)

#     # Prepare results to return as JSON
#     results = []
#     for i in range(len(bbox)):
#         results.append({
#             'label': label[i],
#             'confidence': float(conf[i]),
#             'bbox': bbox[i].tolist()
#         })

#     return jsonify({
#         'image_path': output_path,
#         'detections': results
#     })

# if __name__ == '__main__':
#     # Run the app on all available interfaces (useful for platforms like Render)
#     app.run(host='0.0.0.0', port=5000, debug=True)



# from flask import Flask, request, jsonify
# import cv2
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# import numpy as np
# import os
# import base64
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     image_bytes = file.read()
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Perform object detection
#     bbox, label, conf = cv.detect_common_objects(img, confidence=0.25, model='yolov3-tiny')

#     # Draw bounding boxes on the image
#     output_image = draw_bbox(img, bbox, label, conf)

#     # Encode image as base64
#     _, encoded_image = cv2.imencode('.jpg', output_image)
#     base64_image = base64.b64encode(encoded_image).decode('utf-8')

#     detections = [{"label": l, "confidence": c, "bbox": b} for l, c, b in zip(label, conf, bbox)]

#     print("Response data:", {"image_path": "data:image/jpeg;base64," + base64_image, "detections": detections})  # Log response data

#     return jsonify({
#         "image_path": "data:image/jpeg;base64," + base64_image,
#         "detections": detections
#     })


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=9898, debug=True)



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
