from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")
print("App started and model loaded")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Received POST request")

    if 'image' not in request.files:
        print("No image found in request")
        return {'error': 'No image uploaded'}, 400

    try:
        image = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)
        print("Image saved to", image_path)

        # Run YOLO on image
        results = model(image_path)[0]
        print("Model inference completed")

        # Load image with OpenCV
        img = cv2.imread(image_path)
        print("Image loaded with OpenCV")

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            print("Detected:", label)

            if label == 'car':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save result image
        result_path = os.path.join(RESULT_FOLDER, 'result_' + image.filename)
        cv2.imwrite(result_path, img)
        print("Saved result to", result_path)

        return send_file(result_path, mimetype='image/jpeg')

    except Exception as e:
        print("Error during analysis:", str(e))
        return {'error': str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
