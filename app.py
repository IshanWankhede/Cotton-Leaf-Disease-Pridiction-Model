from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import io
from PIL import Image
import base64
import numpy as np
import cv2

app = Flask(__name__)


try:
    model = YOLO('best.pt')
    
    CLASS_NAMES = model.names
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'best.pt' is in the same directory as app.py")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
       
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

       
        results = model(img)
        
        
        result = results[0]
        
        
        annotated_frame = result.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(annotated_frame)
        
        
        buff = io.BytesIO()
        im_pil.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        
        predictions = []
        best_conf = 0
        best_class = "Unknown"

        
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if class_id in CLASS_NAMES:
                    label = CLASS_NAMES[class_id]
                else:
                    label = str(class_id)
                
                predictions.append({
                    'class': label,
                    'confidence': round(conf * 100, 2)
                })

                
                if conf > best_conf:
                    best_conf = conf
                    best_class = label
        else:
            
            best_class = "No Detection"
            best_conf = 0.0

        return jsonify({
            'status': 'success',
            'image': f"data:image/jpeg;base64,{img_str}",
            'best_class': best_class,
            'confidence': round(best_conf * 100, 2),
            'all_predictions': predictions
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    
    print("Starting app...")
    print("If running locally, go to http://127.0.0.1:5000")
    print("To access on mobile, ensure you are on the same Wi-Fi and use your computer's IP address.")
    app.run(debug=True, host='0.0.0.0', port=5000)