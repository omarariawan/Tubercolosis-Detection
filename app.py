# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import feature_extractor as fe
import traceback

app = Flask(__name__)
# Enable CORS for all routes so frontend can access this API
CORS(app) 

# --- LOAD MODEL AND SCALER ---
print("Loading SVM Model and Scaler...")
model = None
scaler = None
try:
    # Ensure these files exist in the same directory!
    model = joblib.load('svm_tb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Model or Scaler file not found. Train your model first! ({str(e)})")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint.
    Expected Input: A file upload with key 'file'.
    Returns: JSON {'result': 'Tuberculosis' or 'Normal', 'status': 200}
    """
    if not model or not scaler:
        return jsonify({'error': 'Model or Scaler not loaded. Train the model first.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # 1. Pipeline: Preprocess -> Segment -> Extract
        print(f"Processing image: {file.filename}")
        processed_img = fe.preprocess_image(file)
        segmented_img = fe.segment_lung(processed_img)
        features = fe.extract_features(segmented_img)
        
        # 2. Scale features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # 3. Predict
        # We assume 0 = Normal, 1 = Tuberculosis based on previous training
        prediction_index = model.predict(features_scaled)[0]
        
        result_label = "Tuberculosis" if prediction_index == 1 else "Normal"
        
        # Try to get confidence score
        confidence = None
        try:
            confidence = float(np.max(model.predict_proba(features_scaled)))
        except:
            pass
        
        response_data = {
            'status': 'success',
            'filename': file.filename,
            'prediction': result_label
        }
        
        if confidence is not None:
            response_data['confidence'] = confidence
        
        return jsonify(response_data)

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)