from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Load the trained model and scaler
model = load('best_breast_cancer_model.joblib')
scaler = load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        return jsonify({
            'prediction': 'Benign' if prediction == 1 else 'Malignant',
            'isMalignant': bool(prediction == 0),
            'confidence': float(max(probability) * 100)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'Server is running'})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Model loaded successfully!")
    app.run(debug=True, port=5000)