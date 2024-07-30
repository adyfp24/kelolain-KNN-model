from flask import request, jsonify
import app
from .model import load_model, predict

model = load_model()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    prediction, probabilities = predict(model, data)
    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities
    })
