from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'kelolain-model.pkl')
model = joblib.load(model_path)

def predict(data):
    df = pd.DataFrame(data)
    
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)

    analyses = {
        'baik': "Keuangan Anda dalam kondisi baik. Pengeluaran Anda terkendali dan tabungan cukup.",
        'sedang': "Keuangan Anda dalam kondisi sedang. Ada ruang untuk perbaikan dalam pengelolaan keuangan.",
        'buruk': "Keuangan Anda dalam kondisi buruk. Pengeluaran melebihi pendapatan dan tabungan rendah."
    }
    
    recommendations = {
        'baik': "Teruskan kebiasaan baik ini. Pertimbangkan untuk menambah investasi atau tabungan lebih lanjut.",
        'sedang': "Perhatikan pengeluaran Anda dan cobalah meningkatkan tabungan. Buat anggaran untuk memantau keuangan Anda.",
        'buruk': "Buat rencana anggaran untuk mengurangi pengeluaran. Cobalah mencari sumber pendapatan tambahan dan hindari utang."
    }
    
    result = {
        'prediction': prediction.tolist(),
        'probabilities': probabilities.tolist(),
        'analysis': analyses[prediction[0]],
        'recommendation': recommendations[prediction[0]]
    }
    
    return result

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'message': 'hello world'})

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()

    if isinstance(data, dict):
        data = [data]  

    try:
        result = predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)