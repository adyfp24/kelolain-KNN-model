from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

def create_app():
    app = Flask(__name__)

    def load_model():
        return joblib.load('model/kelolain-model.pkl')

    model = load_model()

    def predict(model, data):

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
    def hello_wolrd():
        return jsonify({'message': 'hello world'})
    
    @app.route('/predict', methods=['POST'])
    def predict_endpoint():
        data = request.get_json()

        if isinstance(data, dict):
            data = [data]  

        try:
            result = predict(model, data)
            return jsonify(result)
        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 400


    return app
