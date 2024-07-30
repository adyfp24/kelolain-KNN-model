import joblib
import pandas as pd

def load_model():
    return joblib.load('../model/model.pkl')

def predict(model, data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probabilities = model.predict_proba(df)[0]
    return prediction[0], probabilities.tolist()
