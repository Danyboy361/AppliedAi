from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load("co2_linear_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "CO2 Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"predicted_co2": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
