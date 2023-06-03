from math import expm1

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("assets/price_prediction_model2.h5")
transformer = joblib.load("assets/data_transformer2.joblib")


@app.route("/", methods=["POST"])
def index():
    print("This came here")
    print(request)
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(transformer.transform(df))
    predicted_price = expm1(prediction.flatten()[0])
    # return jsonify({  "message": "Hello, world!"})
    return jsonify({"price": str(predicted_price)})
