from math import expm1

import joblib
import pandas as pd
from flask import Flask, jsonify, request
# from tensorflow import keras
import pickle

app = Flask(__name__)
# model = keras.models.load_model("assets/price_prediction_model3.h5")
# transformer = joblib.load("assets/data_transformer3.joblib")
filename = 'decision_tree_model.sav'
model = pickle.load(open(filename, 'rb'))

@app.route("/", methods=["POST"])
def index():
    print("This came here")
    print(request)
    data = request.json
    prediction = model.predict([[6.8, 3.2, 5.9, 2.3]])
    # df = pd.DataFrame(data, index=[0])
    # prediction = model.predict(transformer.transform(df))
    # predicted_price = expm1(prediction.flatten()[0])
    # # return jsonify({  "message": "Hello, world!"})
    return jsonify({"Specis": str(prediction[0])})
