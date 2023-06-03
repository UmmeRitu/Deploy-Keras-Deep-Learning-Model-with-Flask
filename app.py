from math import expm1

import joblib
import pandas as pd
from flask import Flask, jsonify, request
# from tensorflow import keras
import pickle

app = Flask(__name__)
# model = keras.models.load_model("assets/price_prediction_model3.h5")
# transformer = joblib.load("assets/data_transformer3.joblib")
filename = 'iris/decision_tree_model.sav'
model = pickle.load(open(filename, 'rb'))

@app.route("/", methods=["POST"])
def index():
    print("This came here")
    print(request.json)
    data = request.json

    number1 = data['data'][0][0]
    number2 = data['data'][0][1]
    number3 = data['data'][0][2]
    number4 = data['data'][0][3]


    prediction = model.predict([[number1, number2, number3, number4 ]])
    # df = pd.DataFrame(data, index=[0])
    # prediction = model.predict(transformer.transform(df))
    # predicted_price = expm1(prediction.flatten()[0])
    # # return jsonify({  "message": "Hello, world!"})
    return jsonify({"Specis": str(prediction[0])})
