import os

import pandas as pd
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model_1 import train_model, preproseccing_data
#from sklearn.externals import joblib
import joblib

app = Flask(__name__)
api = Api(app)

if not os.path.isfile('flats_price.model'):
    train_model()
model = joblib.load('flats_price.model')

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        data_df = pd.DataFrame(posted_data, index=[0])
        prediction = int(model.predict(preproseccing_data(data_df))[0])
        return jsonify({
            'Prediction': prediction
        })

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)