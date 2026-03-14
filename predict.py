# Import Libraries
import pandas as pd
import numpy as np
import json
import pickle
from flask import Flask, jsonify

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Load the model from disk
    filename = 'model/rental_prediction_model.pkl'
    model = pickle.load(open(filename, 'rb'))

    # Read inputs from inputs.json
    with open('inputs/inputs.json', 'r') as f:
        user_input = json.load(f)

    rooms = int(user_input['rooms'])
    sqft = int(user_input['sqft'])

    user_input_prediction = np.array([[rooms, sqft]])
    predicted_rental_price = model.predict(user_input_prediction)

    # Predict the Rental Price
    output = {'Rental Price Prediction using Model': float(predicted_rental_price[0])}

    # Write Outputs to outputs/outputs.json
    with open('outputs/outputs.json', 'w') as f:
        json.dump(output, f)

    print(output)

    # Return prediction as API response
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
