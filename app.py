# Import Libraries
import numpy as np
import json
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model from disk
    filename = 'model/rental_prediction_model.pkl'
    model = pickle.load(open(filename, 'rb'))

    # Read input from request JSON
    user_input = request.json

    rooms = int(user_input.get('rooms', 0))
    sqft = int(user_input.get('sqft', 0))

    user_input_prediction = np.array([[rooms, sqft]])
    predicted_rental_price = model.predict(user_input_prediction)

    # Predict the Rental Price
    output = {"Rental Price Prediction Using Model": float(predicted_rental_price[0])}

    # Write Outputs to outputs.json
    with open('outputs/outputs.json', 'w') as f:
        json.dump(output, f)

    # Return prediction as API response
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

