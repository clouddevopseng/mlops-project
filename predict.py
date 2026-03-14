# Import Libraries
import numpy as np
import json
import pickle

# Load the model from disk
filename = 'model/model.pkl'
model = pickle.load(open(filename, 'rb'))

# Read Inputs from Inputs/inputs.json
with open('inputs/inputs.json', 'r') as f:
    user_input = json.load(f)

rooms = user_input['rooms']
sqft = user_input['sqft']

user_input_prediction = np.array([[rooms, sqft]])
predicted_rental_price = model.predict(user_input_prediction)

# Predict the Rental Price
output = {'Rental Price Prediction using Model': float(predicted_rental_price[0])}

# Write Outputs to outputs/outputs.json
with open('outputs/outputs.json', 'w') as f:
    json.dump(output, f)

