import pandas as pd
import numpy as np
import json
import os
import pickle
from flask import Flask

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

app = Flask(__name__)

@app.route('/train', methods=['GET'])
def train():
    # Create Pandas DataFrame
    df = pd.read_csv('data/rent-data.csv')

    # Features and Labels
    X = df[['rooms', 'sqft']].values
    y = df['price'].values

    # Split the Training Data and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Algorithm Selection - Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to disk
    filename = 'model/model.pkl'
    pickle.dump(model, open(filename, 'wb'))    

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print("Root means square error:" , rmse)

    return("Model trained successfully")

@app.route('/predict', methods=['GET'])
def predict():
    # Load the model from disk
    filename = "model/model.pkl"
    model = pickle.load(open(filename, 'rb'))

    # Read Inputs from inputs/inputs.json
    with open("inputs/inputs.json", "r") as f:
        user_input = json.load(f)
    
    rooms = user_input["rooms"]
    sqft = user_input["sqft"]

    # Prepare input for prediction
    user_input_prediction = np.array([[rooms, sqft]])
    predicted_rental_price = model.predict(user_input_prediction)

    # Predict the Rental Price
    output = {"Rental Price Prediction Using Model": float(predicted_rental_price[0])}

    # Write Outputs to outputs/outputs.json
    with open("outputs/outputs.json", "w") as f:
        json.dump(output, f)

    print(output)   

    return(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

predict()