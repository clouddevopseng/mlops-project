import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# Create Pandas Dataframe
df = pd.read_csv('data/rental_1000.csv')

# Features and Labels
X = df[['rooms', 'sqft']].values
y = df['price'].values

# Split the Training Data and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Algorithm Selection - Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to disk
# filename = 'model/rental_prediction_model.pkl'

filename = 'model/rental_prediction_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Evaluate the model
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error : ", rmse)

print("Model Trained Successfully...Ready for Prediction")
