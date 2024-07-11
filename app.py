import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
try:
    with open('decision_tree_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading decision_tree_model.pkl: {e}")
    st.stop()

# Define function to make predictions
def predict_selling_price(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app
st.title("Car Selling Price Prediction")

st.write("""
## Predict the selling price of a car based on its features.
""")

# Input fields for the features
km_driven = st.number_input('Kilometers Driven', min_value=0)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner Type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# One-hot encoding for categorical features
fuel_dict = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
seller_type_dict = {'Individual': 0, 'Dealer': 1, 'Trustmark Dealer': 2}
transmission_dict = {'Manual': 0, 'Automatic': 1}
owner_dict = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}

fuel_encoded = fuel_dict[fuel]
seller_type_encoded = seller_type_dict[seller_type]
transmission_encoded = transmission_dict[transmission]
owner_encoded = owner_dict[owner]

# Button for prediction
if st.button('Predict'):
    features = [km_driven, fuel_encoded, seller_type_encoded, transmission_encoded, owner_encoded]
    prediction = predict_selling_price(features)
    st.write(f'The predicted selling price is {prediction:.2f}')
