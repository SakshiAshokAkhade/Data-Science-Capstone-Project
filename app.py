import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pipeline
try:
    with open('decision_tree_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except Exception as e:
    st.error(f"Error loading decision_tree_pipeline.pkl: {e}")
    st.stop()

# Define function to make predictions
def predict_selling_price(features):
    try:
        features_df = pd.DataFrame([features], columns=['km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])
        prediction = pipeline.predict(features_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

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

# Button for prediction
if st.button('Predict'):
    features = {
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }
    st.write(f"Input features: {features}")  # Debugging information
    prediction = predict_selling_price(features)
    if prediction is not None:
        st.write(f'The predicted selling price is {prediction:.2f}')
