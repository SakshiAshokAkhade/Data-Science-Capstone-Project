import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the feature columns as used during model training
feature_columns = ['year', 'km_driven', 
                   'name_Audi', 'name_BMW', 'name_Ford', 'name_Honda', 'name_Hyundai', 'name_Maruti', 'name_Toyota',
                   'fuel_CNG', 'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol',
                   'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
                   'transmission_Automatic', 'transmission_Manual',
                   'owner_First Owner', 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner']

# Initialize the scaler
scaler = StandardScaler()

# Define the Streamlit app
def main():
    st.title("Car Selling Price Prediction App")

    name = st.selectbox('Name', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Audi'])
    year = st.number_input('Year', min_value=1990, max_value=2024, value=2020)
    km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=10000)
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

    input_data = pd.DataFrame({
        'name': [name], 'year': [year], 'km_driven': [km_driven],
        'fuel': [fuel], 'seller_type': [seller_type], 'transmission': [transmission], 'owner': [owner]
    })

    input_data = pd.get_dummies(input_data, columns=['name', 'fuel', 'seller_type', 'transmission', 'owner'])

    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_columns]

    input_data[['km_driven']] = scaler.fit_transform(input_data[['km_driven']])

    if st.button('Predict Selling Price'):
        prediction = model.predict(input_data)
        st.write(f"The predicted selling price is: ₹{prediction[0]:.2f}")

if __name__ == '__main__':
    main()
