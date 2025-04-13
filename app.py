import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("mlr_model.pkl", "rb") as file:
    model = pickle.load(file)

# Custom page config
st.set_page_config(page_title="NYC Green Taxi Fare Predictor", layout="centered")

# Custom styles
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            padding: 20px;
        }
        h1 {
            color: #003366;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #004080;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚖 NYC Green Taxi Fare Predictor</h1>", unsafe_allow_html=True)
st.markdown("Fill in the trip details below to predict the **total fare amount**.")

# Input fields
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, step=0.1)
fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, step=0.1)
extra = st.number_input("Extra Charges ($)", min_value=0.0, step=0.1)
mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, step=0.1)
tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, step=0.1)
tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, step=0.1)
improvement_surcharge = st.number_input("Improvement Surcharge ($)", min_value=0.0, step=0.1)
congestion_surcharge = st.number_input("Congestion Surcharge ($)", min_value=0.0, step=0.1)
passenger_count = st.number_input("Passenger Count", min_value=1, step=1)
trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict Fare"):
    input_features = np.array([[trip_distance, fare_amount, extra, mta_tax, tip_amount,
                                tolls_amount, improvement_surcharge, congestion_surcharge,
                                passenger_count, trip_duration]])
    prediction = model.predict(input_features)[0]
    st.success(f"💵 Estimated Total Fare: ${prediction:.2f}")
