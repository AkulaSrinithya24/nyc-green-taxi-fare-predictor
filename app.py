import streamlit as st
import pickle
import pandas as pd

# Set page config
st.set_page_config(page_title="NYC Taxi Fare Predictor 🚕", layout="wide")

# Load model and scaler
with open("taxi_model.pkl", "rb") as f:
    model, scaler, feature_columns = pickle.load(f)

# Background and header
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f1f6ff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color: #004d99;'>🚕 NYC Green Taxi Fare Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px;'>Fill in the trip details below to predict the <b>total fare amount</b>.</p>", unsafe_allow_html=True)

# User inputs
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, max_value=50.0, value=2.0)
fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, value=8.0)
extra = st.number_input("Extra Charges ($)", min_value=0.0, value=0.5)
mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, value=0.5)
tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, value=1.0)
tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, value=0.0)
improvement_surcharge = st.number_input("Improvement Surcharge ($)", min_value=0.0, value=0.3)
congestion_surcharge = st.number_input("Congestion Surcharge ($)", min_value=0.0, value=2.5)
trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0, value=10.0)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

# Collect features into a dataframe
user_input = pd.DataFrame([[
    trip_distance, fare_amount, extra, mta_tax, tip_amount,
    tolls_amount, improvement_surcharge, congestion_surcharge,
    trip_duration, passenger_count
]], columns=feature_columns)

# Predict button
if st.button("🧮 Predict Total Fare"):
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    st.success(f"💰 Estimated Total Fare: ${prediction:.2f}")
