import joblib
import pandas as pd
import streamlit as st

# Load model and expected feature names
model, features_used = joblib.load('linear_model.pkl')

# UI
st.title("NYC Green Taxi Fare Predictor")
st.markdown("Enter trip details to predict the **total fare**.")

# Input fields
trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, format="%.2f")
passenger_count = st.number_input("Passenger Count", min_value=1, step=1)
fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, format="%.2f")
extra = st.number_input("Extra Charges ($)", min_value=0.0, format="%.2f")
mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, format="%.2f")
tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, format="%.2f")
tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, format="%.2f")
improvement_surcharge = st.number_input("Improvement Surcharge ($)", min_value=0.0, format="%.2f")
congestion_surcharge = st.number_input("Congestion Surcharge ($)", min_value=0.0, format="%.2f")
trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0, format="%.2f")

# Create input dictionary
input_dict = {
    'trip_distance': trip_distance,
    'fare_amount': fare_amount,
    'extra': extra,
    'mta_tax': mta_tax,
    'tip_amount': tip_amount,
    'tolls_amount': tolls_amount,
    'improvement_surcharge': improvement_surcharge,
    'congestion_surcharge': congestion_surcharge,
    'trip_duration': trip_duration,
    'passenger_count': passenger_count
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Reindex to match training features and fill missing ones with 0
input_df = input_df.reindex(columns=features_used).fillna(0)

# Predict
if st.button("Predict Fare"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Total Fare: ${prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
