import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the model, features, and scaler
model, features_used, scaler = joblib.load('linear_model.pkl')

# Set page configuration
st.set_page_config(
    page_title="NYC Green Taxi Fare Predictor",
    page_icon="🚖",
    layout="centered"
)

# Title and description
st.title("🚖 NYC Green Taxi Fare Predictor")
st.markdown("Enter trip details to predict the **total fare**.")

# Create a form for user inputs
with st.form("fare_prediction_form"):
    # Create columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0, format="%.2f")
        passenger_count = st.number_input("Passenger Count", min_value=1, step=1)
        fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, format="%.2f")
        extra = st.number_input("Extra Charges ($)", min_value=0.0, format="%.2f")
        mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, format="%.2f")

    with col2:
        tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, format="%.2f")
        tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, format="%.2f")
        improvement_surcharge = st.number_input("Improvement Surcharge ($)", min_value=0.0, format="%.2f")
        congestion_surcharge = st.number_input("Congestion Surcharge ($)", min_value=0.0, format="%.2f")
        trip_duration = st.number_input("Trip Duration (minutes)", min_value=0.0, format="%.2f")

    # Submit button
    submitted = st.form_submit_button("Predict Fare")

# Process the input and make prediction
if submitted:
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

    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_df)

    # Predict
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted Total Fare: ${prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
