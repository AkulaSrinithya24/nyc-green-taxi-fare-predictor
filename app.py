import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("linear_model.pkl")

# App title and styling
st.set_page_config(page_title="NYC Green Taxi Fare Predictor", layout="wide")

with st.container():
    st.markdown("<h1 style='text-align: center; color: #3C6E71;'>🚖 NYC Green Taxi Fare Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")

# Input Section
st.subheader("📥 Enter Ride Details")

col1, col2, col3 = st.columns(3)
with col1:
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=50.0, value=1.5)
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    hourofday = st.selectbox("Hour of Day (Dropoff)", list(range(0, 24)))

with col2:
    fare_amount = st.number_input("Fare Amount ($)", min_value=0.0, value=10.0)
    mta_tax = st.number_input("MTA Tax ($)", min_value=0.0, value=0.5)
    congestion_surcharge = st.number_input("Congestion Surcharge ($)", min_value=0.0, value=2.5)

with col3:
    tip_amount = st.number_input("Tip Amount ($)", min_value=0.0, value=2.0)
    tolls_amount = st.number_input("Tolls Amount ($)", min_value=0.0, value=0.0)
    trip_duration = st.number_input("Trip Duration (minutes)", min_value=1, max_value=300, value=15)

# Prediction
if st.button("Predict Fare"):
    input_features = np.array([[trip_distance, fare_amount, mta_tax, tip_amount,
                                tolls_amount, congestion_surcharge, trip_duration,
                                passenger_count, hourofday]])
    prediction = model.predict(input_features)[0]
    st.success(f"🚕 Predicted Total Fare: ${prediction:.2f}")

    # Optional: Visual analysis
    st.subheader("🔍 Prediction Breakdown")
    feature_labels = ["Distance", "Fare", "MTA", "Tip", "Tolls", "Cong. Surcharge", "Duration", "Passengers", "Hour"]
    fig, ax = plt.subplots()
    sns.barplot(x=feature_labels, y=input_features[0], palette="YlGnBu", ax=ax)
    ax.set_ylabel("Value")
    ax.set_title("Input Feature Contribution")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ for Internal Assessment", unsafe_allow_html=True)
