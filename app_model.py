import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("vehicle_model_all_features.pkl")
encoders = joblib.load("encoders_all_features.pkl")

st.title("🚗 Vehicle Price Predictor (All Features)")

# Dictionary to collect user inputs
user_inputs = {}

# Input fields for all categorical features
for feature in encoders:
    options = encoders[feature].classes_
    user_inputs[feature] = st.selectbox(f"{feature.capitalize()}", options)

# Additional numerical inputs (you can add more if your dataset includes them)
year = st.slider("Year", 1990, 2025, 2020)
engine = st.number_input("Engine (L)", min_value=0.5, max_value=8.0, value=1.2)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, step=5000)
owners = st.number_input("No. of Previous Owners", min_value=0, max_value=10, step=1)

# Encode categorical inputs using the same encoders used during training
encoded_inputs = [encoders[feature].transform([user_inputs[feature]])[0] for feature in encoders]

# Combine all into a final input vector
input_vector = np.array([encoded_inputs + [year, engine, mileage, owners]])

# Predict price
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_vector)[0]
        st.success(f"Estimated Vehicle Price: ₹ {int(prediction)}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

