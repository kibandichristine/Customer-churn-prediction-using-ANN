import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load the trained model
# For Keras models (e.g., TensorFlow), use load_model()
# If using scikit-learn model, use pickle.load()

try:
    model = tf.keras.models.load_model('trained_model.sav.h5')
    model_type = "TensorFlow/Keras Model"
except Exception as e:
    with open('trained_model.sav', 'rb') as f:
        model = pickle.load(f)
    model_type = "scikit-learn Model"

# Verify model type
st.write(f"Model type: {model_type}")

# App title
st.title("Customer Churn Prediction")

# Collect user input with readable options
st.header("Enter Customer Details:")

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])

# Senior Citizen
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

# Partner
partner = st.selectbox("Partner", ["No", "Yes"])

# Dependents
dependents = st.selectbox("Dependents", ["No", "Yes"])

# Tenure (in months)
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)

# Phone Service
phone_service = st.selectbox("Phone Service", ["No", "Yes"])

# Multiple Lines
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

# Online Security
online_security = st.selectbox("Online Security", ["No", "Yes"])

# Online Backup
online_backup = st.selectbox("Online Backup", ["No", "Yes"])

# Device Protection
device_protection = st.selectbox("Device Protection", ["No", "Yes"])

# Tech Support
tech_support = st.selectbox("Tech Support", ["No", "Yes"])

# Streaming TV
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])

# Streaming Movies
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

# Paperless Billing
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

# Monthly Charges
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

# Total Charges
total_charges = st.number_input("Total Charges", min_value=0.0)

# One-hot encoded inputs for categorical fields
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# Map inputs to the one-hot encoded format for the model
gender_map = {"Male": 0, "Female": 1}
senior_citizen_map = {"No": 0, "Yes": 1}
partner_map = {"No": 0, "Yes": 1}
dependents_map = {"No": 0, "Yes": 1}
phone_service_map = {"No": 0, "Yes": 1}
multiple_lines_map = {"No": 0, "Yes": 1}
online_security_map = {"No": 0, "Yes": 1}
online_backup_map = {"No": 0, "Yes": 1}
device_protection_map = {"No": 0, "Yes": 1}
tech_support_map = {"No": 0, "Yes": 1}
streaming_tv_map = {"No": 0, "Yes": 1}
streaming_movies_map = {"No": 0, "Yes": 1}
paperless_billing_map = {"No": 0, "Yes": 1}

internet_service_map = {"DSL": 1, "Fiber optic": 2, "No": 0}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

# Prepare the input data for prediction
input_data = np.array([[ 
    gender_map[gender], senior_citizen_map[senior_citizen], partner_map[partner], dependents_map[dependents], 
    tenure, phone_service_map[phone_service], multiple_lines_map[multiple_lines],
    online_security_map[online_security], online_backup_map[online_backup], device_protection_map[device_protection],
    tech_support_map[tech_support], streaming_tv_map[streaming_tv], streaming_movies_map[streaming_movies],
    paperless_billing_map[paperless_billing], monthly_charges, total_charges,
    internet_service_map[internet_service], contract_map[contract], payment_map[payment_method]
]])

# Prediction button
if st.button("Predict Churn"):
    try:
        # Check if the model has a predict method (for TensorFlow/Keras)
        if hasattr(model, 'predict'):
            # For Keras/TensorFlow model, predict will return probabilities
            prediction_proba = model.predict(input_data)
            churn_probability = prediction_proba[0][0] if model_type == "TensorFlow/Keras Model" else prediction_proba[0][1]
            result = "Churn" if churn_probability > 0.5 else "No Churn"
            st.subheader(f"Prediction: {result} with probability {churn_probability:.2f}")
        else:
            st.error("The model doesn't have a predict method.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
