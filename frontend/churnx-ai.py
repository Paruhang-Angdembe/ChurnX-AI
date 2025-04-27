import streamlit as st
import requests

# Set your backend API URL (update if necessary)
API_BASE_URL = "http://churnx-v103-1129043589.us-east-2.elb.amazonaws.com"

st.title("ChurnX-AI Dashboard")
st.write("Input customer features to receive churn predictions and natural language explanations.")

# Sidebar: Customer Features Inputs
st.sidebar.header("Customer Features")

tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=700.0)
senior_citizen = st.sidebar.selectbox("Senior Citizen", options=[0, 1], index=0)
contract = st.sidebar.selectbox("Contract", options=["Month-to-month", "One year", "Two year"], index=0)
internet_service = st.sidebar.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"], index=0)
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=0)
partner = st.sidebar.selectbox("Partner", options=["Yes", "No"], index=1)
dependents = st.sidebar.selectbox("Dependents", options=["Yes", "No"], index=1)
phone_service = st.sidebar.selectbox("Phone Service", options=["Yes", "No"], index=0)
multiple_lines = st.sidebar.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"], index=2)
online_security = st.sidebar.selectbox("Online Security", options=["Yes", "No", "No internet service"], index=2)
online_backup = st.sidebar.selectbox("Online Backup", options=["Yes", "No", "No internet service"], index=2)
device_protection = st.sidebar.selectbox("Device Protection", options=["Yes", "No", "No internet service"], index=2)
tech_support = st.sidebar.selectbox("Tech Support", options=["Yes", "No", "No internet service"], index=2)
streaming_tv = st.sidebar.selectbox("Streaming TV", options=["Yes", "No", "No internet service"], index=2)
streaming_movies = st.sidebar.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"], index=2)
paperless_billing = st.sidebar.selectbox("Paperless Billing", options=["Yes", "No"], index=0)
payment_method = st.sidebar.selectbox("Payment Method", options=[
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
], index=0)

# When the user clicks the Send button, build the payload and call the API
if st.sidebar.button("Send"):
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior_citizen,
        "Contract": contract,
        "InternetService": internet_service,
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }
    
    st.subheader("Input Data")
    st.json(payload)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/explain_churn", 
            json=payload, 
            params={"tuning": 1.0}
        )
        if response.ok:
            data = response.json()
            st.subheader("Prediction Results")
            st.write("Prediction:", data.get("prediction"))
            st.write("Probability:", data.get("probability"))
            st.subheader("Explanation")
            st.info(data.get("explanation"))
        else:
            st.error(f"API call failed: {response.status_code} {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")

