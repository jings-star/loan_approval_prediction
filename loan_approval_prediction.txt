import streamlit as st
import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load("loan_approval_rfmodel.joblib")

st.title("💳 Loan Approval Prediction Prototype")

st.write("Fill in the applicant details to predict loan approval:")

# User input form
with st.form("loan_form"):
    age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Applicant Income", min_value=1000, max_value=500000, value=50000)
    emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=100000, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0, max_value=50.0, value=12.0)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2)
    cb_default = st.selectbox("Default History (0 = No, 1 = Yes)", [0,1])
    cred_hist_len = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)
    
    submit = st.form_submit_button("🔍 Predict Loan Approval")

# Prepare input for prediction
if submit:
    applicant = {
        "person_age": age,
        "person_income": income,
        "person_emp_length": emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_default,
        "cb_person_cred_hist_length": cred_hist_len
    }
    
    input_df = pd.DataFrame([applicant])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.success(f"✅ Loan Approved (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Approval Probability: {probability:.2f})")
