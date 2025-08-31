import streamlit as st
import pandas as pd
import joblib

# ==============================
# Load Model
# ==============================
model = joblib.load("loan_approval_rfmodel.joblib")

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Fill in the details below to check loan approval status.")

# ==============================
# Input Form
# ==============================
with st.form("loan_form"):
    st.subheader("👤 Personal Information")
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
    person_income = st.number_input("Annual Income (USD)", min_value=1000, max_value=500000, value=50000)
    person_emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=50, value=5)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    st.subheader("💰 Loan Information")
    loan_amnt = st.number_input("Loan Amount", min_value=500, max_value=100000, value=10000)
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

    st.subheader("📊 Credit Information")
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# ==============================
# Prediction
# ==============================
if submitted:
    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]

    # Display results
    if prediction == 1:
        st.success(f"✅ Loan Approved with probability {proba:.2f}")
    else:
        st.error(f"❌ Loan Rejected with probability {proba:.2f}")
