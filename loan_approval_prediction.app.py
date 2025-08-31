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

    # 👤 Personal Information
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100)
        person_gender = st.selectbox("Gender", ["", "male", "female"])
        person_education = st.selectbox(
            "Education", 
            ["", "High School", "Bachelor", "Master", "Associate", "Doctorate"]
        )
    with col2:
        person_income = st.number_input("Annual Income (USD)", min_value=0, max_value=500000)
        person_emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=50)
        person_home_ownership = st.selectbox(
            "Home Ownership", 
            ["", "Rent", "Own", "Mortgage", "Other"]
        )

    # 💰 Loan Information
    st.subheader("💰 Loan Information")
    col3, col4 = st.columns(2)
    with col3:
        loan_amnt = st.number_input("Loan Amount", min_value=0, max_value=100000)
        loan_intent = st.selectbox(
            "Loan Intent",
            ["", "EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
        )
    with col4:
        loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=1.0, max_value=40.0, step=0.1)

    # 🔥 Live Loan Percent Income
    if person_income > 0 and loan_amnt > 0:
        loan_percent_income = loan_amnt / person_income
        st.info(f"📊 Loan Percent Income: **{loan_percent_income:.2f}**")
    else:
        loan_percent_income = 0

    # 📊 Credit Information
    st.subheader("📊 Credit Information")
    col5, col6 = st.columns(2)
    with col5:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    with col6:
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["", "Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# ==============================
# Prediction
# ==============================
if submitted:
    # Ensure all required inputs are filled
    if (person_age and person_income and loan_amnt and credit_score
        and person_gender and person_education and person_home_ownership
        and loan_intent and previous_loan_defaults_on_file):
        
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
    else:
        st.warning("⚠️ Please fill in all required fields before predicting.")
