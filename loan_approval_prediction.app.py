import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load Model (pipeline)
# ===========================
@st.cache_resource
def load_model():
    return joblib.load("loan_approval_rfmodel.joblib")

model = load_model()

# --- Extract feature names (from training) ---
trained_features = model.feature_names_in_

# ===========================
# App Title
# ===========================
st.title("🏦 Loan Approval Prediction")
st.write("Fill in applicant details below to check loan approval probability.")

# ===========================
# Input Form
# ===========================
with st.form("loan_form"):

    col1, col2 = st.columns(2)
    gender = col1.selectbox("Gender", ["male", "female"])
    age = col2.number_input("Age", min_value=18, max_value=100, step=1)

    col1, col2 = st.columns(2)
    education = col1.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"])
    emp_length = col2.number_input("Years of Employment", step=1, min_value=0)

    col1, col2 = st.columns(2)
    home = col1.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_intent = col2.selectbox(
        "Loan Intent",
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    )

    col1, col2 = st.columns(2)
    previous_default = col1.selectbox("Previous Loan Default", ["Y", "N"])
    cred_hist_length = col2.number_input("Credit History Length (years)", step=1, min_value=0)

    col1, col2 = st.columns(2)
    income = col1.number_input("Annual Income ($)", step=500, min_value=0)
    loan_amount = col2.number_input("Loan Amount ($)", step=500, min_value=0)

    loan_percent_income = 0
    if income > 0 and loan_amount > 0:
        loan_percent_income = loan_amount / income
        st.info(f"📊 Loan Percent Income: **{loan_percent_income:.2f}**")

    loan_int_rate = st.number_input("Loan Interest Rate (%)", step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔮 Predict Loan Approval")

# ===========================
# Prediction
# ===========================
if submitted:
    # Create dictionary with EXACT feature names from training
    input_dict = {
        "person_gender": gender,
        "person_age": age,
        "person_income": income,
        "person_emp_length": emp_length,
        "person_education": education,
        "person_home_ownership": home,
        "loan_intent": loan_intent,
        "loan_amnt": loan_amount,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "previous_loan_defaults_on_file": previous_default,
        "cb_person_cred_hist_length": cred_hist_length,
    }

    # Align DataFrame with training features
    input_data = pd.DataFrame([input_dict])[list(trained_features)]

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Loan Rejected. (Confidence: {probability:.2%})")
