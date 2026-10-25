import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Path setup
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
from feature_engineering import add_derived_features
from database import SessionLocal, ApplicantRecord

st.set_page_config(page_title="EMI Eligibility Prediction - EMIPredict AI", page_icon="🎯", layout="wide")

st.title("🎯 Real-Time EMI Eligibility Prediction")
st.markdown("Automated loan underwriting decisioning powered by trained classification models.")

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_classifier.joblib")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model_bundle = load_model()

with st.form("applicant_form"):
    st.subheader("1. Applicant Information & Demographics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        customer_name = st.text_input("Applicant Name", value="Rahul Sharma")
        age = st.slider("Age", min_value=21, max_value=65, value=32)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    with c3:
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    with c4:
        family_size = st.number_input("Family Size", min_value=1, max_value=12, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=8, value=1)

    st.subheader("2. Employment & Financial Income")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        monthly_salary = st.number_input("Monthly Gross Salary (INR)", min_value=10000.0, max_value=500000.0, value=75000.0, step=5000.0)
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    with e2:
        years_of_employment = st.slider("Work Experience (Years)", min_value=0.0, max_value=40.0, value=4.5, step=0.5)
        company_type = st.selectbox("Company Type", ["Startup", "Small", "Mid-size", "Large", "MNC"])
    with e3:
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0.0, max_value=2000000.0, value=180000.0, step=10000.0)
        emergency_fund = st.number_input("Emergency Savings (INR)", min_value=0.0, max_value=1500000.0, value=120000.0, step=5000.0)
    with e4:
        credit_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=850, value=740)
        existing_loans = st.selectbox("Existing Loans?", ["No", "Yes"])

    st.subheader("3. Monthly Living Expenses & Current Obligations")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        monthly_rent = st.number_input("Rent (INR)", min_value=0.0, value=12000.0, step=1000.0)
    with m2:
        school_fees = st.number_input("School Fees", min_value=0.0, value=3000.0, step=500.0)
    with m3:
        college_fees = st.number_input("College Fees", min_value=0.0, value=0.0, step=500.0)
    with m4:
        travel_expenses = st.number_input("Travel Costs", min_value=0.0, value=4000.0, step=500.0)
    with m5:
        groceries_utilities = st.number_input("Groceries & Utility", min_value=0.0, value=10000.0, step=1000.0)
    with m6:
        other_monthly_expenses = st.number_input("Other Expenses", min_value=0.0, value=3000.0, step=500.0)

    current_emi_amount = st.number_input("Current Monthly EMI Obligations (INR)", min_value=0.0, value=5000.0 if existing_loans == "Yes" else 0.0, step=1000.0)

    st.subheader("4. Loan Application Details")
    l1, l2, l3 = st.columns(3)
    with l1:
        emi_scenario = st.selectbox("Loan Scenario", [
            "E-commerce Shopping EMI",
            "Home Appliances EMI",
            "Vehicle EMI",
            "Personal Loan EMI",
            "Education EMI"
        ])
    with l2:
        requested_amount = st.number_input("Requested Loan Amount (INR)", min_value=5000.0, max_value=2500000.0, value=250000.0, step=10000.0)
    with l3:
        requested_tenure = st.number_input("Tenure (Months)", min_value=3, max_value=84, value=24)

    submit_btn = st.form_submit_button("⚡ Evaluate Loan Eligibility", use_container_width=True)

if submit_btn:
    if model_bundle is None:
        st.error("Model not trained yet! Run `python src/train_classification.py` to train and save the model.")
    else:
        # Construct input DataFrame
        input_data = {
            'age': [age],
            'gender': [gender],
            'marital_status': [marital_status],
            'education': [education],
            'monthly_salary': [monthly_salary],
            'employment_type': [employment_type],
            'years_of_employment': [years_of_employment],
            'company_type': [company_type],
            'house_type': [house_type],
            'monthly_rent': [monthly_rent],
            'family_size': [family_size],
            'dependents': [dependents],
            'school_fees': [school_fees],
            'college_fees': [college_fees],
            'travel_expenses': [travel_expenses],
            'groceries_utilities': [groceries_utilities],
            'other_monthly_expenses': [other_monthly_expenses],
            'existing_loans': [existing_loans],
            'current_emi_amount': [current_emi_amount],
            'credit_score': [credit_score],
            'bank_balance': [bank_balance],
            'emergency_fund': [emergency_fund],
            'emi_scenario': [emi_scenario],
            'requested_amount': [requested_amount],
            'requested_tenure': [requested_tenure]
        }
        df_input = pd.DataFrame(input_data)
        df_feat = add_derived_features(df_input)

        pipeline = model_bundle['pipeline']
        label_encoder = model_bundle['label_encoder']
        feature_cols = model_bundle['feature_cols']

        X_eval = df_feat[feature_cols]
        pred_encoded = pipeline.predict(X_eval)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        pred_probs = pipeline.predict_proba(X_eval)[0]

        st.markdown("---")
        st.subheader("📋 Decision Recommendation")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred_label == "Eligible":
                st.success(f"### Result: {pred_label} ✅")
                st.write("**Risk Assessment**: Low risk, strong repayment capacity.")
            elif pred_label == "High_Risk":
                st.warning(f"### Result: {pred_label} ⚠️")
                st.write("**Risk Assessment**: Marginal approval. Higher interest rate or co-applicant suggested.")
            else:
                st.error(f"### Result: {pred_label} ❌")
                st.write("**Risk Assessment**: Elevated default probability. Application not recommended.")

        with res_col2:
            prob_df = pd.DataFrame({
                "Risk Category": label_encoder.classes_,
                "Probability": [f"{p*100:.1f}%" for p in pred_probs]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

        # Quick Save to DB
        try:
            db = SessionLocal()
            record = ApplicantRecord(
                customer_name=customer_name,
                age=int(age),
                gender=gender,
                marital_status=marital_status,
                education=education,
                monthly_salary=float(monthly_salary),
                employment_type=employment_type,
                years_of_employment=float(years_of_employment),
                company_type=company_type,
                house_type=house_type,
                monthly_rent=float(monthly_rent),
                family_size=int(family_size),
                dependents=int(dependents),
                school_fees=float(school_fees),
                college_fees=float(college_fees),
                travel_expenses=float(travel_expenses),
                groceries_utilities=float(groceries_utilities),
                other_monthly_expenses=float(other_monthly_expenses),
                existing_loans=existing_loans,
                current_emi_amount=float(current_emi_amount),
                credit_score=float(credit_score),
                bank_balance=float(bank_balance),
                emergency_fund=float(emergency_fund),
                emi_scenario=emi_scenario,
                requested_amount=float(requested_amount),
                requested_tenure=int(requested_tenure),
                predicted_eligibility=pred_label
            )
            db.add(record)
            db.commit()
            db.close()
            st.toast("Applicant record successfully saved to local database!", icon="💾")
        except Exception as e:
            st.caption(f"Database sync note: {e}")
