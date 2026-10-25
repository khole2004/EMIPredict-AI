import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
from feature_engineering import add_derived_features

st.set_page_config(page_title="Max EMI Calculator - EMIPredict AI", page_icon="📈", layout="wide")

st.title("📈 Maximum Monthly EMI Capacity Calculator")
st.markdown("Precision estimation of an applicant's maximum monthly repayment ceiling using trained regression models.")

REG_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_regressor.joblib")

@st.cache_resource
def load_reg_model():
    if os.path.exists(REG_MODEL_PATH):
        return joblib.load(REG_MODEL_PATH)
    return None

reg_bundle = load_reg_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Income & Obligations")
    monthly_salary = st.number_input("Monthly Salary (INR)", min_value=15000.0, max_value=500000.0, value=85000.0, step=5000.0)
    current_emi = st.number_input("Existing Monthly EMI", min_value=0.0, value=6000.0, step=1000.0)
    monthly_rent = st.number_input("Rent", min_value=0.0, value=15000.0, step=1000.0)
    groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, value=12000.0, step=1000.0)
    other_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, value=5000.0, step=500.0)

with col2:
    st.subheader("Credit & Application Details")
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=760)
    bank_balance = st.number_input("Current Bank Balance", min_value=0.0, value=250000.0, step=10000.0)
    emergency_fund = st.number_input("Emergency Savings", min_value=0.0, value=100000.0, step=10000.0)
    emi_scenario = st.selectbox("Lending Product", [
        "Personal Loan EMI",
        "Vehicle EMI",
        "Home Appliances EMI",
        "E-commerce Shopping EMI",
        "Education EMI"
    ])
    requested_amount = st.number_input("Requested Loan Amount", min_value=10000.0, value=400000.0, step=10000.0)
    requested_tenure = st.number_input("Repayment Period (Months)", min_value=6, max_value=84, value=36)

calc_btn = st.button("📊 Calculate Safe EMI Ceiling", use_container_width=True)

if calc_btn:
    if reg_bundle is None:
        st.error("Regression model not trained yet! Run `python src/train_regression.py` first.")
    else:
        df_input = pd.DataFrame([{
            'age': 32,
            'gender': 'Male',
            'marital_status': 'Married',
            'education': 'Graduate',
            'monthly_salary': monthly_salary,
            'employment_type': 'Private',
            'years_of_employment': 5.0,
            'company_type': 'Mid-size',
            'house_type': 'Rented',
            'monthly_rent': monthly_rent,
            'family_size': 3,
            'dependents': 1,
            'school_fees': 0.0,
            'college_fees': 0.0,
            'travel_expenses': 3000.0,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_expenses,
            'existing_loans': 'Yes' if current_emi > 0 else 'No',
            'current_emi_amount': current_emi,
            'credit_score': credit_score,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario,
            'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }])

        df_feat = add_derived_features(df_input)
        pipeline = reg_bundle['pipeline']
        feature_cols = reg_bundle['feature_cols']

        pred_emi = pipeline.predict(df_feat[feature_cols])[0]
        pred_emi = float(np.clip(pred_emi, 500.0, 50000.0))

        st.markdown("---")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Recommended Max Safe EMI", f"₹{pred_emi:,.2f}")
        with m_col2:
            disposable = float(df_feat['disposable_income'].iloc[0])
            st.metric("Current Disposable Income", f"₹{disposable:,.2f}")
        with m_col3:
            dti = float(df_feat['dti_ratio'].iloc[0]) * 100
            st.metric("Current Debt-to-Income", f"{dti:.1f}%")

        st.progress(min(max(pred_emi / (disposable + 1e-5), 0.0), 1.0))
        st.caption(f"Estimated Installment Commitment: {pred_emi / (disposable + 1e-5) * 100:.1f}% of disposable earnings.")
