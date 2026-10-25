import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src"))
from feature_engineering import add_derived_features

st.set_page_config(page_title="Max EMI Calculator - EMIPredict AI", page_icon="📈", layout="wide")

st.title("📈 Maximum Monthly EMI & Loan Affordability Simulator")
st.markdown("Precision estimation of an applicant's maximum monthly repayment ceiling combined with dynamic loan repayment stress-testing.")

REG_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "best_regressor.joblib")

@st.cache_resource
def load_reg_model():
    if os.path.exists(REG_MODEL_PATH):
        return joblib.load(REG_MODEL_PATH)
    return None

reg_bundle = load_reg_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Applicant Financial Profile")
    monthly_salary = st.number_input("Monthly Salary (INR)", min_value=15000.0, max_value=500000.0, value=85000.0, step=5000.0)
    current_emi = st.number_input("Existing Monthly EMI Burden", min_value=0.0, value=6000.0, step=1000.0)
    monthly_rent = st.number_input("Monthly Rent", min_value=0.0, value=15000.0, step=1000.0)
    groceries_utilities = st.number_input("Groceries & Utility Bills", min_value=0.0, value=12000.0, step=1000.0)
    other_expenses = st.number_input("Other Living Expenses", min_value=0.0, value=5000.0, step=500.0)
    credit_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=850, value=760)

with col2:
    st.subheader("2. Proposed Loan & Stress-Test Simulation")
    emi_scenario = st.selectbox("Lending Category", [
        "Personal Loan EMI",
        "Vehicle EMI",
        "Home Appliances EMI",
        "E-commerce Shopping EMI",
        "Education EMI"
    ])
    requested_amount = st.number_input("Desired Loan Principal (INR)", min_value=10000.0, max_value=2500000.0, value=400000.0, step=10000.0)
    requested_tenure = st.slider("Loan Tenure (Months)", min_value=6, max_value=84, value=36)
    annual_interest_rate = st.slider("Expected Annual Interest Rate (%)", min_value=5.0, max_value=24.0, value=11.5, step=0.25)
    bank_balance = st.number_input("Current Bank Balance", min_value=0.0, value=250000.0, step=10000.0)
    emergency_fund = st.number_input("Emergency Savings", min_value=0.0, value=100000.0, step=10000.0)

calc_btn = st.button("📊 Calculate Max Safe EMI & Stress Test Loan", use_container_width=True)

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

        # Safe Max EMI from AI model
        pred_max_emi = pipeline.predict(df_feat[feature_cols])[0]
        pred_max_emi = float(np.clip(pred_max_emi, 500.0, 50000.0))

        # Standard Banking EMI Formula: E = P * r * (1+r)^n / ((1+r)^n - 1)
        monthly_rate = (annual_interest_rate / 100.0) / 12.0
        n_months = int(requested_tenure)
        if monthly_rate > 0:
            loan_installment = requested_amount * monthly_rate * ((1 + monthly_rate) ** n_months) / (((1 + monthly_rate) ** n_months) - 1)
        else:
            loan_installment = requested_amount / n_months

        st.markdown("---")
        st.subheader("💡 Affordability Analysis & Decision")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Predicted Safe EMI Ceiling", f"₹{pred_max_emi:,.2f}")
            st.caption("Maximum recommended monthly burden")
        with m2:
            st.metric("Proposed Loan Monthly Installment", f"₹{loan_installment:,.2f}")
            st.caption(f"{n_months} months @ {annual_interest_rate:.1f}% interest")
        with m3:
            diff = pred_max_emi - loan_installment
            if diff >= 0:
                st.metric("Monthly Affordability Buffer", f"+₹{diff:,.2f}", delta="Comfortable")
            else:
                st.metric("Monthly Deficit / Over-commitment", f"-₹{abs(diff):,.2f}", delta="-Deficit", delta_color="inverse")

        # Affordability badge and recommendations
        if loan_installment <= pred_max_emi:
            st.success(f"""
            ✅ **Loan Affordable**: The required installment of **₹{loan_installment:,.2f}** is well within the applicant's safe monthly capacity of **₹{pred_max_emi:,.2f}**.
            * Remaining monthly buffer: **₹{diff:,.2f}**
            * Recommended Action: **Approve requested amount**.
            """)
        else:
            suggested_tenure = requested_tenure + 12
            st.warning(f"""
            ⚠️ **EMI Stress Warning**: The requested installment of **₹{loan_installment:,.2f}** exceeds the safe limit by **₹{abs(diff):,.2f}**.
            * **Underwriter Recommendations**:
              1. Extend repayment tenure to **{suggested_tenure} months** to reduce monthly obligation.
              2. Or cap the loan amount to **₹{int(pred_max_emi * (( (1+monthly_rate)**n_months - 1 ) / (monthly_rate * (1+monthly_rate)**n_months))):,}**.
            """)
