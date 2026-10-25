import streamlit as st
import pandas as pd
import os
import sys

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

st.set_page_config(
    page_title="EMIPredict AI - Financial Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment & EMI Optimization Platform")

st.markdown("""
Welcome to **EMIPredict AI**, a production-grade machine learning platform for **FinTech & Banking Institutions**.
It provides instant automated loan risk assessment, loan eligibility classification, and maximum safe EMI amount recommendation.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Profiles Analyzed", "400,000+", delta="5 Lending Scenarios")
with col2:
    st.metric("Target Classification Accuracy", "> 90.0%", delta="3-Class Risk Tier")
with col3:
    st.metric("Target Regression RMSE", "< ₹2,000", delta="₹500 - ₹50K Safe EMI")
with col4:
    st.metric("Underwriting Speedup", "~80%", delta="Real-time Decisions")

st.markdown("---")

st.markdown("### 📌 Platform Navigation")

n_col1, n_col2 = st.columns(2)

with n_col1:
    st.info("""
    **1. 📊 EDA Dashboard**
    * Deep-dive into financial variables, income distributions, and credit history correlations.
    * Interactive risk distribution across the 5 lending scenarios.
    
    **2. 🎯 EMI Eligibility Prediction (Classification)**
    * Real-time risk assessment for walk-in or digital loan applicants.
    * Output: `Eligible`, `High_Risk`, or `Not_Eligible` with confidence scoring.
    """)

with n_col2:
    st.info("""
    **3. 📈 Max EMI Calculator (Regression)**
    * Computes safe monthly repayment capacity based on disposable income & obligations.
    
    **4. 🤖 MLflow Model Registry**
    * Compare model metrics, confusion matrices, and experiment runs.
    
    **5. 🗄️ Data Management (CRUD)**
    * Administrative portal to add, search, update, or remove applicant records.
    """)

st.markdown("---")
st.markdown("### 🏢 Covered Lending Scenarios")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.write("**🛒 E-commerce Shopping**")
    st.caption("₹10K - ₹200K | 3–24 mos")
with c2:
    st.write("**📺 Home Appliances**")
    st.caption("₹20K - ₹300K | 6–36 mos")
with c3:
    st.write("**🚗 Vehicle Loans**")
    st.caption("₹80K - ₹1.5M | 12–84 mos")
with c4:
    st.write("**💼 Personal Loans**")
    st.caption("₹50K - ₹1.0M | 12–60 mos")
with c5:
    st.write("**🎓 Education Loans**")
    st.caption("₹50K - ₹500K | 6–48 mos")
