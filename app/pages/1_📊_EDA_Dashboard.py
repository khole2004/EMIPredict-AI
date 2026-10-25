import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EDA Dashboard - EMIPredict AI", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis & Risk Patterns")
st.markdown("Interactive exploration of financial distributions, DTI ratios, and approval rates across 400K applicant profiles.")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "train.parquet")
RAW_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "emi_prediction_dataset.csv")

@st.cache_data
def load_sample_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
    elif os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH, nrows=50000, low_memory=False)
    else:
        return None
    return df.sample(n=min(len(df), 25000), random_state=42)

df = load_sample_data()

if df is None:
    st.warning("Processed dataset not found yet. Please run the data preprocessing step first!")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Loan Eligibility Distribution")
    fig_pie = px.pie(
        df,
        names="emi_eligibility",
        title="Applicant Eligibility Breakdown",
        color="emi_eligibility",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("2. Eligibility by EMI Scenario")
    fig_bar = px.histogram(
        df,
        x="emi_scenario",
        color="emi_eligibility",
        barmode="group",
        title="Distribution Across Lending Products",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Credit Score vs. Monthly Salary")
    sample_scatter = df.sample(n=min(len(df), 3000), random_state=42)
    fig_scatter = px.scatter(
        sample_scatter,
        x="credit_score",
        y="monthly_salary",
        color="emi_eligibility",
        opacity=0.6,
        hover_data=["education", "employment_type"],
        title="Credit Score vs. Income Level",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    st.subheader("4. Max EMI Capacity vs. Disposable Income")
    fig_reg = px.scatter(
        sample_scatter,
        x="monthly_salary",
        y="max_monthly_emi",
        color="emi_eligibility",
        opacity=0.6,
        title="Safe EMI Ceiling vs. Monthly Salary",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_reg, use_container_width=True)
