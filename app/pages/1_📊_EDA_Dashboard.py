import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EDA Dashboard - EMIPredict AI", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis & Risk Patterns")
st.markdown("Interactive exploration of financial distributions, DTI ratios, and approval rates across applicant profiles.")

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "train.parquet")
RAW_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "emi_prediction_dataset.csv")
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample_data.parquet")

@st.cache_data
def load_sample_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
    elif os.path.exists(SAMPLE_PATH):
        df = pd.read_parquet(SAMPLE_PATH)
    elif os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH, nrows=50000, low_memory=False)
    else:
        return None
    return df.sample(n=min(len(df), 25000), random_state=42)

df = load_sample_data()

if df is None:
    st.warning("Processed dataset not found yet. Please run data preprocessing first!")
    st.stop()

# Scenario filter in sidebar or top
scenarios = ["All Lending Scenarios"] + list(df['emi_scenario'].dropna().unique())
selected_scenario = st.selectbox("🎯 Filter by Lending Scenario:", scenarios)

if selected_scenario != "All Lending Scenarios":
    filtered_df = df[df['emi_scenario'] == selected_scenario]
else:
    filtered_df = df

# Portfolio KPIs
k1, k2, k3, k4 = st.columns(4)
total_apps = len(filtered_df)
approval_rate = (filtered_df['emi_eligibility'] == 'Eligible').mean() * 100
avg_income = filtered_df['monthly_salary'].mean()
avg_credit = filtered_df['credit_score'].mean()

with k1:
    st.metric("Analyzed Profiles", f"{total_apps:,}")
with k2:
    st.metric("Eligible Approval Rate", f"{approval_rate:.1f}%")
with k3:
    st.metric("Average Salary", f"₹{avg_income:,.0f}")
with k4:
    st.metric("Average Credit Score", f"{avg_credit:.0f}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Loan Eligibility Breakdown")
    fig_pie = px.pie(
        filtered_df,
        names="emi_eligibility",
        title=f"Eligibility Share ({selected_scenario})",
        color="emi_eligibility",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("2. Eligibility by Product Type")
    fig_bar = px.histogram(
        df,
        x="emi_scenario",
        color="emi_eligibility",
        barmode="group",
        title="Comparison Across 5 Lending Products",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Demographic Patterns: Education vs. Eligibility")
    fig_edu = px.histogram(
        filtered_df,
        x="education",
        color="emi_eligibility",
        barmode="stack",
        title="Approval Distribution by Education Qualification",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_edu, use_container_width=True)

with col4:
    st.subheader("4. Credit Score vs. Monthly Salary")
    sample_scatter = filtered_df.sample(n=min(len(filtered_df), 3000), random_state=42)
    fig_scatter = px.scatter(
        sample_scatter,
        x="credit_score",
        y="monthly_salary",
        color="emi_eligibility",
        opacity=0.6,
        hover_data=["education", "employment_type"],
        title="Credit Worthiness & Income Level Separation",
        color_discrete_map={"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

st.subheader("5. Correlation Heatmap of Financial Variables")
corr_cols = [
    'monthly_salary', 'credit_score', 'bank_balance', 'emergency_fund',
    'monthly_rent', 'current_emi_amount', 'requested_amount', 'max_monthly_emi'
]
corr_cols_present = [c for c in corr_cols if c in filtered_df.columns]
corr_matrix = filtered_df[corr_cols_present].corr()

fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Financial Risk & EMI Correlation Matrix"
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.subheader("6. Statistical Summaries by Risk Tier")
summary_table = filtered_df.groupby("emi_eligibility")[['monthly_salary', 'credit_score', 'max_monthly_emi', 'bank_balance']].agg(['mean', 'median', 'std'])
st.dataframe(summary_table, use_container_width=True)
