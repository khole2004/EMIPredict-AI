import streamlit as st
import pandas as pd
import mlflow
import os

st.set_page_config(page_title="MLflow Model Registry - EMIPredict AI", page_icon="🤖", layout="wide")

st.title("🤖 MLflow Model Performance & Registry")
st.markdown("Centralized experiment tracking, model comparison leaderboard, and production registry.")

# Check for MLflow experiments
try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
except Exception as e:
    experiments = []

if not experiments:
    st.info("No active MLflow runs detected yet. Run model training (`python src/train_classification.py` and `python src/train_regression.py`) to log experiment metrics.")
else:
    for exp in experiments:
        st.subheader(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        if runs:
            data = []
            for r in runs:
                row = {
                    "Run Name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                    "Status": r.info.status,
                    **{k: round(v, 4) for k, v in r.data.metrics.items()},
                    **r.data.params
                }
                data.append(row)
            df_runs = pd.DataFrame(data)
            st.dataframe(df_runs, use_container_width=True)
        else:
            st.caption("No completed runs in this experiment.")

st.markdown("---")
st.subheader("💡 Launch MLflow Interactive Dashboard")
st.code("mlflow ui --port 5000", language="bash")
st.caption("Run the above command in your terminal to open the full interactive MLflow comparison dashboard at http://localhost:5000.")
