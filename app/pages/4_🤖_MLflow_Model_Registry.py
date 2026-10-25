import streamlit as st
import pandas as pd
import json
import os
import mlflow

st.set_page_config(page_title="Model Benchmark & Registry - EMIPredict AI", page_icon="🤖", layout="wide")

st.title("🤖 Model Registry & MLflow Benchmarks")
st.markdown("Comprehensive performance evaluation, leaderboard comparisons, and systematic artifact tracking for all 6 trained models.")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
BENCHMARK_PATH = os.path.join(MODELS_DIR, "model_benchmark_summary.json")
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")

# 1. Load benchmark JSON if available
if os.path.exists(BENCHMARK_PATH):
    with open(BENCHMARK_PATH, "r") as f:
        benchmarks = json.load(f)
else:
    benchmarks = None

st.subheader("1. Classification Models Leaderboard (`emi_eligibility`)")
st.caption("Target Performance Threshold: Accuracy > 90.0%")

if benchmarks and "classification" in benchmarks:
    clf_df = pd.DataFrame(benchmarks["classification"])
    clf_df["accuracy"] = clf_df["accuracy"].apply(lambda x: f"{x*100:.2f}%")
    clf_df["macro_f1"] = clf_df["macro_f1"].apply(lambda x: f"{x:.4f}")
    clf_df["roc_auc"] = clf_df["roc_auc"].apply(lambda x: f"{x:.4f}")
    clf_df.columns = ["Model Architecture", "Accuracy", "Macro F1-Score", "ROC-AUC (OVR)", "Description / Role"]
    st.dataframe(clf_df, use_container_width=True, hide_index=True)
else:
    st.info("Run `python src/train_classification.py` to populate classification metrics.")

st.markdown("---")

st.subheader("2. Regression Models Leaderboard (`max_monthly_emi`)")
st.caption("Target Performance Threshold: RMSE < INR 2,000")

if benchmarks and "regression" in benchmarks:
    reg_df = pd.DataFrame(benchmarks["regression"])
    reg_df["rmse"] = reg_df["rmse"].apply(lambda x: f"INR {x:,.2f}")
    reg_df["mae"] = reg_df["mae"].apply(lambda x: f"INR {x:,.2f}")
    reg_df["r2_score"] = reg_df["r2_score"].apply(lambda x: f"{x:.4f}")
    reg_df["mape"] = reg_df["mape"].apply(lambda x: f"{x*100:.2f}%")
    reg_df.columns = ["Model Architecture", "RMSE (Root Mean Sq Error)", "MAE (Mean Abs Error)", "R-Squared (Fit)", "MAPE", "Role"]
    st.dataframe(reg_df, use_container_width=True, hide_index=True)
else:
    st.info("Run `python src/train_regression.py` to populate regression metrics.")

st.markdown("---")

st.subheader("3. Production Model Artifacts & Explainability")
a_col1, a_col2 = st.columns(2)

with a_col1:
    st.markdown("**Top Feature Importances (Risk Drivers)**")
    fi_img = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    if os.path.exists(fi_img):
        st.image(fi_img, caption="Key underwriting risk features driving EMI eligibility decisions.")
    else:
        st.info("Artifact not found yet.")

with a_col2:
    st.markdown("**XGBoost Classifier Confusion Matrix**")
    cm_img = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Multi-class validation matrix across Eligible, High Risk, and Not Eligible.")
    else:
        st.info("Artifact not found yet.")

st.markdown("---")
st.subheader("4. Maximum EMI Regression Fit")
reg_img = os.path.join(ARTIFACTS_DIR, "regression_performance.png")
if os.path.exists(reg_img):
    st.image(reg_img, caption="Actual vs. Predicted Max Monthly EMI (INR) showcasing tight 1:1 regression fit.")

st.markdown("---")
st.subheader("5. Local MLflow Server Tracking")
try:
    client = mlflow.tracking.MlflowClient()
    exps = client.search_experiments()
    if exps:
        st.success(f"Connected to local MLflow tracking store. {len(exps)} experiments logged.")
        for e in exps:
            with st.expander(f"📁 MLflow Experiment: {e.name}"):
                runs = client.search_runs(experiment_ids=[e.experiment_id])
                if runs:
                    r_data = []
                    for r in runs:
                        r_data.append({
                            "Run Name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                            "Status": r.info.status,
                            **{k: round(v, 4) for k, v in r.data.metrics.items()}
                        })
                    st.dataframe(pd.DataFrame(r_data), use_container_width=True)
except Exception:
    pass

st.code("mlflow ui --port 5000", language="bash")
st.caption("Start the MLflow server in your terminal to explore real-time curves and parameter distributions.")
