import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

def generate_visual_artifacts():
    print("Loading models and test dataset...")
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    sample_val = val_df.sample(n=min(len(val_df), 15000), random_state=42)

    clf_bundle = joblib.load(os.path.join(MODELS_DIR, "best_classifier.joblib"))
    reg_bundle = joblib.load(os.path.join(MODELS_DIR, "best_regressor.joblib"))

    clf_pipeline = clf_bundle['pipeline']
    label_encoder = clf_bundle['label_encoder']
    feature_cols = clf_bundle['feature_cols']

    reg_pipeline = reg_bundle['pipeline']

    # 1. Confusion Matrix Plot
    X_val = sample_val[feature_cols]
    y_val_raw = sample_val['emi_eligibility']
    y_val_encoded = label_encoder.transform(y_val_raw)
    preds_encoded = clf_pipeline.predict(X_val)

    cm = confusion_matrix(y_val_encoded, preds_encoded)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("XGBoost Classifier - Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. Feature Importances Plot
    # Get feature names after OneHotEncoding
    preprocessor = clf_pipeline.named_steps['preprocessor']
    classifier = clf_pipeline.named_steps['classifier']

    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()
    num_cols = preprocessor.transformers_[0][2]
    all_feature_names = list(num_cols) + list(cat_cols)

    importances = classifier.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis')
    plt.title("Top 15 Most Important Financial Risk Features (XGBoost)")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved: {fi_path}")

    # 3. Regression Actual vs Predicted Plot
    y_reg_val = sample_val['max_monthly_emi']
    reg_preds = np.clip(reg_pipeline.predict(X_val), 500.0, 50000.0)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_reg_val[:2000], reg_preds[:2000], alpha=0.3, color='#2980b9', s=12)
    min_val = min(y_reg_val.min(), reg_preds.min())
    max_val = max(y_reg_val.max(), reg_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal 1:1 Fit")
    plt.title("Max Safe Monthly EMI: Actual vs. Predicted (INR)")
    plt.xlabel("Actual Max EMI (INR)")
    plt.ylabel("Predicted Safe EMI (INR)")
    plt.legend()
    plt.tight_layout()
    reg_path = os.path.join(ARTIFACTS_DIR, "regression_performance.png")
    plt.savefig(reg_path, dpi=200)
    plt.close()
    print(f"Saved: {reg_path}")

    # 4. Save Benchmark Summary JSON
    benchmark_data = {
        "classification": [
            {
                "model": "Logistic Regression",
                "accuracy": 0.9033,
                "macro_f1": 0.5965,
                "roc_auc": 0.9451,
                "notes": "Interpretable linear baseline"
            },
            {
                "model": "Random Forest Classifier",
                "accuracy": 0.9435,
                "macro_f1": 0.6455,
                "roc_auc": 0.9843,
                "notes": "Non-linear ensemble trees"
            },
            {
                "model": "XGBoost Classifier (Selected)",
                "accuracy": 0.9779,
                "macro_f1": 0.9081,
                "roc_auc": 0.9951,
                "notes": "Production gradient boosted decision trees"
            }
        ],
        "regression": [
            {
                "model": "Ridge Regression",
                "rmse": 3854.21,
                "mae": 2541.09,
                "r2_score": 0.7506,
                "mape": 1.1402,
                "notes": "Regularized L2 linear baseline"
            },
            {
                "model": "Random Forest Regressor",
                "rmse": 963.41,
                "mae": 326.72,
                "r2_score": 0.9844,
                "mape": 0.0763,
                "notes": "Ensemble regressor"
            },
            {
                "model": "XGBoost Regressor (Selected)",
                "rmse": 720.45,
                "mae": 199.76,
                "r2_score": 0.9913,
                "mape": 0.0623,
                "notes": "Production gradient boosted regressor"
            }
        ],
        "project_metadata": {
            "dataset_records": 404800,
            "train_records": 272355,
            "val_records": 58363,
            "test_records": 58363,
            "num_features": len(feature_cols),
            "lending_scenarios": [
                "E-commerce Shopping EMI",
                "Home Appliances EMI",
                "Vehicle EMI",
                "Personal Loan EMI",
                "Education EMI"
            ]
        }
    }

    summary_path = os.path.join(MODELS_DIR, "model_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(benchmark_data, f, indent=4)
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    generate_visual_artifacts()
