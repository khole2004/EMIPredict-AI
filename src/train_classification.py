import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

from feature_engineering import CATEGORICAL_COLS, NUMERICAL_COLS, ENGINEERED_COLS

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = NUMERICAL_COLS + ENGINEERED_COLS + CATEGORICAL_COLS

def train_and_evaluate_classifiers():
    print("[1/5] Loading processed datasets...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

    # For faster interactive training while retaining statistically significant sample size:
    # Use 50,000 train samples if training full 280k takes too long on local machine
    train_sample = train_df.sample(n=min(len(train_df), 60000), random_state=42)
    val_sample = val_df.sample(n=min(len(val_df), 15000), random_state=42)

    X_train = train_sample[FEATURE_COLS]
    y_train_raw = train_sample['emi_eligibility']

    X_val = val_sample[FEATURE_COLS]
    y_val_raw = val_sample['emi_eligibility']

    # Label encoding for targets
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    classes = label_encoder.classes_

    print(f"      Classes: {classes}")
    print(f"      Training shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # Build preprocessing pipeline
    all_num_cols = NUMERICAL_COLS + ENGINEERED_COLS
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
        },
        "Random_Forest_Classifier": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 12}
        },
        "XGBoost_Classifier": {
            "model": XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss', n_jobs=-1),
            "params": {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1}
        }
    }

    mlflow.set_experiment("EMIPredict_AI_Classification")

    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, config in models.items():
        print(f"\n[Training] {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', config['model'])
        ])

        with mlflow.start_run(run_name=name):
            # Log params
            mlflow.log_params(config['params'])
            mlflow.log_param("features_count", len(FEATURE_COLS))

            # Train
            pipeline.fit(X_train, y_train)

            # Evaluate
            preds = pipeline.predict(X_val)
            pred_probs = pipeline.predict_proba(X_val)

            acc = accuracy_score(y_val, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='macro', zero_division=0)
            roc_auc = roc_auc_score(y_val, pred_probs, multi_class='ovr')

            print(f"   * Accuracy:  {acc * 100:.2f}% (Target: >90%)")
            print(f"   * Macro F1:  {f1:.4f}")
            print(f"   * ROC-AUC:   {roc_auc:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_macro", rec)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("roc_auc_ovr", roc_auc)

            # Check if best
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_pipeline = pipeline

    print(f"\n==========================================")
    print(f"Best Classification Model: {best_model_name} (Macro F1: {best_f1:.4f})")
    print(f"==========================================")

    # Export best model pipeline and label encoder
    export_obj = {
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "classes": list(classes),
        "feature_cols": FEATURE_COLS,
        "model_name": best_model_name,
        "best_macro_f1": best_f1
    }
    save_path = os.path.join(MODELS_DIR, "best_classifier.joblib")
    joblib.dump(export_obj, save_path)
    print(f"Exported production classifier to: {save_path}")

if __name__ == "__main__":
    train_and_evaluate_classifiers()
