import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from feature_engineering import CATEGORICAL_COLS, NUMERICAL_COLS, ENGINEERED_COLS

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = NUMERICAL_COLS + ENGINEERED_COLS + CATEGORICAL_COLS

def train_and_evaluate_regressors():
    print("[1/5] Loading processed datasets for Regression...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet"))

    train_sample = train_df.sample(n=min(len(train_df), 60000), random_state=42)
    val_sample = val_df.sample(n=min(len(val_df), 15000), random_state=42)

    X_train = train_sample[FEATURE_COLS]
    y_train = train_sample['max_monthly_emi']

    X_val = val_sample[FEATURE_COLS]
    y_val = val_sample['max_monthly_emi']

    all_num_cols = NUMERICAL_COLS + ENGINEERED_COLS
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), all_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    models = {
        "Ridge_Regression": {
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0}
        },
        "Random_Forest_Regressor": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 12}
        },
        "XGBoost_Regressor": {
            "model": XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
            "params": {"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1}
        }
    }

    mlflow.set_experiment("EMIPredict_AI_Regression")

    best_model_name = None
    best_rmse = float('inf')
    best_pipeline = None

    for name, config in models.items():
        print(f"\n[Training] {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', config['model'])
        ])

        with mlflow.start_run(run_name=name):
            mlflow.log_params(config['params'])
            mlflow.log_param("features_count", len(FEATURE_COLS))

            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_val)
            preds = np.clip(preds, 500.0, 50000.0)

            rmse = root_mean_squared_error(y_val, preds)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            mape = mean_absolute_percentage_error(y_val, preds)

            print(f"   * RMSE:   INR {rmse:,.2f} (Target: < INR 2,000)")
            print(f"   * MAE:    INR {mae:,.2f}")
            print(f"   * R2:     {r2:.4f}")
            print(f"   * MAPE:   {mape * 100:.2f}%")

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mape", mape)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
                best_pipeline = pipeline

    print(f"\n==========================================")
    print(f"Best Regression Model: {best_model_name} (RMSE: INR {best_rmse:,.2f})")
    print(f"==========================================")

    export_obj = {
        "pipeline": best_pipeline,
        "feature_cols": FEATURE_COLS,
        "model_name": best_model_name,
        "best_rmse": best_rmse
    }
    save_path = os.path.join(MODELS_DIR, "best_regressor.joblib")
    joblib.dump(export_obj, save_path)
    print(f"Exported production regressor to: {save_path}")

if __name__ == "__main__":
    train_and_evaluate_regressors()
