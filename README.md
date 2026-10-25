# EMIPredict AI - Intelligent Financial Risk Assessment Platform

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-97.79%25_Accuracy-brightgreen.svg)
![RMSE](https://img.shields.io/badge/RMSE-INR_720.45-success.svg)

**EMIPredict AI** is an end-to-end FinTech & Banking risk-assessment platform that automates loan underwriting and predicts customer EMI eligibility and safe repayment ceilings in real time using Machine Learning and MLflow.

---

## 🚀 Key Features

* **Dual Machine Learning Problem Solving**:
  * **Classification (EMI Eligibility)**: Categorizes applicants into 3 risk tiers: `Eligible` (Low Risk), `High_Risk` (Marginal / High Interest), and `Not_Eligible` (High Default Risk).
  * **Regression (Max Monthly EMI Amount)**: Predicts safe monthly installments (`max_monthly_emi`) between ₹500 and ₹50,000 based on disposable earnings.
* **400,000 Financial Records**: Evaluated across 5 lending scenarios:
  1. *E-commerce Shopping EMI* (₹10K - ₹200K | 3–24 mos)
  2. *Home Appliances EMI* (₹20K - ₹300K | 6–36 mos)
  3. *Vehicle EMI* (₹80K - ₹1.5M | 12–84 mos)
  4. *Personal Loan EMI* (₹50K - ₹1.0M | 12–60 mos)
  5. *Education EMI* (₹50K - ₹500K | 6–48 mos)
* **Domain Feature Engineering**: Incorporates derived ratios such as Debt-to-Income (DTI), Expense-to-Income (ETI), Emergency Cushion, and EMI Burden.
* **MLflow Experiment Tracking**: Tracks runs, hyperparameters, metrics (Accuracy, F1, ROC-AUC, RMSE, MAE, $R^2$), and model artifacts.
* **Multi-Page Streamlit App**:
  * Executive Overview Dashboard
  * Interactive Exploratory Data Analysis (Plotly)
  * Real-Time Eligibility Evaluator
  * Maximum Monthly EMI Calculator
  * MLflow Model Comparison Leaderboard
  * SQLite CRUD Interface for Loan Officers

---

## 📊 Model Evaluation Results

### 1. Classification Models (`emi_eligibility`)

| Model | Accuracy (Target > 90%) | Macro F1 | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 90.33% | 0.5965 | 0.9451 | Evaluated |
| **Random Forest Classifier** | 94.35% | 0.6455 | 0.9843 | Evaluated |
| **XGBoost Classifier** | **97.79%** | **0.9081** | **0.9951** | 🏆 **Production Best** |

### 2. Regression Models (`max_monthly_emi`)

| Model | RMSE (Target < ₹2,000) | MAE | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Baseline)** | ₹3,854.21 | ₹2,541.09 | 0.7506 | Evaluated |
| **Random Forest Regressor** | ₹963.41 | ₹326.72 | 0.9844 | Evaluated |
| **XGBoost Regressor** | **₹720.45** | **₹199.76** | **0.9913** | 🏆 **Production Best** |

---

## 🛠️ Project Structure

```text
EMIpredict AI/
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                 # Downloaded 400K raw records
│   ├── processed/           # Cleaned train/val/test splits (.parquet)
│   └── applicants.db        # SQLite CRUD persistent database
├── models/
│   ├── best_classifier.joblib  # Trained XGBoost Classifier
│   └── best_regressor.joblib   # Trained XGBoost Regressor
├── src/
│   ├── data_loader.py          # Data ingestion and stratified splits
│   ├── feature_engineering.py  # Domain ratio transformations
│   ├── train_classification.py # MLflow-tracked classification
│   ├── train_regression.py     # MLflow-tracked regression
│   └── database.py             # SQLAlchemy applicant CRUD models
├── app/
│   ├── app.py                  # Streamlit entry point
│   └── pages/
│       ├── 1_📊_EDA_Dashboard.py
│       ├── 2_🎯_EMI_Eligibility_Prediction.py
│       ├── 3_📈_Max_EMI_Calculator.py
│       ├── 4_🤖_MLflow_Model_Registry.py
│       └── 5_🗄️_Data_Management_CRUD.py
└── mlruns/                     # MLflow tracking experiments
```

---

## 🚀 How to Run Locally

### 1. Set Up Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Training
```powershell
# Preprocess and split dataset
python src/data_loader.py

# Train models with MLflow
python src/train_classification.py
python src/train_regression.py
```

### 3. Launch Streamlit Application
```powershell
streamlit run app/app.py
```

### 4. View MLflow Dashboard
```powershell
mlflow ui
```
Navigate to `http://localhost:5000` to inspect experiment metrics.

---

## 🌐 Cloud Deployment (Streamlit Cloud)

1. Push this repository to GitHub:
   ```powershell
   git init
   git add .
   git commit -m "Initial release of EMIPredict AI"
   git branch -M main
   git remote add origin <YOUR_GITHUB_REPO_URL>
   git push -u origin main
   ```
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository and specify `app/app.py` as the main app entry.
4. Click **Deploy**!
