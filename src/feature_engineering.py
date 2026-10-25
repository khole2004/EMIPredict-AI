import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL_COLS = [
    'gender',
    'marital_status',
    'education',
    'employment_type',
    'company_type',
    'house_type',
    'existing_loans',
    'emi_scenario'
]

NUMERICAL_COLS = [
    'age',
    'monthly_salary',
    'years_of_employment',
    'monthly_rent',
    'family_size',
    'dependents',
    'school_fees',
    'college_fees',
    'travel_expenses',
    'groceries_utilities',
    'other_monthly_expenses',
    'current_emi_amount',
    'credit_score',
    'bank_balance',
    'emergency_fund',
    'requested_amount',
    'requested_tenure'
]

ENGINEERED_COLS = [
    'total_monthly_expenses',
    'total_obligations',
    'disposable_income',
    'dti_ratio',
    'expense_to_income_ratio',
    'emergency_cushion',
    'bank_balance_to_income',
    'estimated_requested_emi',
    'emi_burden_ratio'
]

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds financial risk & affordability ratios to the input dataframe."""
    df = df.copy()

    # Total expenses
    df['total_monthly_expenses'] = (
        df.get('monthly_rent', 0) +
        df.get('school_fees', 0) +
        df.get('college_fees', 0) +
        df.get('travel_expenses', 0) +
        df.get('groceries_utilities', 0) +
        df.get('other_monthly_expenses', 0)
    )

    # Total financial commitments
    df['total_obligations'] = df['total_monthly_expenses'] + df.get('current_emi_amount', 0)

    # Disposable income
    df['disposable_income'] = df['monthly_salary'] - df['total_obligations']

    # Financial ratios
    df['dti_ratio'] = df['current_emi_amount'] / (df['monthly_salary'] + 1.0)
    df['expense_to_income_ratio'] = df['total_monthly_expenses'] / (df['monthly_salary'] + 1.0)
    df['emergency_cushion'] = df['emergency_fund'] / (df['total_obligations'] + 1.0)
    df['bank_balance_to_income'] = df['bank_balance'] / (df['monthly_salary'] + 1.0)

    # Loan application specific
    df['estimated_requested_emi'] = df['requested_amount'] / (df['requested_tenure'].replace(0, 1))
    
    # EMI burden relative to disposable income
    positive_disposable = df['disposable_income'].apply(lambda x: max(x, 100.0))
    df['emi_burden_ratio'] = df['estimated_requested_emi'] / positive_disposable

    return df

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_derived_features(X)
