import os
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engineering import add_derived_features

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "emi_prediction_dataset.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

def load_and_preprocess_data(filepath: str = RAW_DATA_PATH):
    """
    Loads raw dataset, cleans inconsistencies, applies feature engineering,
    and performs a stratified 70/15/15 train-val-test split.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at {filepath}")
    
    print(f"[1/4] Reading dataset from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"      Initial record count: {len(df):,} rows, {len(df.columns)} columns")

    # 1. Deduplication
    initial_count = len(df)
    df = df.drop_duplicates()
    dedup_removed = initial_count - len(df)
    if dedup_removed > 0:
        print(f"[2/4] Removed {dedup_removed:,} duplicate records.")
    else:
        print("[2/4] No duplicate records found.")

    # 2. Type cleansing & null handling
    numeric_columns = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
        'family_size', 'dependents', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
        'requested_amount', 'requested_tenure', 'max_monthly_emi'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    null_before = df.isnull().sum().sum()
    if null_before > 0:
        print(f"      Dropping {null_before:,} invalid/null values across columns...")
        df = df.dropna()

    # 3. Add Feature Engineering
    print("[3/4] Engineering financial risk ratios & domain features...")
    df = add_derived_features(df)
    print(f"      Total features after engineering: {len(df.columns)}")

    # 4. Stratified Train / Val / Test Split (70% / 15% / 15%)
    print("[4/4] Creating stratified Train (70%), Val (15%), Test (15%) splits...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df['emi_eligibility']
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=(0.15 / 0.85),
        random_state=42,
        stratify=train_val_df['emi_eligibility']
    )

    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("\nPre-processing Complete! Summary:")
    print(f"   * Train set:      {len(train_df):,} records ({train_path})")
    print(f"   * Validation set: {len(val_df):,} records ({val_path})")
    print(f"   * Test set:       {len(test_df):,} records ({test_path})")

    return train_df, val_df, test_df

if __name__ == "__main__":
    load_and_preprocess_data()
