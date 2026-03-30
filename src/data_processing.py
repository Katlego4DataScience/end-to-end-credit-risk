import pandas as pd
import numpy as np
import os

def load_raw_data(filepath):
    print(f"Loading raw data from {filepath}...")
    return pd.read_csv(filepath)

def clean_loan_portfolio(df):
    print("Cleaning and engineering features for Loan Portfolio...")
    df_clean = df.copy()
    
    # 1. Handle Missing Values
    # Fill numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    
    # 2. Feature Engineering
    # Create a 'time_on_books' feature if origination and default dates were used
    df_clean['origination_date'] = pd.to_datetime(df_clean['origination_date'])
    df_clean['maturity_date'] = pd.to_datetime(df_clean['maturity_date'])
    df_clean['loan_duration_years'] = (df_clean['maturity_date'] - df_clean['origination_date']).dt.days / 365.25
    
    # 3. Categorical Encoding Preparation
    # Ensure standard formatting for strings (lowercase, strip whitespace)
    categorical_cols = ['sector', 'loan_type', 'collateral', 'initial_rating']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
            
    print(f"Data cleaned. Final shape: {df_clean.shape}")
    return df_clean

def main():
    # Define paths
    raw_loan_path = 'data/raw/loan_portfolio.csv'
    processed_dir = 'data/processed/'
    
    # Execute ETL Pipeline
    df_raw = load_raw_data(raw_loan_path)
    df_clean = clean_loan_portfolio(df_raw)
    
    # Save Processed Data
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'clean_loan_portfolio.csv')
    df_clean.to_csv(out_path, index=False)
    
    print(f"💾 Master cleaned dataset saved to {out_path}")

if __name__ == "__main__":
    main()