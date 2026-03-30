import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data(filepath):
    print(f"Loading loan portfolio data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_lgd_data(df):
    print("Filtering for defaults and preprocessing data for LGD modeling...")
    
    # We only train LGD on loans that actually defaulted
    df_defaults = df[df['defaulted'] == 1].copy()
    
    # Select features relevant for predicting recovery/loss
    features = ['sector', 'loan_type', 'collateral', 
                'leverage', 'debt_to_equity', 'ead']
    
    X = df_defaults[features].copy()
    
    # Target variable: Loss Given Default (LGD)
    y = df_defaults['loss_given_default'].copy()
    
    # Handle Missing Values (if any slipped through)
    X.fillna(X.median(numeric_only=True), inplace=True)
    y.fillna(y.median(), inplace=True)
    
    # One-Hot Encode Categorical Variables
    categorical_cols = ['sector', 'loan_type', 'collateral']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y

def train_and_compare_regressors(X, y):
    print("Splitting data and applying standard scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define our 4 Regression models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    results = {}
    
    print("\n--- Training and Comparing LGD Models ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'RMSE': rmse, 'R2': r2}
        
        print(f"{name} trained. RMSE: {rmse:.4f} | R-Squared: {r2:.4f}")
        
        # Track the best performing model (Highest R-squared)
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    # Print Leaderboard
    print("\n--- Final LGD Model Leaderboard (R-Squared) ---")
    # Sort by R2 descending
    for name, metrics in sorted(results.items(), key=lambda item: item[1]['R2'], reverse=True):
        print(f"{name}: R2 = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")
        
    print(f"\n🏆 Winner: {best_name} wins with an R-Squared of {best_r2:.4f}!")
    
    return best_model, best_name, scaler, X_train.columns

def main():
    # Define paths
    data_path = 'data/raw/loan_portfolio.csv'
    model_dir = 'models/'
    
    # Execute pipeline
    df = load_data(data_path)
    X, y = preprocess_lgd_data(df)
    
    if len(X) == 0:
        print("Error: No defaulted loans found in the dataset to train LGD.")
        return
        
    best_model, best_name, scaler, feature_names = train_and_compare_regressors(X, y)
    
    # Save Winning Model and Assets
    joblib.dump(best_model, os.path.join(model_dir, 'best_lgd_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'lgd_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'lgd_feature_names.pkl'))
    
    print(f"\n💾 Saved {best_name} (LGD) and scaling assets successfully to {model_dir}")

if __name__ == "__main__":
    main()