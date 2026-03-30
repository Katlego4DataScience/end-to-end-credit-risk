import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def load_data(filepath):
    print(f"Loading loan portfolio data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    print("Preprocessing data for PD modeling...")
    
    # Select features relevant for predicting default before it happens
    features = ['sector', 'loan_type', 'collateral', 'initial_rating', 
                'credit_score', 'coupon_rate', 'leverage', 'interest_coverage', 
                'debt_to_equity', 'maturity_months']
    
    X = df[features].copy()
    y = df['defaulted'].copy()
    
    # Handle Missing Values
    X.fillna(X.median(numeric_only=True), inplace=True)
    
    # One-Hot Encode Categorical Variables
    categorical_cols = ['sector', 'loan_type', 'collateral', 'initial_rating']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y

def train_and_compare_models(X, y):
    print("Splitting data and applying standard scaling...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features (Critical for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define our 4 models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    best_model = None
    best_auc = 0
    best_name = ""
    results = {}
    
    print("\n--- Training and Comparing Models ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_prob)
        results[name] = auc
        
        print(f"{name} trained. ROC-AUC: {auc:.4f}")
        
        # Track the best performing model
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    # Print Leaderboard
    print("\n--- Final Model Leaderboard (ROC-AUC) ---")
    for name, auc in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {auc:.4f}")
        
    print(f"\n🏆 Winner: {best_name} wins with a score of {best_auc:.4f}!")
    
    # Print detailed report for the winner
    y_pred_best = best_model.predict(X_test_scaled)
    print(f"\nClassification Report for {best_name}:")
    print(classification_report(y_test, y_pred_best))
    
    return best_model, best_name, scaler, X_train.columns

def main():
    # Define paths (Relative to the root directory)
    data_path = 'data/processed/clean_loan_portfolio.csv'
    model_dir = 'models/'
    
    # Execute pipeline
    df = load_data(data_path)
    X, y = preprocess_data(df)
    best_model, best_name, scaler, feature_names = train_and_compare_models(X, y)
    
    # Save Winning Model and Assets
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, 'best_pd_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'pd_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'pd_feature_names.pkl'))
    
    print(f"\n💾 Saved {best_name} and scaling assets successfully to {model_dir}")

if __name__ == "__main__":
    main()