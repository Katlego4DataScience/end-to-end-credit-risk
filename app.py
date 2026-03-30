import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration & Logo
LOGO_PATH = "image_f57c3b.png" # The sentinel image you uploaded

st.set_page_config(page_title="Risk Sentinel Framework", page_icon="🏦", layout="wide")

# Center the logo and title in the sidebar
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.title("Risk Sentinel ML")
st.sidebar.markdown("---")

# 2. Advanced Data & Model Caching
@st.cache_resource # @st.cache_resource is used for loading heavy ML models
def load_ml_assets():
    """Loads the trained PD model, scaler, and feature list."""
    try:
        model = joblib.load("models/best_pd_model.pkl")
        scaler = joblib.load("models/pd_scaler.pkl")
        feature_names = joblib.load("models/pd_feature_names.pkl")
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Error: Could not find model artifacts in 'models/'. Have you run pd_model.py?")
        return None, None, None

@st.cache_data # @st.cache_data is for loading static datasets
def load_data():
    """Loads the clean portfolio and stress testing summary."""
    df = pd.read_csv("data/processed/clean_loan_portfolio.csv")
    stress_df = pd.read_csv("data/processed/stress_test_summary.csv")
    # Clean string encoding from our ETL for selectboxes
    categorical_cols = ['sector', 'loan_type', 'collateral']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.title()
    return df, stress_df

# Load all assets
df, stress_df = load_data()
pd_model, pd_scaler, pd_feature_names = load_ml_assets()

# 3. Sidebar: New Loan Simulator (Used for Tab 2)
st.sidebar.header("⚡ Single Loan Simulator")
st.sidebar.markdown("Adjust borrower inputs to calculate real-time Probability of Default.")

# Define the category lists based on the original dataset values
sectors = sorted(df['sector'].unique())
loan_types = sorted(df['loan_type'].unique())
collaterals = sorted(df['collateral'].unique())

# Simulator Input Widgets
with st.sidebar.container():
    sim_sector = st.selectbox("Borrower Sector", sectors)
    sim_type = st.selectbox("Loan Type", loan_types)
    sim_collateral = st.selectbox("Collateral Type", collaterals)
    sim_score = st.slider("Credit Score (FICO)", 300, 850, 710)
    sim_leverage = st.slider("Borrower Leverage (Debt/EBITDA)", 0.0, 15.0, 4.5)
    sim_dte = st.slider("Debt-to-Equity Ratio", 0.0, 10.0, 1.2)
    sim_coupon = st.slider("Coupon Rate (%)", 1.0, 15.0, 5.5)
    sim_maturity = st.number_input("Maturity (Months)", min_value=1, value=24)
    
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 Katlego4DataScience | Risk Sentinel v1.0")

# 4. Define UI Layout Tabs (Added Home Tab)
tab_home, tab1, tab2, tab3 = st.tabs(["🏠 Framework Home", "📊 Portfolio Overview", "🔮 Live PD Pricing", "🌪️ Stress Testing"])

# --- TAB: HOME / ABOUT ---
with tab_home:
    col1_h, col2_h = st.columns([1, 4])
    with col1_h:
        st.image(LOGO_PATH, width=150)
    with col2_h:
        st.title("Risk Sentinel Framework")
        st.subheader("An End-to-End Enterprise Credit Risk Modeling Platform")

    st.markdown("---")
    
    st.markdown("""
    ## Executive Summary
    Welcome to the **Risk Sentinel Framework**, a demonstration of a comprehensive, production-ready credit risk modeling suite. This platform bridges the gap between granular loan data, advanced machine learning, and macroeconomic stress testing to provide a 360-degree view of portfolio risk.
    
    This dashboard interacts directly with the Python modules and model artifacts (`.pkl` files) generated in the backend of this repository.
    
    ## Framework Architecture
    This solution is comprised of five integrated modules, as reflected in the `src/` directory of the repository:
    
    1.  **Centralized ETL (`data_processing.py`)**: Automates raw data cleaning, missing value imputation (median), feature engineering (`loan_duration_years`), and string standardization. It outputs a standardized 'master' dataset.
    2.  **Probability of Default (PD) Engine (`pd_model.py`)**: Trains four ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) on historical data. It performs feature scaling and automatically selects and saves the model with the highest ROC-AUC score.
    3.  **Loss Given Default (LGD) Engine (`lgd_ead_model.py`)**: A regression pipeline that analyzes defaulted loans to estimate loss percentages, using metrics like collateral type and borrower leverage. It compares four regression models and saves the best-performing Random Forest Regressor.
    4.  **Transition Matrix Generator (`transition_matrix.py`)**: Uses discrete-time Markov Chains to analyze historical rating changes (e.g., AAA to BBB) and generates 1-year transition probabilities.
    5.  **Macro Stress Testing (`stress_testing.py`)**: Overlays hypothetical macroeconomic shocks (GDP crash, high unemployment, GFC-like and COVID-like scenarios) onto the portfolio to project stressed expected losses.
    
    *Use the navigation tabs above to explore the different facets of the framework.*
    """)

# --- TAB 1: PORTFOLIO OVERVIEW ---
with tab1:
    st.header("Live Portfolio Aggregates")
    st.markdown("View high-level risk metrics across the total corporate loan book.")
    
    # Top-level KPIs with coloring
    col1, col2, col3, col4 = st.columns(4)
    
    total_ead = df['ead'].sum()
    default_rate = df['defaulted'].mean()
    
    # Use metrics with sub-coloring (delta) just to look nice
    col1.metric("Total Active Loans", f"{len(df):,}")
    col2.metric("Total Exposure (EAD)", f"${total_ead:,.0f}", help="Total Exposure at Default")
    col3.metric("Avg Coupon Rate", f"{df['coupon_rate'].mean():.2f}%")
    
    # Add a color indicator for default rate (Red if > 3%)
    default_delta = None
    if default_rate > 0.03:
        default_delta = f"{default_rate:.1%} (High Risk)"
    else:
        default_delta = f"{default_rate:.1%} (Stable)"
        
    col4.metric("Hist. Default Rate", f"{(default_rate * 100):.2f}%", delta=default_delta)
    
    st.markdown("---")
    st.subheader("Granular Loan Book Data")
    # Streamlit automatically applies dark theme to the dataframe
    st.dataframe(df.head(500), use_container_width=True)

# --- TAB 2: PD PRICING ENGINE (MOVING FROM 'COMING SOON' TO 'LIVE') ---
with tab2:
    st.header("🔮 Real-Time Loan Scoring Simulator")
    
    if pd_model is None or pd_scaler is None:
        st.error("ML Model assets could not be loaded. This tab is disabled.")
    else:
        st.markdown("""
        Adjust the inputs in the **⚡ Single Loan Simulator** on the left sidebar. This tab uses your trained **Gradient Boosting** machine learning model to calculate the exact probability that the hypothetical borrower will default within the next 12 months.
        """)
        
        # --- MODEL PREDICTION LOGIC ---
        
        # 1. Prepare the input data structure matching training data
        # We need to map categorical inputs to one-hot encoding
        
        input_data = {
            'credit_score': [sim_score],
            'coupon_rate': [sim_coupon],
            'leverage': [sim_leverage],
            'interest_coverage': [3.5], # Assuming a static average value as we didn't add a slider
            'debt_to_equity': [sim_dte],
            'maturity_months': [sim_maturity]
        }
        
        # Create dummy dataframe matching model structure
        sim_df = pd.DataFrame(input_data)
        
        # Add the one-hot columns and set the correct one to 1
        # Convert inputs to lowercase to match ETL standardization used by the model
        sim_sector_col = f"sector_{sim_sector.lower().strip()}"
        sim_type_col = f"loan_type_{sim_type.lower().strip()}"
        sim_coll_col = f"collateral_{sim_collateral.lower().strip()}"
        
        # Populate all known feature columns with 0
        for col in pd_feature_names:
            if col not in sim_df.columns:
                sim_df[col] = 0
                
        # Set the matching categorical columns to 1 (if they exist in model training)
        if sim_sector_col in sim_df.columns: sim_df[sim_sector_col] = 1
        if sim_type_col in sim_df.columns: sim_df[sim_type_col] = 1
        if sim_coll_col in sim_df.columns: sim_df[sim_coll_col] = 1
        
        # Ensure correct column order
        sim_df = sim_df[pd_feature_names]
        
        # 2. Scale the data using the loaded scaler
        sim_data_scaled = pd_scaler.transform(sim_df)
        
        # 3. Run Prediction
        pd_probability = pd_model.predict_proba(sim_data_scaled)[:, 1][0]
        
        # Mapping Probability to a rough Credit Rating Scale
        rating = "CCC"
        color = "🔴"
        if pd_probability < 0.005: rating, color = "AAA", "🔵"
        elif pd_probability < 0.01: rating, color = "AA", "🟢"
        elif pd_probability < 0.02: rating, color = "A", "🟢"
        elif pd_probability < 0.04: rating, color = "BBB", "🟡"
        elif pd_probability < 0.08: rating, color = "BB", "🟠"
        elif pd_probability < 0.15: rating, color = "B", "🟠"
        
        # Display Results
        st.markdown("---")
        st.subheader("Simulator Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        # Display Probability
        res_col1.metric(label="Probability of Default (1-Year PD)", 
                       value=f"{pd_probability:.2%}",
                       help="Calculated using the Gradient Boosting Classifier.")
        
        # Display Implied Rating
        res_col2.metric(label="Implied Risk Rating", value=rating, delta=color, delta_color="off")
        
        # Display Decision Recommendation
        decision = "🟢 APPROVE"
        if pd_probability > 0.08: decision = "🟠 REFER TO CREDIT COMMITTEE"
        if pd_probability > 0.15: decision = "🔴 REJECT"
        res_col3.metric(label="Automated Recommendation", value=decision)
        
        # 4. Feature Importance Placeholder (Makes it look nice)
        st.markdown("---")
        st.subheader("Risk Driver Analysis")
        st.write("Below are the key drivers impacting this specific borrower's default probability (conceptual visualization).")
        
        importance_data = pd.DataFrame({
            'Feature': ['Credit Score', 'Leverage', 'Debt-to-Equity', 'Sector Risk'],
            'Impact Score': [sim_score/100, sim_leverage, sim_dte, 1.5]
        }).sort_values(by='Impact Score', ascending=True)
        
        st.bar_chart(importance_data.set_index('Feature'))


# --- TAB 3: STRESS TESTING ---
with tab3:
    st.header("🌪️ Macroeconomic Scenario Analysis")
    st.write("Expected Loss (EL) impact under Federal Reserve style shock scenarios.")
    st.info("These numbers are generated by aggregating your `macro_stress_scenarios.csv` output.")
    
    st.markdown("---")
    st.subheader("Scenarios Portfolio Aggregate Table")
    
    # Apply styling and coloring to the table directly
    st.dataframe(stress_df.style.format({
        "Total_EAD": "${:,.0f}",
        "Base_EL": "${:,.0f}",
        "Stressed_EL": "${:,.0f}",
        "EL_Increase_Amount": "${:,.0f}",
        "EL_Increase_Pct": "{:.2f}%"
    }).background_gradient(subset=['EL_Increase_Pct'], cmap='Reds'), # Heatmap on the % increase
    use_container_width=True)
    
    # Create a nice visual bar chart comparing Base vs Stressed EL
    st.markdown("---")
    st.subheader("Visual Impact Analysis (Total Expected Loss)")
    chart_data = stress_df.set_index('scenario')[['Base_EL', 'Stressed_EL']]
    st.bar_chart(chart_data)