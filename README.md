# End-to-End Credit Risk Modeling Framework

## Overview
This repository contains a comprehensive, end-to-end Credit Risk Modeling suite. It spans granular loan-level Expected Credit Loss (ECL) calculations (PD, LGD, EAD), cohort-based vintage analysis, corporate ratings migration (Markov chains), and macroeconomic stress testing.

## Datasets Used
* **Loan Portfolio (`loan_portfolio.csv`)**: 50k loan facilities with borrower financials, credit scores, and default flags used for core ECL modeling.
* **Portfolio Metrics (`portfolio_metrics.csv`)**: Time-series aggregate risk metrics correlated with macroeconomic indicators.
* **Vintage Analysis (`vintage_analysis.csv`)**: Cohort tracking of default rates over time.
* **Credit Ratings (`credit_ratings.csv`)**: Historical ratings transitions for 18k corporate issuers used to generate Transition Matrices.
* **Macro Stress Scenarios (`macro_stress_scenarios.csv`)**: Stressed PD/LGD outputs based on baseline, GFC-like, and COVID-like economic shocks.

## Model Architecture
The framework is divided into four primary modules:
1. **Expected Credit Loss (ECL) Engine**:
   - **PD Model**: Predicts the Probability of Default using borrower financials and credit scores (Logistic Regression / Random Forest).
   - **LGD Model**: Estimates Loss Given Default based on collateral types and recovery rates (Fractional Logit / Beta Regression).
   - **EAD Model**: Calculates Exposure at Default using amortization schedules and credit conversion factors (CCF).
2. **Transition Matrix Generator**: Calculates annual migration probabilities of corporate credit ratings using a discrete-time Markov Chain approach.
3. **Vintage & Cohort Analytics**: Tracks and visualizes cumulative and marginal default rates across origination quarters.
4. **Macro Stress Testing**: Overlays economic shocks (GDP, Unemployment, Interest Rates) onto baseline risk parameters to calculate stressed expected losses.

## Installation
1. Clone the repository: `git clone https://github.com/yourusername/end-to-end-credit-risk.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Place the raw datasets into the `data/raw/` directory.

## Usage
*Run the ECL pipeline:*
`python src/pd_model.py`