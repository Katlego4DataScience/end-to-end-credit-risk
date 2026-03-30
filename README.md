# Risk Sentinel: End-to-End Credit Risk ML Framework 🏦

## Overview
**Risk Sentinel** is a comprehensive Credit Risk Management platform designed for financial institutions. It bridges the gap between raw loan data and executive decision-making using Machine Learning (Gradient Boosting/Random Forest), Markov Chain transition modeling, and regulatory-style stress testing.

The framework features a live **Streamlit Dashboard** that allows users to simulate new loans and visualize portfolio-wide impact under economic shocks.

## 🚀 Interactive Dashboard
The project includes a high-performance Streamlit application with the following features:
* **Portfolio Overview**: Real-time KPIs (EAD, Avg Coupon, Default Rates).
* **Live PD Pricing Engine**: An interactive simulator where users can input borrower financials to get a real-time Probability of Default (PD) and Credit Rating from our trained ML model.
* **Stress Testing Visualizer**: Graphical analysis of Expected Loss (EL) jumps under COVID-like and GFC-like scenarios.

## 🛠️ Framework Architecture
The system is built as a modular Python pipeline:
1.  **ETL (`data_processing.py`)**: Automates cleaning and feature engineering.
2.  **PD Model (`pd_model.py`)**: A classification pipeline comparing 4 models to predict default.
3.  **LGD Model (`lgd_ead_model.py`)**: A regression pipeline estimating Loss Given Default.
4.  **Transition Matrix (`transition_matrix.py`)**: Markov Chain migration analysis.
5.  **Stress Testing (`stress_testing.py`)**: Macro-economic scenario aggregation.

## 📂 Data Reference
The data used in this project is sourced from the [Credit Risk Dataset (50k loans, 10 sectors)](https://www.kaggle.com/datasets/sergionefedov/credit-risk-dataset-50k-loans-10-sectors) by **Sergio Nefedov**. It contains a rich set of features including borrower leverage, debt-to-equity ratios, and historical default flags across diverse industries.

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Katlego4DataScience/end-to-end-credit-risk.git](https://github.com/Katlego4DataScience/end-to-end-credit-risk.git)
   cd end-to-end-credit-risk