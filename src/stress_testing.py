import pandas as pd
import os

def load_data(filepath):
    print(f"Loading macro stress scenarios from {filepath}...")
    return pd.read_csv(filepath)

def analyze_stress_scenarios(df):
    print("Aggregating Expected Loss (EL) under different macroeconomic scenarios...")
    
    # Group by scenario to calculate the total portfolio impact
    scenario_summary = df.groupby('scenario').agg(
        Total_EAD=('total_ead', 'sum'),
        Base_EL=('expected_loss_base', 'sum'),
        Stressed_EL=('expected_loss_stress', 'sum')
    ).reset_index()
    
    # Calculate the absolute dollar increase and the percentage increase
    scenario_summary['EL_Increase_Amount'] = scenario_summary['Stressed_EL'] - scenario_summary['Base_EL']
    scenario_summary['EL_Increase_Pct'] = (scenario_summary['EL_Increase_Amount'] / scenario_summary['Base_EL']) * 100
    
    return scenario_summary

def main():
    # Define paths
    data_path = 'data/raw/macro_stress_scenarios.csv'
    output_dir = 'data/processed/'
    
    # Execute
    df = load_data(data_path)
    summary = analyze_stress_scenarios(df)
    
    print("\n--- Macroeconomic Stress Testing Portfolio Impact ---")
    
    # Format numbers nicely for the console (adds commas for thousands)
    pd.options.display.float_format = '{:,.2f}'.format
    print(summary.to_string(index=False))
    
    # Save the output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'stress_test_summary.csv')
    summary.to_csv(out_path, index=False)
    
    print(f"\n💾 Saved Stress Test Summary successfully to {out_path}")

if __name__ == "__main__":
    main()