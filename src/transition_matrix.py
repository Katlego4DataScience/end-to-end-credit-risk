import pandas as pd
import numpy as np
import os

def load_data(filepath):
    print(f"Loading credit ratings data from {filepath}...")
    return pd.read_csv(filepath)

def build_transition_matrix(df):
    print("Calculating Markov Chain transition probabilities...")
    
    # Standard credit rating scale
    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
    
    # If a company defaulted, force their "to_rating" to be "D" for the matrix
    df.loc[df['defaulted'] == 1, 'to_rating'] = 'D'
    
    # Calculate the empirical transition matrix using a crosstab
    # normalize='index' turns counts into row-wise percentages (probabilities)
    tm = pd.crosstab(df['from_rating'], df['to_rating'], normalize='index')
    
    # Reindex to ensure our standard order is enforced for both rows and columns
    # Fill any missing transitions with 0.0
    tm = tm.reindex(index=rating_order, columns=rating_order, fill_value=0.0)
    
    # A defaulted company stays in default (absorbing state in a Markov Chain)
    tm.loc['D'] = 0.0
    tm.loc['D', 'D'] = 1.0
    
    return tm

def main():
    # Define paths
    data_path = 'data/raw/credit_ratings.csv'
    output_dir = 'data/processed/'
    
    # Execute
    df = load_data(data_path)
    transition_matrix = build_transition_matrix(df)
    
    # Display the Matrix
    print("\n--- 1-Year Corporate Ratings Transition Matrix (%) ---")
    # Multiply by 100 and round for a clean display
    display_tm = (transition_matrix * 100).round(2)
    print(display_tm)
    
    # Save the output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'transition_matrix.csv')
    transition_matrix.to_csv(out_path)
    
    print(f"\n💾 Saved Transition Matrix successfully to {out_path}")

if __name__ == "__main__":
    main()