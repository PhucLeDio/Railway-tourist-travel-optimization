# import pandas as pd
# import random
# import numpy as np

# def add_service_costs_to_taxi_matrix():
#     """
#     Read the existing taxi distance matrix and add random service costs
#     between 100-1000 baht to make the costs more specialized.
#     """
    
#     # Read the existing CSV file
#     print("Reading existing taxi distance matrix...")
#     df = pd.read_csv('formatted_distance_matrix_taxi_with_cost.csv')
    
#     # Set random seed for reproducibility (optional)
#     random.seed(42)
#     np.random.seed(42)
    
#     # Generate random service costs between 100-1000 baht
#     print("Generating random service costs...")
#     service_costs = np.random.uniform(100, 1000, len(df))
    
#     # Add service costs to existing cost_baht values
#     print("Adding service costs to existing costs...")
#     df['cost_baht'] = df['cost_baht'] + service_costs
    
#     # Round to 2 decimal places for currency
#     df['cost_baht'] = df['cost_baht'].round(2)
    
#     # Create new filename
#     output_filename = 'data/formatted_distance_matrix_taxi_with_service_cost.csv'
    
#     # Save to new CSV file
#     print(f"Saving to {output_filename}...")
#     df.to_csv(output_filename, index=False)
    
#     # Display some statistics
#     print("\n=== Cost Statistics ===")
#     print(f"Original cost range: {df['cost_baht'].min():.2f} - {df['cost_baht'].max():.2f} baht")
#     print(f"Average cost: {df['cost_baht'].mean():.2f} baht")
#     print(f"Total records processed: {len(df)}")
    
#     # Show first few rows as example
#     print("\n=== First 5 rows of new file ===")
#     print(df.head())
    
#     print(f"\n✅ Successfully created {output_filename}")
#     print("The new file includes original costs plus random service costs (100-1000 baht)")

# if __name__ == "__main__":
#     add_service_costs_to_taxi_matrix()

import pandas as pd

old = pd.read_csv('formatted_distance_matrix_taxi_with_cost.csv')
new = pd.read_csv('data/formatted_distance_matrix_taxi_with_service_cost.csv')

print(old.dtypes)
print(new.dtypes)