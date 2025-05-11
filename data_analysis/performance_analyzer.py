# performance_analyzer.py

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import argparse
import os
import scipy.stats as stats # Added for Q-Q plot and Pearson R
import numpy as np # Added for np.inf and np.nan

def analyze_predictions(csv_file_path):
    """
    Analyzes the performance of predictions against target values from a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file containing 'pred' and 'target' columns.
    """
    try:
        # Load the data
        data = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_file_path} was not found.")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    if 'pred' not in data.columns or 'target' not in data.columns:
        print("Error: CSV file must contain 'pred' and 'target' columns.")
        return

    # --- Data Cleaning: Handle NaNs, non-numeric data, and infinities in target and pred columns ---
    initial_row_count = len(data)

    # Attempt to convert 'target' and 'pred' to numeric, coercing errors to NaN
    # This handles cases where columns might be object type due to non-numeric entries
    if 'target' in data.columns:
        data['target'] = pd.to_numeric(data['target'], errors='coerce')
    else:
        print("Critical Error: 'target' column somehow disappeared after initial check. Aborting.")
        return
        
    if 'pred' in data.columns:
        data['pred'] = pd.to_numeric(data['pred'], errors='coerce')
    else:
        print("Critical Error: 'pred' column somehow disappeared after initial check. Aborting.")
        return

    # Now check for NaNs (which include original NaNs and coerced non-numeric values)
    nan_in_target = data['target'].isnull().sum()
    nan_in_pred = data['pred'].isnull().sum()

    # Check for infinities
    inf_in_target = np.isinf(data['target']).sum()
    inf_in_pred = np.isinf(data['pred']).sum()

    if nan_in_target > 0:
        print(f"Warning: Found or coerced {nan_in_target} NaN values in 'target' column.")
    if nan_in_pred > 0:
        print(f"Warning: Found or coerced {nan_in_pred} NaN values in 'pred' column.")
    
    if inf_in_target > 0:
        print(f"Warning: Found {inf_in_target} Inf/-Inf values in 'target' column. These will be treated as NaN for removal.")
    if inf_in_pred > 0:
        print(f"Warning: Found {inf_in_pred} Inf/-Inf values in 'pred' column. These will be treated as NaN for removal.")

    # Replace infinities with NaN so dropna can handle them uniformly along with other NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows where either 'target' or 'pred' is NaN 
    # (now includes original NaNs, coerced errors, and infinities)
    data.dropna(subset=['target', 'pred'], inplace=True)
    cleaned_row_count = len(data)

    if cleaned_row_count < initial_row_count:
        print(f"Dropped {initial_row_count - cleaned_row_count} rows due to non-convertible, NaN, or Inf values in 'target' or 'pred' columns.")
        if cleaned_row_count == 0:
            print("Error: No valid (numeric, finite) data points remaining in 'target'/'pred' after cleaning. Cannot proceed with analysis.")
            return
    # --- End Data Cleaning ---

    predictions = data['pred']
    targets = data['target']

    # Calculate metrics
    mse = mean_squared_error(targets, predictions)
    rmse = mse**0.5
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson_r, p_value_pearson = stats.pearsonr(targets, predictions)

    print("\n--- Performance Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (Coefficient of Determination): {r2:.4f}")
    print(f"Pearson Correlation Coefficient (R): {pearson_r:.4f} (p-value: {p_value_pearson:.4g})")
    print("---------------------------\n")

    # --- Identify and report best-performing individual data points ---
    print("\n--- Top 20 Best-Performing Data Points (Smallest Absolute Error) ---")
    # Create a DataFrame that includes the original data plus the absolute error
    data_with_error = data.copy() # data is the DataFrame loaded from the CSV
    data_with_error['absolute_error'] = (data_with_error['target'] - data_with_error['pred']).abs()
    
    # Get the top 20 rows with the smallest absolute error
    # nsmallest will handle cases where there are fewer than 20 rows
    best_performing = data_with_error.nsmallest(20, 'absolute_error')

    if best_performing.empty:
        print("No data points to report (input might be empty or contain only NaN values for pred/target).")
    else:
        print("Showing data points with the smallest absolute difference between target and prediction:")
        
        identifier_cols_non_float = ['gene_name', 'gene_id', 'drug_id'] # Typically strings
        
        for original_index, row_data in best_performing.iterrows():
            print(f"\nData Point (Original CSV Index: {original_index})")
            
            # Print non-float identifier columns first if they exist
            for col_name in identifier_cols_non_float:
                if col_name in row_data:
                    print(f"  {col_name}: {row_data[col_name]}")
            
            # Explicitly handle drug_dose with float formatting if it exists
            if 'drug_dose' in row_data:
                value = row_data['drug_dose']
                try: # Attempt to convert to float for consistent formatting
                    print(f"  drug_dose: {float(value):.4f}")
                except (ValueError, TypeError): # If conversion fails, print as is
                    print(f"  drug_dose: {value}")

            # Print core performance fields: target, pred, and absolute_error

            if 'target' in row_data: # Should always be true if initial checks passed
                print(f"  target: {row_data['target']:.4f}")
            if 'pred' in row_data: # Should always be true if initial checks passed
                print(f"  pred: {row_data['pred']:.4f}")
            if 'absolute_error' in row_data: # Calculated column
                print(f"  absolute_error: {row_data['absolute_error']:.4f}")

            # Print any remaining columns not yet covered

            printed_cols = identifier_cols_non_float + ['drug_dose', 'target', 'pred', 'absolute_error']
            additional_custom_cols = []
            for col_name, value in row_data.items():
                if col_name not in printed_cols:
                    if isinstance(value, float):
                        additional_custom_cols.append(f"  {col_name}: {value:.4f}")
                    else:
                        additional_custom_cols.append(f"  {col_name}: {value}")
            
            if additional_custom_cols:

                for item_str in additional_custom_cols:
                    print(item_str)
    print("-------------------------------------------------------------------\n")

    output_dir = os.path.dirname(csv_file_path)
    residuals = targets - predictions

    # Plot 1: Scatter plot of Predictions vs. Targets
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5, label='Predictions vs. Targets')
    # Add a line for perfect predictions
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Target Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs. Target Values")
    plt.legend()
    plt.grid(True)
    plot_filename_scatter = os.path.join(output_dir, 'predictions_vs_targets_scatter.png')
    try:
        plt.savefig(plot_filename_scatter)
        print(f"Scatter plot saved to: {plot_filename_scatter}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()
    plt.close()

    # Plot 2: Residual Plot (Residuals vs. Predicted Values)
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5, edgecolor='k')
    plt.axhline(0, color='red', linestyle='--', lw=2, label='Zero Residual Line')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Target - Predicted)")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True)
    plot_filename_residual = os.path.join(output_dir, 'residual_plot.png')
    try:
        plt.savefig(plot_filename_residual)
        print(f"Residual plot saved to: {plot_filename_residual}")
    except Exception as e:
        print(f"Error saving residual plot: {e}")
    plt.show()
    plt.close() 

    # Plot 3: Histogram of Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Residuals (Target - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.grid(True)
    plot_filename_hist_residuals = os.path.join(output_dir, 'residuals_histogram.png')
    try:
        plt.savefig(plot_filename_hist_residuals)
        print(f"Histogram of residuals saved to: {plot_filename_hist_residuals}")
    except Exception as e:
        print(f"Error saving histogram of residuals: {e}")
    plt.show()
    plt.close() 


    # Plot 4: Distribution Comparison Plot (Histograms of Targets vs. Predictions)
    plt.figure(figsize=(12, 7)) # Slightly wider for two histograms
    plt.hist(targets, bins=30, alpha=0.7, label='Target Values', edgecolor='blue', color='skyblue')
    plt.hist(predictions, bins=30, alpha=0.7, label='Predicted Values', edgecolor='green', color='lightgreen')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Target vs. Predicted Values")
    plt.legend()
    plt.grid(True)
    plot_filename_dist_comp = os.path.join(output_dir, 'target_vs_pred_distribution.png')
    try:
        plt.savefig(plot_filename_dist_comp)
        print(f"Distribution comparison plot saved to: {plot_filename_dist_comp}")
    except Exception as e:
        print(f"Error saving distribution comparison plot: {e}")
    plt.show()
    plt.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze prediction performance from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file with 'pred' and 'target' columns.")

    args = parser.parse_args()

    analyze_predictions(args.csv_file)

    print("\nScript finished.") 
