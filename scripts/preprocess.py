# scripts/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os
import argparse

def preprocess_data(input_path, output_path, handle_missing='drop', impute_strategy='most_frequent'):
    """
    Preprocesses the genetic data by handling missing values and encoding SNPs.

    Parameters:
    - input_path (str): Path to the raw genetic data CSV file.
    - output_path (str): Path to save the preprocessed data CSV file.
    - handle_missing (str): Strategy to handle missing data ('drop' or 'impute').
    - impute_strategy (str): Strategy for imputation if handling missing data by imputation.
    """
    # Load raw data
    data = pd.read_csv(input_path)
    print(f"Loaded data with shape: {data.shape}")

    # Handle missing values
    if handle_missing == 'drop':
        initial_shape = data.shape
        data.dropna(inplace=True)
        print(f"Dropped rows with missing values: {initial_shape} -> {data.shape}")
    elif handle_missing == 'impute':
        imputer = SimpleImputer(strategy=impute_strategy)
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        print(f"Imputed missing values using strategy: {impute_strategy}")
        data = data_imputed
    else:
        raise ValueError("handle_missing must be either 'drop' or 'impute'.")

    # Encode categorical SNP data
    snp_columns = [col for col in data.columns if col.startswith('SNP')]
    le = LabelEncoder()
    for col in snp_columns:
        data[col] = le.fit_transform(data[col].astype(str))
        print(f"Encoded {col} with LabelEncoder.")

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess genetic data for clustering.")
    parser.add_argument('--input', type=str, default='data/raw/genetic_data.csv', help='Path to raw genetic data CSV.')
    parser.add_argument('--output', type=str, default='data/processed/preprocessed_data.csv', help='Path to save preprocessed data CSV.')
    parser.add_argument('--handle_missing', type=str, default='drop', choices=['drop', 'impute'], help="Strategy to handle missing data.")
    parser.add_argument('--impute_strategy', type=str, default='most_frequent', choices=['mean', 'median', 'most_frequent'], help="Imputation strategy if handling missing data by imputation.")

    args = parser.parse_args()

    preprocess_data(args.input, args.output, args.handle_missing, args.impute_strategy)

if __name__ == "__main__":
    main()
