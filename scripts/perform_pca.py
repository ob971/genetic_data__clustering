# scripts/perform_pca.py

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import joblib
import argparse

def perform_pca(input_path, output_path, n_components=50):
    """
    Applies PCA to the preprocessed genetic data for dimensionality reduction.

    Parameters:
    - input_path (str): Path to the preprocessed data CSV file.
    - output_path (str): Path to save the PCA-transformed data CSV file.
    - n_components (int): Number of principal components to retain.
    """
    # Load preprocessed data
    data = pd.read_csv(input_path)
    print(f"Loaded preprocessed data with shape: {data.shape}")

    # Separate features and labels
    feature_cols = [col for col in data.columns if col.startswith('SNP')]
    X = data[feature_cols]
    y = data['Disease_Status']
    sample_ids = data['Sample_ID']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data standardized.")

    # Save the scaler for future use
    scaler_path = os.path.join(os.path.dirname(output_path), 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA applied with {n_components} components.")

    # Save the PCA model for future use
    pca_path = os.path.join(os.path.dirname(output_path), 'pca_model.pkl')
    joblib.dump(pca, pca_path)
    print(f"PCA model saved to {pca_path}")

    # Create a DataFrame for PCA results
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(X_pca, columns=pca_columns)
    pca_df['Sample_ID'] = sample_ids
    pca_df['Disease_Status'] = y

    # Save PCA-transformed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pca_df.to_csv(output_path, index=False)
    print(f"PCA-transformed data saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform PCA on preprocessed genetic data.")
    parser.add_argument('--input', type=str, default='data/processed/preprocessed_data.csv', help='Path to preprocessed data CSV.')
    parser.add_argument('--output', type=str, default='data/processed/pca_data.csv', help='Path to save PCA-transformed data CSV.')
    parser.add_argument('--n_components', type=int, default=50, help='Number of principal components to retain.')

    args = parser.parse_args()

    perform_pca(args.input, args.output, args.n_components)

if __name__ == "__main__":
    main()
