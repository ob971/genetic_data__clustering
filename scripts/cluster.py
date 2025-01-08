# scripts/cluster.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import joblib
import argparse

def determine_optimal_k(data, k_range):
    """
    Determines the optimal number of clusters (K) using Silhouette Scores.

    Parameters:
    - data (DataFrame): PCA-transformed data.
    - k_range (range): Range of K values to evaluate.

    Returns:
    - sil_scores (list): List of Silhouette Scores corresponding to each K.
    """
    sil_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        sil = silhouette_score(data, labels)
        sil_scores.append(sil)
        print(f"K={k}, Silhouette Score={sil:.4f}")
    return sil_scores

def plot_silhouette(k_range, sil_scores, output_path):
    """
    Plots Silhouette Scores for different K values.

    Parameters:
    - k_range (range): Range of K values.
    - sil_scores (list): Corresponding Silhouette Scores.
    - output_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10,6))
    plt.plot(k_range, sil_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different K')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Silhouette scores plot saved to {output_path}")

def perform_kmeans(data, n_clusters):
    """
    Applies K-Means clustering to the data.

    Parameters:
    - data (DataFrame): PCA-transformed data.
    - n_clusters (int): Number of clusters.

    Returns:
    - kmeans (KMeans object): Fitted K-Means model.
    - labels (ndarray): Cluster labels for each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    print(f"K-Means clustering performed with K={n_clusters}")
    return kmeans, labels

def cluster_data(input_path, output_path, k_min=2, k_max=10, plot_path='outputs/silhouette_scores.png'):
    """
    Orchestrates the clustering process: determines optimal K, performs clustering, and saves results.

    Parameters:
    - input_path (str): Path to PCA-transformed data CSV file.
    - output_path (str): Path to save cluster assignments CSV file.
    - k_min (int): Minimum number of clusters to evaluate.
    - k_max (int): Maximum number of clusters to evaluate.
    - plot_path (str): Path to save the Silhouette Scores plot.
    """
    # Load PCA-transformed data
    data = pd.read_csv(input_path)
    feature_cols = [col for col in data.columns if col.startswith('PC')]
    X = data[feature_cols]
    print(f"Loaded PCA data with shape: {X.shape}")

    # Determine optimal K
    k_range = range(k_min, k_max + 1)
    sil_scores = determine_optimal_k(X, k_range)

    # Plot Silhouette Scores
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_silhouette(k_range, sil_scores, plot_path)

    # Choose K with the highest Silhouette Score
    optimal_k = k_range[sil_scores.index(max(sil_scores))]
    print(f"Optimal number of clusters determined: K={optimal_k}")

    # Perform K-Means clustering with optimal K
    kmeans_model, labels = perform_kmeans(X, optimal_k)

    # Save the K-Means model
    model_path = os.path.join(os.path.dirname(output_path), 'optimal_clusters.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(kmeans_model, model_path)
    print(f"K-Means model saved to {model_path}")

    # Assign cluster labels to the original data
    data['Cluster'] = labels
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Cluster assignments saved to {output_path}")

    # Generate Clustering Report
    generate_report(k_range, sil_scores, optimal_k, data, 'outputs/reports/clustering_report.txt')

def generate_report(k_range, sil_scores, optimal_k, data, report_path):
    """
    Generates a textual report summarizing clustering results.

    Parameters:
    - k_range (range): Range of K values evaluated.
    - sil_scores (list): Silhouette Scores for each K.
    - optimal_k (int): Optimal number of clusters determined.
    - data (DataFrame): Data with cluster assignments.
    - report_path (str): Path to save the clustering report.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("Clustering Report\n")
        f.write("=================\n\n")
        f.write(f"Optimal Number of Clusters (K): {optimal_k}\n\n")
        f.write("Silhouette Scores:\n")
        for k, score in zip(k_range, sil_scores):
            f.write(f"- K={k}: {score:.4f}\n")
        f.write("\nCluster Distribution:\n")
        cluster_counts = data['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            yes_count = data[data['Cluster'] == cluster]['Disease_Status'].value_counts().get('Yes', 0)
            no_count = data[data['Cluster'] == cluster]['Disease_Status'].value_counts().get('No', 0)
            f.write(f"- Cluster {cluster}: {count} samples (Yes: {yes_count}, No: {no_count})\n")
        f.write("\nObservations:\n")
        f.write(f"- The highest Silhouette Score of {max(sil_scores):.4f} was achieved with K={optimal_k} clusters.\n")
        f.write("- Clusters show varying distributions of disease statuses, indicating potential genetic associations.\n")
        f.write("\nRecommendations:\n")
        f.write("- Further analyze the SNPs contributing most to each principal component.\n")
        f.write("- Explore additional clustering algorithms for comparison.\n")
        f.write("- Validate clusters with external biological data if available.\n")
    print(f"Clustering report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform clustering on PCA-transformed genetic data.")
    parser.add_argument('--input', type=str, default='data/processed/pca_data.csv', help='Path to PCA-transformed data CSV.')
    parser.add_argument('--output', type=str, default='outputs/clusters.csv', help='Path to save cluster assignments CSV.')
    parser.add_argument('--k_min', type=int, default=2, help='Minimum number of clusters to evaluate.')
    parser.add_argument('--k_max', type=int, default=10, help='Maximum number of clusters to evaluate.')
    parser.add_argument('--plot_path', type=str, default='outputs/silhouette_scores.png', help='Path to save Silhouette Scores plot.')
    parser.add_argument('--report_path', type=str, default='outputs/reports/clustering_report.txt', help='Path to save clustering report.')

    args = parser.parse_args()

    cluster_data(args.input, args.output, args.k_min, args.k_max, args.plot_path)

if __name__ == "__main__":
    main()
