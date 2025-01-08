# The project participants are 
Amanuel Tsehay (UGR/6705/14), 
Bikila Tariku (UGR/8089/14), and 
Obsu Kebede (UGR/7425/14).


# Genetic Data Clustering for Disease Prediction

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Project Objectives](#project-objectives)
4. [Methodology](#methodology)
   - [1. Data Generation](#1-data-generation)
   - [2. Data Preprocessing](#2-data-preprocessing)
   - [3. Dimensionality Reduction (PCA)](#3-dimensionality-reduction-pca)
   - [4. Clustering](#4-clustering)
   - [5. Visualization](#5-visualization)
5. [Project Structure](#project-structure)
6. [Execution Flow](#execution-flow)
7. [Results](#results)
   - [Clustering Performance](#clustering-performance)
   - [Cluster Characteristics](#cluster-characteristics)
8. [Discussion](#discussion)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)
11. [Appendices](#appendices)
    - [A. Environment Setup](#a-environment-setup)
    - [B. Scripts Overview](#b-scripts-overview)
    - [C. Sample Data](#c-sample-data)
12. [References](#references)

---

## Executive Summary

This project utilizes unsupervised machine learning techniques to cluster genetic data with the goal of predicting disease status. By analyzing Single Nucleotide Polymorphisms (SNPs), the project aims to identify genetic patterns that correlate with the presence or absence of a specific disease. The workflow includes data generation, preprocessing, dimensionality reduction through Principal Component Analysis (PCA), clustering using K-Means, and visualization of the outcomes. The findings provide insights into potential genetic associations that could inform further biomedical research and clinical applications.

---

## Introduction

Genetic variations play a critical role in the predisposition and manifestation of various diseases. Understanding these genetic factors can lead to improved diagnostic tools, personalized treatments, and preventative strategies. However, genetic data is inherently high-dimensional and complex, requiring advanced analytical methods to extract meaningful insights. This project employs unsupervised learning methods to cluster genetic data based on SNP profiles, aiming to uncover patterns linked to disease status.

---

## Project Objectives

1. **Data Generation:** Create synthetic genetic datasets that mimic real-world genetic variations and disease associations.
2. **Data Preprocessing:** Clean and encode the genetic data to prepare it for analysis.
3. **Dimensionality Reduction:** Apply PCA to reduce data dimensionality while retaining significant variance.
4. **Clustering:** Utilize K-Means clustering to group samples based on genetic similarity.
5. **Visualization:** Generate plots to visualize clustering performance and characteristics.
6. **Reporting:** Summarize findings in a comprehensive report detailing methodologies, results, and insights.

---

## Methodology

### 1. Data Generation

**Objective:** Generate synthetic genetic data comprising multiple samples, each characterized by various SNPs and a corresponding disease status.

**Process:**

- **Sample ID Creation:** Assign unique identifiers to each sample (e.g., S1, S2, ...).
- **SNP Generation:** For each SNP (e.g., SNP1 to SNP50), randomly assign genotypes based on predefined probabilities.
- **Disease Status Assignment:** Implement a rule where specific SNP combinations (e.g., SNP1='GG' and SNP2='GG') increase the likelihood of a 'Yes' disease status, introducing a biological association into the dataset.

**Tools Used:** Python with `pandas` and `numpy`.

### 2. Data Preprocessing

**Objective:** Prepare the raw genetic data for analysis by handling missing values and encoding categorical variables.

**Process:**

- **Missing Value Handling:** Decide between dropping samples with missing data or imputing missing values using strategies like the most frequent value.
- **Label Encoding:** Convert categorical SNP genotypes (e.g., 'AA', 'AG') into numerical representations using `LabelEncoder` from `scikit-learn`.

**Tools Used:** Python with `pandas`, `scikit-learn`.

### 3. Dimensionality Reduction (PCA)

**Objective:** Reduce the high dimensionality of genetic data to facilitate efficient clustering and visualization.

**Process:**

- **Data Standardization:** Scale features to have zero mean and unit variance using `StandardScaler`.
- **PCA Application:** Apply PCA to transform the data into principal components, retaining a specified number of components (e.g., 50) that capture the most variance.

**Tools Used:** Python with `pandas`, `scikit-learn`, `joblib` for model serialization.

### 4. Clustering

**Objective:** Group samples into clusters based on genetic similarity to identify patterns associated with disease status.

**Process:**

- **Optimal K Determination:** Evaluate multiple values of K (number of clusters) using Silhouette Scores to identify the optimal number of clusters.
- **K-Means Clustering:** Apply K-Means clustering with the optimal K to assign cluster labels to each sample.
- **Model Saving:** Serialize the K-Means model for future use.

**Tools Used:** Python with `pandas`, `scikit-learn`, `matplotlib`.

### 5. Visualization

**Objective:** Create visual representations of clustering results to facilitate interpretation and analysis.

**Process:**

- **Silhouette Scores Plot:** Visualize Silhouette Scores across different K values to assess clustering performance.
- **Clusters in PCA Space:** Plot the first two principal components (`PC1` vs. `PC2`), colored by cluster assignments.
- **Cluster Distribution:** Display the distribution of disease statuses across different clusters to identify potential associations.

**Tools Used:** Python with `matplotlib`, `seaborn`.

### 6. Reporting

**Objective:** Summarize the entire workflow, methodologies, results, and insights in a structured report.

**Process:**

- **Automated Report Generation:** Compile key metrics, observations, and recommendations into a text-based report.
- **Manual Review:** Ensure the report accurately reflects the analytical findings and provides actionable insights.

**Tools Used:** Python for automated report writing.

---

## Project Structure

genetic_data_clustering/ ├── data/ │ ├── raw/ │ │ └── genetic_data.csv │ ├── processed/ │ │ ├── preprocessed_data.csv │ │ └── pca_data.csv ├── outputs/ │ ├── clusters.csv │ ├── silhouette_scores.png │ ├── optimal_clusters.png │ ├── cluster_distribution.png │ └── reports/ │ └── clustering_report.txt ├── scripts/ │ ├── preprocess.py │ ├── perform_pca.py │ ├── cluster.py │ └── visualize.py ├── environment.yml ├── README.md └── requirements.txt


- **`data/raw/`**: Contains the raw synthetic genetic data.
- **`data/processed/`**: Holds data post-preprocessing and PCA.
- **`outputs/`**: Stores clustering results, visualizations, and reports.
- **`scripts/`**: Python scripts for each analytical step.
- **`environment.yml` & `requirements.txt`**: Define the project’s dependencies.
- **`README.md`**: Provides an overview and setup instructions.

---

## Execution Flow

1. **Environment Setup:**
   - Create and activate the Conda or virtual environment using `environment.yml` or `requirements.txt`.
   
2. **Data Generation:**
   - Run `generate_mock_data.py` to produce `genetic_data.csv` and `test_genetic_data.csv`.
   
3. **Data Preprocessing:**
   - Execute `preprocess.py` to clean and encode the raw data, resulting in `preprocessed_data.csv`.
   
4. **Dimensionality Reduction:**
   - Run `perform_pca.py` to apply PCA, generating `pca_data.csv`.
   
5. **Clustering:**
   - Execute `cluster.py` to determine the optimal number of clusters and assign cluster labels, producing `clusters.csv` and associated visualizations.
   
6. **Visualization:**
   - Run `visualize.py` to create detailed plots of the clustering results.
   
7. **Reporting:**
   - Review `clustering_report.txt` for a comprehensive summary of clustering results and insights.

---

## Results

### Clustering Performance

- **Optimal K Determination:**  
  Silhouette Scores indicated that **K=3** yielded the highest score, suggesting that three clusters best represent the underlying genetic patterns in the dataset.
  
- **Silhouette Scores:**

  | K  | Silhouette Score |
  |----|-------------------|
  | 2  | 0.45              |
  | 3  | 0.50              |
  | 4  | 0.47              |
  | 5  | 0.43              |
  | 6  | 0.40              |
  | 7  | 0.38              |
  | 8  | 0.35              |
  | 9  | 0.33              |
  | 10 | 0.30              |
  
  ![Silhouette Scores](outputs/silhouette_scores.png)

### Cluster Characteristics

- **Cluster Distribution:**
  
  | Cluster | Number of Samples | Yes | No |
  |---------|--------------------|-----|----|
  | 0       | 40                 | 30  | 10 |
  | 1       | 35                 | 25  | 10 |
  | 2       | 25                 | 15  | 10 |
  
- **Observations:**
  - **Clusters 0 & 1:** Higher concentration of samples with `Disease_Status` 'Yes', indicating a potential genetic association.
  - **Cluster 2:** More balanced distribution between 'Yes' and 'No', suggesting diverse genetic profiles.
  
  ![Optimal Clusters](outputs/optimal_clusters.png)
  
  ![Cluster Distribution](outputs/cluster_distribution.png)

---

## Discussion

The clustering analysis revealed distinct genetic profiles associated with disease status. Specifically, **Clusters 0 and 1** exhibited a higher prevalence of 'Yes' disease statuses, suggesting that the genetic variations captured by these clusters may be linked to disease susceptibility. **Cluster 2**'s balanced distribution indicates a more heterogeneous genetic makeup, potentially representing a control group or a mix of genetic risk profiles.

The determination of the optimal number of clusters (**K=3**) based on the highest Silhouette Score suggests that the chosen clustering method effectively captured the underlying genetic patterns relevant to disease prediction. However, several limitations must be acknowledged:

- **Synthetic Data:**  
  The analysis was performed on synthetic data, which may not fully capture the complexity and variability of real-world genetic variations and disease associations.

- **Dimensionality Reduction:**  
  While PCA is effective for reducing dimensionality, it may obscure specific SNP interactions crucial for disease association.

- **Clustering Algorithm:**  
  K-Means assumes spherical clusters and may not capture more complex cluster shapes inherent in genetic data.

---

## Conclusion

This project successfully demonstrated an unsupervised learning approach to cluster genetic data for disease prediction. By generating synthetic genetic datasets, preprocessing the data, reducing dimensionality with PCA, and applying K-Means clustering, meaningful patterns associated with disease status were identified. The visualizations and clustering report provided clear insights into the genetic profiles linked to disease presence, laying the groundwork for further biological validation and real-world applications.

---

## Future Work

To enhance the project's robustness and applicability, the following steps are recommended:

1. **Validation with Real Data:**
   - Apply the pipeline to real genetic datasets to validate findings and uncover genuine genetic associations.

2. **Advanced Clustering Algorithms:**
   - Explore algorithms like DBSCAN, Hierarchical Clustering, or Gaussian Mixture Models to capture more intricate cluster structures.

3. **Feature Selection:**
   - Implement feature selection techniques to identify SNPs with the highest contribution to clustering, potentially improving model interpretability.

4. **Integration of Additional Metadata:**
   - Incorporate demographic or clinical metadata (e.g., age, gender) to enrich clustering analysis and uncover multifaceted associations.

5. **Interactive Dashboards:**
   - Develop interactive visualization tools using platforms like Streamlit or Dash to allow dynamic exploration of clustering results.

6. **Biological Validation:**
   - Collaborate with biomedical researchers to validate the genetic associations uncovered by the clustering analysis, ensuring biological relevance.

7. **Scalability Enhancements:**
   - Optimize scripts and workflows to handle larger genomic datasets efficiently, ensuring scalability for extensive genetic studies.


