import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Ensure plots are displayed inline
#%matplotlib inline

print("All libraries imported successfully!")

# Correct file path
file_path = '/Users/skalaliya/Desktop/YouTube/DBSCAN and Agglomerative Clustering/DetailedData.xls'

def load_data(file_path):
    """Load the dataset from the given file path."""
    return pd.read_excel(file_path)

def preprocess_data(data):
    """Scale the data using StandardScaler."""
    X = data.values  # Adjust based on your data structure
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def perform_pca(X_scaled, n_components=40):
    """Reduce dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def dbscan_clustering(X_pca):
    """Apply DBSCAN clustering algorithm."""
    dbscan = DBSCAN(metric="euclidean")
    y_dbscan = dbscan.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, y_dbscan)
    print(f'DBSCAN Silhouette Score: {silhouette_avg:.2f}')
    return y_dbscan

def agglomerative_clustering(X_pca, n_clusters=3):
    """Apply Agglomerative clustering algorithm."""
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    y_agg = agglomerative.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, y_agg)
    print(f'Agglomerative Clustering Silhouette Score: {silhouette_avg:.2f}')
    return y_agg

def visualize_2d(X_pca, labels, title):
    """Visualize the first two principal components with clustering labels in 2D."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cluster', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_3d(X_pca, labels, title):
    """Visualize the first three principal components with clustering labels in 3D."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis', s=100, alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", fontsize=10)
    ax.add_artist(legend1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Define the file path to your dataset
    file_path = '/Users/skalaliya/Desktop/YouTube/DBSCAN and Agglomerative Clustering/DetailedData .xls'
    
    # Load the data
    data = load_data(file_path)
    
    # Preprocess the data
    X_scaled = preprocess_data(data)
    
    # Perform PCA
    X_pca = perform_pca(X_scaled)
    
    # Perform DBSCAN clustering and print silhouette score
    y_dbscan = dbscan_clustering(X_pca)
    
    # Perform Agglomerative clustering and print silhouette score
    y_agg = agglomerative_clustering(X_pca)
    
    # Visualize clustering results in 2D
    visualize_2d(X_pca, y_dbscan, 'PCA - Time Series Data with DBSCAN Clustering')
    visualize_2d(X_pca, y_agg, 'PCA - Time Series Data with Agglomerative Clustering')
    
    # Visualize clustering results in 3D
    visualize_3d(X_pca, y_dbscan, '3D PCA - Time Series Data with DBSCAN Clustering')
    visualize_3d(X_pca, y_agg, '3D PCA - Time Series Data with Agglomerative Clustering')

# Run the main function
if __name__ == "__main__":
    main()
