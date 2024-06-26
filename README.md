# DBSCAN and Agglomerative Clustering

This repository demonstrates how to perform DBSCAN and Agglomerative Clustering on a time series dataset using Principal Component Analysis (PCA) for dimensionality reduction.

## Directory Structure
DBSCAN-and-Agglomerative-Clustering/
│
├── data/
│ └── DetailedData.xls
├── src/
│ └── clustering.py
└── README.md

bash
Copy code

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/skalaliya/DBSCAN-and-Agglomerative-Clustering.git
   cd DBSCAN-and-Agglomerative-Clustering
Create and activate the Conda environment:

sh
Copy code
conda create --name DBSCANandAGGLO python=3.8
conda activate DBSCANandAGGLO
Install the required libraries:

sh
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn
Place your dataset in the data directory.

Usage
Run the clustering.py script to perform clustering and visualize the results:

sh
Copy code
python src/clustering.py
Description
PCA: Reduces the dimensionality of the dataset.
DBSCAN: Clusters the data using the Density-Based Spatial Clustering of Applications with Noise algorithm.
Agglomerative Clustering: Clusters the data using a hierarchical clustering approach.
Visualization: Provides 2D and 3D visualizations of the clustering results.
Results
The script prints the Silhouette Scores for DBSCAN and Agglomerative Clustering and generates plots to visualize the clustering results in 2D and 3D.

Contributing
Contributions are welcome. Please open an issue or submit a pull request.

License
This project is licensed under the MIT License.

sql
Copy code

### Final Steps

1. **Add and Commit Files:**
   ```sh
   git add .
   git commit -m "Initial commit with clustering code and README"
Push to GitHub:
sh
Copy code
git push origin main