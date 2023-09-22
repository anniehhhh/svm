import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset from a CSV file
iris_df = pd.read_csv('Iris - Iris.csv')

# Extract the features (sepal length and sepal width) for clustering
X = iris_df[['SepalLengthCm', 'SepalWidthCm']]

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (K)
K = 3

# Create a K-Means model
kmeans = KMeans(n_clusters=K, random_state=0)

# Fit the model to the data
kmeans.fit(X_scaled)

# Get the cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Add cluster labels to the original Iris dataset
iris_df['Cluster'] = labels

# Visualize the data with cluster colors
colors = ['red', 'green', 'blue','pink']
for i in range(K):
    cluster_data = iris_df[iris_df['Cluster'] == i]
    plt.scatter(cluster_data['SepalLengthCm'], cluster_data['SepalWidthCm'], c=colors[i], label=f'Cluster {i+1}')

# Plot the cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', label='Cluster Centers')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend()
plt.title('K-Means Clustering of Iris Dataset')
plt.show()
