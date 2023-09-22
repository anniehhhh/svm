import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset from a CSV file
def load_dataset(filename):
    dataset = pd.read_csv(filename)
    return dataset.iloc[:, :-1].values  # Extract only the feature columns

# Initialize K cluster centroids randomly
def initialize_centroids(data, K):
    centroids = data.copy()
    np.random.shuffle(centroids)
    return centroids[:K]

# Assign each data point to the nearest centroid
def assign_to_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Update the centroids based on the assigned data points
def update_centroids(data, cluster_assignments, K):
    centroids = np.zeros((K, data.shape[1]))
    for k in range(K):
        cluster_data = data[cluster_assignments == k]
        if len(cluster_data) > 0:
            centroids[k] = cluster_data.mean(axis=0)
    return centroids

# K-Means clustering algorithm
def kmeans(data, K, max_iterations=100):
    centroids = initialize_centroids(data, K)
    prev_centroids = None
    iteration = 0
    
    while not np.array_equal(centroids, prev_centroids) and iteration < max_iterations:
        cluster_assignments = assign_to_clusters(data, centroids)
        prev_centroids = centroids.copy()
        centroids = update_centroids(data, cluster_assignments, K)
        iteration += 1
    return centroids, cluster_assignments

K = 4
data = load_dataset('Iris - Iris.csv')
    
centroids, cluster_assignments= kmeans(data, K)

plt.scatter(data[:, 0], data[:, 1], c=cluster_assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title('K-Means Clustering of Iris Dataset')
plt.legend()
plt.show()
