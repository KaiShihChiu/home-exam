from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import RANSACRegressor
import numpy as np

def perform_kmeans(data, min_clusters=2, max_clusters=6):
    """
    Perform KMeans clustering with silhouette score to find the optimal number of clusters.
    
    Args:
        data (np.array): Feature data (e.g., contour areas and perimeters).
        min_clusters (int): Minimum number of clusters to test.
        max_clusters (int): Maximum number of clusters to test.
    
    Returns:
        tuple: Best number of clusters, cluster labels, and KMeans model.
    """
    best_k = min_clusters
    best_silhouette = -1
    best_model = None
    
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        if silhouette_avg > best_silhouette:
            best_k = k
            best_silhouette = silhouette_avg
            best_model = kmeans

    return best_k, best_model.labels_, best_model

def apply_ransac(data):
    """
    Apply RANSAC to remove outliers from the dataset.
    
    Args:
        data (np.array): Feature data (e.g., contour areas and perimeters).
    
    Returns:
        tuple: Inlier mask (boolean array), RANSAC model, and inlier data points.
    """
    ransac = RANSACRegressor()
    ransac.fit(data[:, 0].reshape(-1, 1), data[:, 1])
    
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return inlier_mask, ransac, data[inlier_mask]
