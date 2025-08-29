#!/usr/bin/env python3
"""
Evaluation utilities for Emitter Classification
"""

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster assignments
        
    Returns:
        Clustering accuracy
    """
    # create confusion matrix
    cm = pd.crosstab(y_pred, y_true)
    
    # use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cm.values)
    
    # compute accuracy
    accuracy = cm.values[row_ind, col_ind].sum() / len(y_true)
    
    return accuracy

def gpu_kmeans_clustering(embeddings: torch.Tensor, n_clusters: int, max_iter: int = 300, 
                         device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-optimized K-means clustering using PyTorch
    
    Args:
        embeddings: Embeddings tensor on GPU
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        device: Target device
        
    Returns:
        Tuple of (cluster_labels, cluster_centers) on GPU
    """
    if device is None:
        device = embeddings.device
    
    n_samples, n_features = embeddings.shape
    
    # initialize centroids randomly
    indices = torch.randperm(n_samples, device=device)[:n_clusters]
    centroids = embeddings[indices].clone()
    
    for iteration in range(max_iter):
        # compute distances to centroids
        distances = torch.cdist(embeddings, centroids)
        
        # assign to nearest centroid
        labels = torch.argmin(distances, dim=1)
        
        # update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                new_centroids[k] = embeddings[mask].mean(dim=0)
            else:
                # if no points assigned to this cluster, keep old centroid
                new_centroids[k] = centroids[k]
        
        # check convergence
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
            
        centroids = new_centroids
    
    return labels, centroids

def gpu_silhouette_score(embeddings: torch.Tensor, labels: torch.Tensor, device: torch.device = None) -> float:
    """
    GPU-optimized silhouette score computation
    
    Args:
        embeddings: Embeddings tensor on GPU
        labels: Cluster labels on GPU
        device: Target device
        
    Returns:
        Silhouette score
    """
    if device is None:
        device = embeddings.device
    
    n_samples = embeddings.shape[0]
    unique_labels = torch.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    # compute pairwise distances
    distances = torch.cdist(embeddings, embeddings)
    
    silhouette_scores = torch.zeros(n_samples, device=device)
    
    for i in range(n_samples):
        # get cluster of current point
        current_cluster = labels[i]
        
        # compute intra-cluster distance (a)
        intra_mask = (labels == current_cluster)
        if intra_mask.sum() > 1:  # more than just this point
            a = distances[i, intra_mask].sum() / (intra_mask.sum() - 1)
        else:
            a = 0.0
        
        # compute nearest inter-cluster distance (b)
        b = float('inf')
        for cluster in unique_labels:
            if cluster != current_cluster:
                inter_mask = (labels == cluster)
                if inter_mask.sum() > 0:
                    inter_dist = distances[i, inter_mask].mean()
                    b = min(b, inter_dist)
        
        if b == float('inf'):
            silhouette_scores[i] = 0.0
        else:
            silhouette_scores[i] = (b - a) / max(a, b)
    
    return silhouette_scores.mean().item()

def gpu_clustering_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, device: torch.device = None) -> float:
    """
    GPU-optimized clustering accuracy computation
    
    Args:
        y_true: True labels on GPU
        y_pred: Predicted cluster labels on GPU
        device: Target device
        
    Returns:
        Clustering accuracy
    """
    if device is None:
        device = y_true.device
    
    # create confusion matrix on GPU
    unique_true = torch.unique(y_true)
    unique_pred = torch.unique(y_pred)
    
    n_true = len(unique_true)
    n_pred = len(unique_pred)
    
    confusion_matrix = torch.zeros(n_pred, n_true, device=device)
    
    for i, pred_label in enumerate(unique_pred):
        for j, true_label in enumerate(unique_true):
            mask = (y_pred == pred_label) & (y_true == true_label)
            confusion_matrix[i, j] = mask.sum()
    
    # use Hungarian algorithm (still on CPU for now, but matrix is small)
    confusion_cpu = confusion_matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-confusion_cpu)
    
    # compute accuracy
    accuracy = confusion_cpu[row_ind, col_ind].sum() / len(y_true)
    
    return accuracy

def evaluate_model(model: torch.nn.Module, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict[str, float]:
    """
    Evaluate model using clustering accuracy
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    with torch.no_grad():
        # convert to tensor and move to device
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # get embeddings
        embeddings = model(X_tensor).cpu().numpy()
    
    # Check if we have real labels or dummy labels
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        # This is unlabeled test data with dummy labels
        print("Warning: Test data appears to be unlabeled (all dummy labels). Using clustering-only evaluation.")
        
        # Use silhouette score for unlabeled data
        from sklearn.metrics import silhouette_score
        
        # Try different numbers of clusters and find the best silhouette score
        best_silhouette = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, min(11, len(embeddings) // 10)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                silhouette = silhouette_score(embeddings, cluster_labels)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_n_clusters = n_clusters
            except:
                continue
        
        # Use the best number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        return {
            'clustering_accuracy': 0.0,  # Not meaningful for unlabeled data
            'silhouette_score': best_silhouette,
            'n_clusters': best_n_clusters,
            'embedding_shape': embeddings.shape,
            'note': 'Unlabeled test data - accuracy not meaningful'
        }
    else:
        # This is labeled test data
        # perform clustering
        n_clusters = len(unique_labels)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # compute accuracy
        accuracy = clustering_accuracy(y_test, cluster_labels)
        
        return {
            'clustering_accuracy': accuracy,
            'n_clusters': n_clusters,
            'embedding_shape': embeddings.shape
        } 

def gpu_evaluate_model(model: torch.nn.Module, 
                      X_test: torch.Tensor, 
                      y_test: torch.Tensor,
                      device: torch.device = None) -> Dict[str, float]:
    """
    GPU-optimized model evaluation
    
    Args:
        model: Trained model
        X_test: Test features on GPU
        y_test: Test labels on GPU
        device: Target device
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = X_test.device
    
    model.eval()
    
    with torch.no_grad():
        # get embeddings (already on GPU)
        embeddings = model(X_test)
    
    # Check if we have real labels or dummy labels
    unique_labels = torch.unique(y_test)
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        # This is unlabeled test data with dummy labels
        print("Warning: Test data appears to be unlabeled (all dummy labels). Using GPU clustering evaluation.")
        
        # Try different numbers of clusters and find the best silhouette score
        best_silhouette = -1
        best_n_clusters = 2
        best_labels = None
        
        for n_clusters in range(2, min(11, len(embeddings) // 10)):
            try:
                labels, _ = gpu_kmeans_clustering(embeddings, n_clusters, device=device)
                silhouette = gpu_silhouette_score(embeddings, labels, device=device)
                
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_n_clusters = n_clusters
                    best_labels = labels
            except:
                continue
        
        return {
            'clustering_accuracy': 0.0,  # Not meaningful for unlabeled data
            'silhouette_score': best_silhouette,
            'n_clusters': best_n_clusters,
            'embedding_shape': embeddings.shape,
            'note': 'Unlabeled test data - accuracy not meaningful'
        }
    else:
        # This is labeled test data - use GPU clustering
        n_clusters = len(unique_labels)
        labels, _ = gpu_kmeans_clustering(embeddings, n_clusters, device=device)
        
        # compute clustering accuracy on GPU
        accuracy = gpu_clustering_accuracy(y_test, labels, device=device)
        silhouette = gpu_silhouette_score(embeddings, labels, device=device)
        
        return {
            'clustering_accuracy': accuracy,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters,
            'embedding_shape': embeddings.shape
        } 