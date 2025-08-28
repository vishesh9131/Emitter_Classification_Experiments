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