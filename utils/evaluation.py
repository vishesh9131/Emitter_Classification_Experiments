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
    
    # perform clustering
    n_clusters = len(np.unique(y_test))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # compute accuracy
    accuracy = clustering_accuracy(y_test, cluster_labels)
    
    return {
        'clustering_accuracy': accuracy,
        'n_clusters': n_clusters,
        'embedding_shape': embeddings.shape
    } 