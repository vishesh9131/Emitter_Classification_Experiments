#!/usr/bin/env python3
"""
Dataset classes for Emitter Classification
"""

import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict, Any

class PDWDataset(Dataset):
    """Basic PDW dataset for supervised learning"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class TripletPDWDataset(Dataset):
    """Dataset for triplet loss training"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        
        # create label to index mapping
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_to_indices[label].append(i)
        
        # check if we have enough classes and samples
        if len(self.label_to_indices) < 2:
            raise ValueError("Need at least 2 classes for triplet training")
        
        # check if each class has at least 2 samples
        for label, indices in self.label_to_indices.items():
            if len(indices) < 2:
                raise ValueError(f"Class {label} has less than 2 samples")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # anchor
        anchor_features = torch.from_numpy(self.features[idx]).float()
        anchor_label = self.labels[idx]
        
        # positive (same class)
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice(positive_indices)
        positive_features = torch.from_numpy(self.features[positive_idx]).float()
        
        # negative (different class)
        negative_label = np.random.choice([l for l in self.label_to_indices if l != anchor_label])
        negative_indices = self.label_to_indices[negative_label]
        negative_idx = np.random.choice(negative_indices)
        negative_features = torch.from_numpy(self.features[negative_idx]).float()
        
        return anchor_features, positive_features, negative_features

class PairPDWDataset(Dataset):
    """Dataset for contrastive learning with pairs"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        
        # create label to index mapping
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_to_indices[label].append(i)
        
        # check if we have enough classes and samples
        if len(self.label_to_indices) < 2:
            raise ValueError("Need at least 2 classes for pair training")
        
        # check if each class has at least 2 samples
        for label, indices in self.label_to_indices.items():
            if len(indices) < 2:
                raise ValueError(f"Class {label} has less than 2 samples")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # anchor
        anchor_features = torch.from_numpy(self.features[idx]).float()
        anchor_label = self.labels[idx]
        
        # positive (same class)
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice(positive_indices)
        positive_features = torch.from_numpy(self.features[positive_idx]).float()
        
        return anchor_features, positive_features 