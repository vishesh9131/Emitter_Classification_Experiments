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
        # Convert to tensors but keep on CPU (move to GPU in training loop)
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
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
        anchor_features = self.features[idx]
        anchor_label = self.labels[idx].item()
        
        # positive (same class)
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice(positive_indices)  # use numpy instead of torch.randint
        positive_features = self.features[positive_idx]
        
        # negative (different class)
        negative_label = np.random.choice([l for l in self.label_to_indices if l != anchor_label])
        negative_indices = self.label_to_indices[negative_label]
        negative_idx = np.random.choice(negative_indices)
        negative_features = self.features[negative_idx]
        
        return anchor_features, positive_features, negative_features

class UltraGPUOptimizedTripletDataset(Dataset):
    """Ultra-optimized GPU dataset that pre-computes all triplets on GPU"""
    def __init__(self, features: np.ndarray, labels: np.ndarray, device: torch.device = None, 
                 max_triplets_per_epoch: int = 100000):
        # Convert to tensors and move to GPU
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
        if device is not None:
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
        
        # create label to index mapping on GPU
        unique_labels = torch.unique(self.labels)
        self.label_to_indices = {}
        self.label_counts = {}
        
        for label in unique_labels:
            mask = (self.labels == label)
            indices = torch.where(mask)[0]
            self.label_to_indices[label.item()] = indices
            self.label_counts[label.item()] = len(indices)
        
        # check if we have enough classes and samples
        if len(self.label_to_indices) < 2:
            raise ValueError("Need at least 2 classes for triplet training")
        
        for label, count in self.label_counts.items():
            if count < 2:
                raise ValueError(f"Class {label} has less than 2 samples")
        
        # pre-compute all possible triplets on GPU
        self.max_triplets = min(max_triplets_per_epoch, len(features) * 10)  # limit memory usage
        self._precompute_triplets()
        
        print(f"Pre-computed {len(self.triplet_indices)} triplets on GPU")
    
    def _precompute_triplets(self):
        """Pre-compute all triplet combinations on GPU"""
        anchors = []
        positives = []
        negatives = []
        
        # get all unique labels
        unique_labels = list(self.label_to_indices.keys())
        
        # create triplets for each anchor
        for anchor_idx in range(len(self.features)):
            anchor_label = self.labels[anchor_idx].item()
            
            # get positive indices (same class)
            positive_indices = self.label_to_indices[anchor_label]
            if len(positive_indices) > 1:  # need at least 2 samples in class
                # randomly select positive
                pos_idx = torch.randint(0, len(positive_indices), (1,), device=self.device).item()
                positive_idx = positive_indices[pos_idx].item()
                
                # get negative indices (different class)
                negative_labels = [l for l in unique_labels if l != anchor_label]
                if negative_labels:
                    neg_label = negative_labels[torch.randint(0, len(negative_labels), (1,), device=self.device).item()]
                    negative_indices = self.label_to_indices[neg_label]
                    neg_idx = torch.randint(0, len(negative_indices), (1,), device=self.device).item()
                    negative_idx = negative_indices[neg_idx].item()
                    
                    anchors.append(anchor_idx)
                    positives.append(positive_idx)
                    negatives.append(negative_idx)
        
        # convert to tensors on GPU
        self.triplet_indices = torch.stack([
            torch.tensor(anchors, device=self.device),
            torch.tensor(positives, device=self.device),
            torch.tensor(negatives, device=self.device)
        ], dim=1)
        
        # limit number of triplets to prevent memory issues
        if len(self.triplet_indices) > self.max_triplets:
            perm = torch.randperm(len(self.triplet_indices), device=self.device)[:self.max_triplets]
            self.triplet_indices = self.triplet_indices[perm]
    
    def __len__(self) -> int:
        return len(self.triplet_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get pre-computed triplet indices
        anchor_idx, positive_idx, negative_idx = self.triplet_indices[idx]
        
        # return features directly (already on GPU)
        return (self.features[anchor_idx], 
                self.features[positive_idx], 
                self.features[negative_idx])

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