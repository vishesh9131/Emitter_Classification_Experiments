#!/usr/bin/env python3
"""
Triplet Loss implementation for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class TripletLoss(nn.Module):
    """
    Triplet Margin Loss for learning embeddings
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings (batch_size, embed_dim)
            positive: Positive embeddings (batch_size, embed_dim)
            negative: Negative embeddings (batch_size, embed_dim)
            
        Returns:
            Triplet loss value
        """
        # compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TripletLoss':
        """Create loss from configuration dictionary"""
        return cls(margin=config.get('margin', 1.0)) 