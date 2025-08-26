#!/usr/bin/env python3
"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for contrastive learning
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss
        
        Args:
            z1: First set of embeddings (batch_size, embed_dim)
            z2: Second set of embeddings (batch_size, embed_dim)
            
        Returns:
            NT-Xent loss value
        """
        batch_size = z1.shape[0]
        
        # concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, embed_dim)
        
        # compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # create labels for positive pairs
        labels = torch.arange(2 * batch_size, device=z.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)
        labels = labels[~mask].view(2 * batch_size, -1)
        
        # compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels.argmax(dim=1))
        
        return loss
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NTXentLoss':
        """Create loss from configuration dictionary"""
        return cls(temperature=config.get('temperature', 0.1)) 