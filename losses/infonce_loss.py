#!/usr/bin/env python3
"""
InfoNCE Loss implementation for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive learning
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss
        
        Args:
            query: Query embeddings (batch_size, embed_dim)
            key: Key embeddings (batch_size, embed_dim)
            
        Returns:
            InfoNCE loss value
        """
        batch_size = query.shape[0]
        
        # normalize embeddings
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        
        # compute similarity matrix
        logits = torch.mm(query, key.T) / self.temperature
        
        # create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=query.device)
        
        # compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'InfoNCELoss':
        """Create loss from configuration dictionary"""
        return cls(temperature=config.get('temperature', 0.07)) 