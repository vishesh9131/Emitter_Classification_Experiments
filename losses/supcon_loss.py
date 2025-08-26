#!/usr/bin/env python3
"""
Supervised Contrastive Loss implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss
        
        Args:
            features: Feature embeddings (batch_size, embed_dim)
            labels: Class labels (batch_size,)
            
        Returns:
            Supervised contrastive loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # normalize features
        features = F.normalize(features, dim=1)
        
        # compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # remove self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # compute logits
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SupConLoss':
        """Create loss from configuration dictionary"""
        return cls(temperature=config.get('temperature', 0.07)) 