#!/usr/bin/env python3
"""
Dual Encoder Model - Two separate encoders for query and key
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

class DualEncoder(nn.Module):
    """
    Dual encoder model with separate query and key encoders
    """
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int,
                 layers: List[int] = [128, 64, 32],
                 dropout: float = 0.2,
                 symmetric: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.layers = layers
        self.dropout = dropout
        self.symmetric = symmetric
        
        # build query encoder
        self.query_encoder = self._build_encoder(input_dim, embed_dim, layers, dropout)
        
        # build key encoder (same as query if symmetric)
        if symmetric:
            self.key_encoder = self.query_encoder
        else:
            self.key_encoder = self._build_encoder(input_dim, embed_dim, layers, dropout)
    
    def _build_encoder(self, input_dim: int, embed_dim: int, layers: List[int], dropout: float) -> nn.Module:
        """Build encoder network"""
        modules = []
        prev_dim = input_dim
        
        # hidden layers
        for layer_dim in layers:
            modules.append(nn.Linear(prev_dim, layer_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        
        # output layer
        modules.append(nn.Linear(prev_dim, embed_dim))
        
        return nn.Sequential(*modules)
    
    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        """Encode query with L2 normalization"""
        return F.normalize(self.query_encoder(x), p=2, dim=1)
    
    def encode_key(self, x: torch.Tensor) -> torch.Tensor:
        """Encode key with L2 normalization"""
        return F.normalize(self.key_encoder(x), p=2, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - same as encode_query for compatibility"""
        return self.encode_query(x)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> 'DualEncoder':
        """Create model from configuration dictionary"""
        return cls(
            input_dim=input_dim,
            embed_dim=config.get('embed_dim', 32),
            layers=config.get('layers', [128, 64, 32]),
            dropout=config.get('dropout', 0.2),
            symmetric=config.get('symmetric', True)
        ) 