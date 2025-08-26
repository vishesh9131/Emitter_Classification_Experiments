#!/usr/bin/env python3
"""
Emitter Encoder Model - MLP based encoder with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

class ResidualBlock(nn.Module):
    """Residual block with dropout and activation"""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class EmitterEncoder(nn.Module):
    """
    Emitter Encoder model with configurable layers and residual connections
    """
    def __init__(self, 
                 input_dim: int, 
                 embed_dim: int,
                 layers: List[int] = [64, 64, 64],
                 dropout: float = 0.3,
                 residual: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.layers = layers
        self.dropout = dropout
        self.residual = residual
        
        # build the network
        modules = []
        prev_dim = input_dim
        
        # input layer
        modules.append(nn.Linear(prev_dim, layers[0]))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        prev_dim = layers[0]
        
        # hidden layers with optional residual connections
        for layer_dim in layers[1:]:
            if residual and layer_dim == prev_dim:
                modules.append(ResidualBlock(layer_dim, dropout))
            else:
                modules.append(nn.Linear(prev_dim, layer_dim))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        
        # output layer
        modules.append(nn.Linear(prev_dim, embed_dim))
        
        self.net = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization"""
        return F.normalize(self.net(x), p=2, dim=1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> 'EmitterEncoder':
        """Create model from configuration dictionary"""
        return cls(
            input_dim=input_dim,
            embed_dim=config.get('embed_dim', 32),
            layers=config.get('layers', [64, 64, 64]),
            dropout=config.get('dropout', 0.3),
            residual=config.get('residual', True)
        ) 