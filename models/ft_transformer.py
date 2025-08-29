#!/usr/bin/env python3
"""
FT-Transformer Model - Feature Tokenizer + Transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class FeatureTokenizer(nn.Module):
    """Feature tokenizer that converts numerical features to embeddings"""
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.feature_embeddings = nn.Embedding(num_features, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_features)
        batch_size = x.shape[0]
        
        # create feature indices on the same device as input
        feature_indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # get embeddings for each feature
        embeddings = self.feature_embeddings(feature_indices)  # (batch_size, num_features, embed_dim)
        
        # add cls token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        return embeddings

class FTTransformer(nn.Module):
    """
    FT-Transformer model for tabular data
    """
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # feature tokenizer
        self.tokenizer = FeatureTokenizer(input_dim, embed_dim)
        
        # positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim + 1, embed_dim))
        
        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with L2 normalization"""
        # tokenize features
        tokens = self.tokenizer(x)  # (batch_size, num_features + 1, embed_dim)
        
        # add positional embeddings
        tokens = tokens + self.pos_embedding
        
        # pass through transformer
        transformed = self.transformer(tokens)
        
        # use cls token for final representation
        cls_output = transformed[:, 0]  # (batch_size, embed_dim)
        
        # project to final embedding
        output = self.output_proj(cls_output)
        
        return F.normalize(output, p=2, dim=1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], input_dim: int) -> 'FTTransformer':
        """Create model from configuration dictionary"""
        return cls(
            input_dim=input_dim,
            embed_dim=config.get('embed_dim', 128),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1)
        ) 