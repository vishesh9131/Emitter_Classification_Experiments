#!/usr/bin/env python3
"""
Debug script to test FT-Transformer device placement
"""

import torch
import torch.nn as nn
from models.ft_transformer import FTTransformer

def test_device_placement():
    """Test device placement for FT-Transformer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # create model
    model = FTTransformer(input_dim=5, embed_dim=128)
    print(f"Model created on device: {next(model.parameters()).device}")
    
    # move to device
    model = model.to(device)
    print(f"Model moved to device: {next(model.parameters()).device}")
    
    # check all components
    print(f"Tokenizer feature_embeddings device: {model.tokenizer.feature_embeddings.weight.device}")
    print(f"Tokenizer cls_token device: {model.tokenizer.cls_token.device}")
    print(f"Pos embedding device: {model.pos_embedding.device}")
    print(f"Transformer device: {next(model.transformer.parameters()).device}")
    print(f"Output proj device: {next(model.output_proj.parameters()).device}")
    
    # create test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 5).to(device)
    print(f"Input tensor device: {input_tensor.device}")
    
    # test forward pass
    try:
        with torch.no_grad():
            output = model(input_tensor)
        print(f"Forward pass successful! Output shape: {output.shape}")
        print(f"Output device: {output.device}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_device_placement() 