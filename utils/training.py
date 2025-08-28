#!/usr/bin/env python3
"""
Training utilities for Emitter Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import os
import json
from typing import Dict, Any, Tuple, Optional
import random
import numpy as np

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_training(model: nn.Module, 
                  config: Dict[str, Any],
                  device: torch.device) -> Tuple[optim.Optimizer, Any]:
    """
    Setup optimizer and scheduler for training
    
    Args:
        model: Model to train
        config: Training configuration
        device: Device to train on
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # setup scheduler
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config['epochs'],
            eta_min=1e-6
        )
    elif scheduler_type == 'warmup_cosine':
        # warmup cosine scheduler
        warmup_epochs = config.get('warmup_epochs', 2)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (config['epochs'] - warmup_epochs)))
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None
    
    return optimizer, scheduler

def save_model(model: nn.Module, 
               config: Dict[str, Any], 
               save_path: str,
               metrics: Optional[Dict[str, float]] = None):
    """
    Save model and configuration
    
    Args:
        model: Model to save
        config: Configuration dictionary
        save_path: Path to save the model
        metrics: Optional metrics to save
    """
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics or {}
    }, save_path)
    
    print(f"Model saved to {save_path}")

def load_model(model: nn.Module, 
               load_path: str,
               device: torch.device) -> Tuple[nn.Module, Dict[str, Any], Dict[str, float]]:
    """
    Load model and configuration
    
    Args:
        model: Model to load state into
        load_path: Path to load the model from
        device: Device to load on
        
    Returns:
        Tuple of (model, config, metrics)
    """
    # Load with weights_only=False to handle older PyTorch checkpoints
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    # load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # get config and metrics
    config = checkpoint.get('config', {})
    metrics = checkpoint.get('metrics', {})
    
    print(f"Model loaded from {load_path}")
    
    return model, config, metrics 