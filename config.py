#!/usr/bin/env python3
"""
Configuration file for Emitter Classification Experiments
Contains all model variants, loss functions, and training parameters
"""

import argparse
from typing import Dict, Any, List

# Model configurations
MODEL_CONFIGS = {
    'emitter_encoder': {
        'type': 'mlp',
        'layers': [64, 64, 64],  # hidden layers
        'dropout': 0.3,
        'residual': True
    },
    'ft_transformer': {
        'type': 'transformer',
        'embed_dim': 128,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1
    },
    'dual_encoder': {
        'type': 'dual_mlp',
        'layers': [128, 64, 32],
        'dropout': 0.2,
        'symmetric': True
    }
}

# Loss function configurations
LOSS_CONFIGS = {
    'triplet': {
        'margin': 1.0,
        'type': 'triplet_margin'
    },
    'nt_xent': {
        'temperature': 0.1,
        'type': 'nt_xent'
    },
    'supcon': {
        'temperature': 0.07,
        'type': 'supervised_contrastive'
    },
    'infonce': {
        'temperature': 0.07,
        'type': 'info_nce'
    }
}

# Training configurations
TRAIN_CONFIGS = {
    'default': {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'warmup_epochs': 2,
        'eval_every': 10,
        'save_every': 20
    },
    'fast': {
        'batch_size': 128,
        'epochs': 50,
        'learning_rate': 3e-3,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'warmup_epochs': 1,
        'eval_every': 5,
        'save_every': 10
    },
    'thorough': {
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 5e-4,
        'weight_decay': 1e-5,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'eval_every': 20,
        'save_every': 50
    }
}

def get_parser() -> argparse.ArgumentParser:
    """Create argument parser with all available options"""
    parser = argparse.ArgumentParser(
        description='Emitter Classification Training with Multiple Models and Loss Functions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        choices=list(MODEL_CONFIGS.keys()),
        default='emitter_encoder',
        help='Model architecture to use'
    )
    
    # Loss function selection
    parser.add_argument(
        '--loss', 
        type=str, 
        choices=list(LOSS_CONFIGS.keys()),
        default='triplet',
        help='Loss function to use'
    )
    
    # Training configuration
    parser.add_argument(
        '--config', 
        type=str, 
        choices=list(TRAIN_CONFIGS.keys()),
        default='default',
        help='Training configuration preset'
    )
    
    # Data paths
    parser.add_argument(
        '--train_data', 
        type=str, 
        default='dataset/',
        help='Path to training dataset directory'
    )
    
    parser.add_argument(
        '--test_data', 
        type=str, 
        default='raw_dt/rawdataset/',
        help='Path to test dataset directory'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results/',
        help='Output directory for results and models'
    )
    
    # Model specific parameters
    parser.add_argument(
        '--embed_dim', 
        type=int, 
        default=32,
        help='Embedding dimension for the model'
    )
    
    # Loss specific parameters
    parser.add_argument(
        '--margin', 
        type=float, 
        default=1.0,
        help='Margin for triplet loss'
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.1,
        help='Temperature for contrastive losses'
    )
    
    # Training parameters (can override config preset)
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=None,
        help='Batch size (overrides config preset)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of epochs (overrides config preset)'
    )
    
    parser.add_argument(
        '--lr', 
        type=float, 
        default=None,
        help='Learning rate (overrides config preset)'
    )
    
    # System parameters
    parser.add_argument(
        '--gpus', 
        type=int, 
        default=1,
        help='Number of GPUs to use'
    )
    
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed'
    )
    
    # Feature selection
    parser.add_argument(
        '--features', 
        type=str, 
        nargs='+',
        default=['PW(µs)', 'Azimuth(º)', 'Elevation(º)', 'Power(dBm)', 'Freq(MHz)'],
        help='Features to use for training'
    )
    
    # Evaluation
    parser.add_argument(
        '--eval_only', 
        action='store_true',
        help='Only evaluate, do not train'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None,
        help='Path to pretrained model for evaluation'
    )
    
    # Visualization parameters
    parser.add_argument(
        '--max_test_samples', 
        type=int, 
        default=10000,
        help='Maximum number of test samples to use for visualization (default: 10000)'
    )
    
    parser.add_argument(
        '--skip_visualization', 
        action='store_true',
        help='Skip visualization generation to save time'
    )
    
    return parser

def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Get complete configuration dictionary from arguments"""
    config = {
        'model': MODEL_CONFIGS[args.model].copy(),
        'loss': LOSS_CONFIGS[args.loss].copy(),
        'training': TRAIN_CONFIGS[args.config].copy(),
        'data': {
            'train_path': args.train_data,
            'test_path': args.test_data,
            'features': args.features,
            'label_col': 'Name'
        },
        'system': {
            'gpus': args.gpus,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'output_dir': args.output_dir
        },
        'evaluation': {
            'eval_only': args.eval_only,
            'model_path': args.model_path
        },
        'visualization': {
            'max_test_samples': args.max_test_samples,
            'skip_visualization': args.skip_visualization
        }
    }
    
    # Override with command line arguments
    if args.embed_dim is not None:
        config['model']['embed_dim'] = args.embed_dim
    
    if args.margin is not None:
        config['loss']['margin'] = args.margin
    
    if args.temperature is not None:
        config['loss']['temperature'] = args.temperature
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    return config 