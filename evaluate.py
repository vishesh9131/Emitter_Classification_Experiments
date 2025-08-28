#!/usr/bin/env python3
"""
Evaluation script for trained Emitter Classification models
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np

# add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_parser, get_config
from models import EmitterEncoder, FTTransformer, DualEncoder
from data import load_data
from utils import load_model, evaluate_model, generate_all_visualizations

def get_model(model_config: Dict[str, Any], input_dim: int) -> nn.Module:
    """Get model based on configuration"""
    model_type = model_config['type']
    
    if model_type == 'mlp':
        return EmitterEncoder.from_config(model_config, input_dim)
    elif model_type == 'transformer':
        return FTTransformer.from_config(model_config, input_dim)
    elif model_type == 'dual_mlp':
        return DualEncoder.from_config(model_config, input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained Emitter Classification model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, default='raw_dt/rawdataset/', help='Path to test data')
    parser.add_argument('--features', type=str, nargs='+', 
                       default=['PW(µs)', 'Azimuth(º)', 'Elevation(º)', 'Power(dBm)', 'Freq(MHz)'],
                       help='Features to use')
    parser.add_argument('--output_dir', type=str, default='evaluation_results/', help='Output directory')
    parser.add_argument('--max_test_samples', type=int, default=10000,
                       help='Maximum number of test samples to use for visualization (default: 10000)')
    parser.add_argument('--skip_visualization', action='store_true',
                       help='Skip visualization generation to save time')
    
    args = parser.parse_args()
    
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # load model and config
    print(f"Loading model from {args.model_path}")
    
    # first, we need to create a dummy model to load the state
    # we'll get the config from the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # load test data to get input dimension
    print("Loading test data...")
    X_train, y_train, X_test, y_test, label_mapping, scaler = load_data(
        'dataset/',  # dummy train path, won't be used
        args.test_data,
        args.features,
        'Name'
    )
    
    # create model
    model = get_model(config['model'], X_test.shape[1])
    
    # load model state
    model, loaded_config, metrics = load_model(model, args.model_path, device)
    
    print(f"Model loaded successfully")
    print(f"Model type: {config['model']['type']}")
    print(f"Embedding dimension: {config['model'].get('embed_dim', 'N/A')}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_mapping)}")
    
    # evaluate model
    print("Evaluating model...")
    evaluation_metrics = evaluate_model(model, X_test, y_test, device)
    
    print("\nEvaluation Results:")
    print(f"Clustering Accuracy: {evaluation_metrics['clustering_accuracy']:.4f}")
    print(f"Number of Clusters: {evaluation_metrics['n_clusters']}")
    print(f"Embedding Shape: {evaluation_metrics['embedding_shape']}")
    
    # get embeddings for visualization (use subset for faster processing)
    if not args.skip_visualization:
        print("Generating embeddings for visualization...")
        model.eval()
        
        # Use subset of test data for visualization
        max_samples = args.max_test_samples
        if len(X_test) > max_samples:
            print(f"Using {max_samples} samples out of {len(X_test)} for visualization...")
            # Randomly sample test data
            np.random.seed(42)
            indices = np.random.choice(len(X_test), max_samples, replace=False)
            X_test_viz = X_test[indices]
            y_test_viz = y_test[indices]
        else:
            X_test_viz = X_test
            y_test_viz = y_test
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_viz, dtype=torch.float32).to(device)
            embeddings = model(X_test_tensor).cpu().numpy()
        
        # perform clustering for visualization
        from sklearn.cluster import KMeans
        n_clusters = len(np.unique(y_test_viz))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # generate visualizations
        print("Generating visualizations...")
        generate_all_visualizations(
            model=model,
            embeddings=embeddings,
            true_labels=y_test_viz,
            cluster_labels=cluster_labels,
            training_history=[],  # Empty for evaluation
            input_features=args.features,
            config=config,
            output_dir=args.output_dir
        )
    else:
        print("Skipping visualization generation...")
    
    # save results
    results = {
        'model_path': args.model_path,
        'test_data_path': args.test_data,
        'features': args.features,
        'evaluation_metrics': evaluation_metrics,
        'model_config': config,
        'label_mapping': label_mapping
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Visualizations saved to {os.path.join(args.output_dir, 'plots')}")

if __name__ == "__main__":
    main() 