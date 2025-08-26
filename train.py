#!/usr/bin/env python3
"""
Main training script for Emitter Classification
Supports multiple models and loss functions with argparse configuration
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_parser, get_config
from models import EmitterEncoder, FTTransformer, DualEncoder
from losses import TripletLoss, NTXentLoss, SupConLoss, InfoNCELoss
from data import TripletPDWDataset, PairPDWDataset, PDWDataset, load_data
from utils import setup_training, save_model, load_model, evaluate_model, set_seed, generate_all_visualizations

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

def get_loss(loss_config: Dict[str, Any]) -> nn.Module:
    """Get loss function based on configuration"""
    loss_type = loss_config['type']
    
    if loss_type == 'triplet_margin':
        return TripletLoss.from_config(loss_config)
    elif loss_type == 'nt_xent':
        return NTXentLoss.from_config(loss_config)
    elif loss_type == 'supervised_contrastive':
        return SupConLoss.from_config(loss_config)
    elif loss_type == 'info_nce':
        return InfoNCELoss.from_config(loss_config)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def get_dataset(dataset_type: str, features: np.ndarray, labels: np.ndarray):
    """Get dataset based on type"""
    if dataset_type == 'triplet':
        return TripletPDWDataset(features, labels)
    elif dataset_type == 'pair':
        return PairPDWDataset(features, labels)
    elif dataset_type == 'basic':
        return PDWDataset(features, labels)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                loss_type: str) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if loss_type == 'triplet_margin':
            anchor, positive, negative = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
        elif loss_type == 'nt_xent':
            anchor, positive = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            
            loss = criterion(anchor_emb, positive_emb)
            
        elif loss_type == 'supervised_contrastive':
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            
            embeddings = model(features)
            loss = criterion(embeddings, labels)
            
        elif loss_type == 'info_nce':
            anchor, positive = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            
            loss = criterion(anchor_emb, positive_emb)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def main():
    """Main training function"""
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # get configuration
    config = get_config(args)
    
    # set random seed
    set_seed(config['system']['seed'])
    
    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['system']['gpus'] > 0 else 'cpu')
    print(f"Using device: {device}")
    
    # create output directory
    os.makedirs(config['system']['output_dir'], exist_ok=True)
    
    # load data
    print("Loading data...")
    X_train, y_train, X_test, y_test, label_mapping, scaler = load_data(
        config['data']['train_path'],
        config['data']['test_path'],
        config['data']['features'],
        config['data']['label_col']
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_mapping)}")
    
    # determine dataset type based on loss
    loss_type = config['loss']['type']
    if loss_type == 'triplet_margin':
        dataset_type = 'triplet'
    elif loss_type in ['nt_xent', 'info_nce']:
        dataset_type = 'pair'
    elif loss_type == 'supervised_contrastive':
        dataset_type = 'basic'
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # create dataset and dataloader
    train_dataset = get_dataset(dataset_type, X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=True
    )
    
    # create model
    print("Creating model...")
    model = get_model(config['model'], X_train.shape[1])
    model = model.to(device)
    
    # create loss function
    print("Creating loss function...")
    criterion = get_loss(config['loss'])
    criterion = criterion.to(device)
    
    # setup training
    print("Setting up training...")
    optimizer, scheduler = setup_training(model, config['training'], device)
    
    # training loop
    print("Starting training...")
    best_accuracy = 0.0
    training_history = []
    
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
        
        # train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, loss_type
        )
        
        # update scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # evaluate periodically
        if (epoch + 1) % config['training']['eval_every'] == 0:
            metrics = evaluate_model(model, X_test, y_test, device)
            accuracy = metrics['clustering_accuracy']
            
            print(f"Epoch {epoch+1:3d}/{config['training']['epochs']:3d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Accuracy: {accuracy:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                save_path = os.path.join(config['system']['output_dir'], 'best_model.pth')
                save_model(model, config, save_path, metrics)
        else:
            print(f"Epoch {epoch+1:3d}/{config['training']['epochs']:3d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # save checkpoint periodically
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['system']['output_dir'], 
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            save_model(model, config, checkpoint_path)
        
        # record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'time': epoch_time
        })
    
    # final evaluation
    print("Final evaluation...")
    final_metrics = evaluate_model(model, X_test, y_test, device)
    print(f"Final accuracy: {final_metrics['clustering_accuracy']:.4f}")
    
    # get embeddings for visualization (use subset for faster processing)
    if not config['visualization']['skip_visualization']:
        print("Generating embeddings for visualization...")
        model.eval()
        
        # Use subset of test data for visualization
        max_samples = config['visualization']['max_test_samples']
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
            training_history=training_history,
            input_features=config['data']['features'],
            config=config,
            output_dir=config['system']['output_dir']
        )
    else:
        print("Skipping visualization generation...")
    
    # save final model
    final_model_path = os.path.join(config['system']['output_dir'], 'final_model.pth')
    save_model(model, config, final_model_path, final_metrics)
    
    # save training history
    history_path = os.path.join(config['system']['output_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # save configuration
    config_path = os.path.join(config['system']['output_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training complete! Results saved to {config['system']['output_dir']}")
    print(f"Visualizations saved to {os.path.join(config['system']['output_dir'], 'plots')}")

if __name__ == "__main__":
    main() 