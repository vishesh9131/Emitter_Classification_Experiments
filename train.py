#!/usr/bin/env python3
"""
Multi-GPU training script for Emitter Classification
Uses all 4 GPUs with zero CPU usage for data processing
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import from existing modules
from models import EmitterEncoder, FTTransformer
from losses import TripletLoss
from data import TripletPDWDataset, load_data_with_files, preprocess_data
from utils import (setup_training, save_model, set_seed, 
                  gpu_evaluate_model, generate_all_visualizations)

# ──────────────────────────────────────────────────────────────
# Global configuration
EMBEDDING_DIMS_TO_TEST = [32]
MARGIN = 1.0
BASE_LR = 1e-3
BATCH_SIZE = 128  # Per GPU batch size
EPOCHS = 100
CLUSTER_EVERY = 10
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# GPU optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# ──────────────────────────────────────────────────────────────
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision and optimized GPU transfers"""
    model.train()
    running_loss = 0.0
    
    for a, p, n in dataloader:
        # Optimized GPU transfer with non_blocking=True
        a = a.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        n = n.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use mixed precision training
        if scaler is not None:
            with autocast():
                loss = criterion(model(a), model(p), model(n))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# ──────────────────────────────────────────────────────────────
def setup_gpu_dataset(features, labels, device):
    """Create GPU-optimized dataset that keeps everything on GPU"""
    class GPUTripletDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels, device):
            # Move everything to GPU immediately
            self.features = torch.tensor(features, dtype=torch.float32, device=device)
            self.labels = torch.tensor(labels, dtype=torch.long, device=device)
            self.device = device
            
            # Create label to index mapping on GPU
            unique_labels = torch.unique(self.labels)
            self.label_to_indices = {}
            
            for label in unique_labels:
                mask = (self.labels == label)
                indices = torch.where(mask)[0]
                self.label_to_indices[label.item()] = indices
            
            # Pre-compute a subset of triplets on GPU for efficiency
            self.max_triplets = min(50000, len(features) * 2)  # Limit memory usage
            self._precompute_triplets()
            
        def _precompute_triplets(self):
            """Pre-compute triplets on GPU"""
            anchors = []
            positives = []
            negatives = []
            
            unique_labels = list(self.label_to_indices.keys())
            
            # Sample triplets efficiently
            for _ in range(self.max_triplets):
                # Random anchor
                anchor_idx = torch.randint(0, len(self.features), (1,), device=self.device).item()
                anchor_label = self.labels[anchor_idx].item()
                
                # Random positive from same class
                positive_indices = self.label_to_indices[anchor_label]
                if len(positive_indices) > 1:
                    pos_idx = torch.randint(0, len(positive_indices), (1,), device=self.device).item()
                    positive_idx = positive_indices[pos_idx].item()
                    
                    # Random negative from different class
                    negative_labels = [l for l in unique_labels if l != anchor_label]
                    if negative_labels:
                        neg_label = negative_labels[torch.randint(0, len(negative_labels), (1,), device=self.device).item()]
                        negative_indices = self.label_to_indices[neg_label]
                        neg_idx = torch.randint(0, len(negative_indices), (1,), device=self.device).item()
                        negative_idx = negative_indices[neg_idx].item()
                        
                        anchors.append(anchor_idx)
                        positives.append(positive_idx)
                        negatives.append(negative_idx)
            
            # Convert to tensors on GPU
            self.triplet_indices = torch.stack([
                torch.tensor(anchors, device=self.device),
                torch.tensor(positives, device=self.device),
                torch.tensor(negatives, device=self.device)
            ], dim=1)
        
        def __len__(self):
            return len(self.triplet_indices)
        
        def __getitem__(self, idx):
            anchor_idx, positive_idx, negative_idx = self.triplet_indices[idx]
            return (self.features[anchor_idx], 
                    self.features[positive_idx], 
                    self.features[negative_idx])
    
    return GPUTripletDataset(features, labels, device)

# ──────────────────────────────────────────────────────────────
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-GPU training for emitter classification')
    parser.add_argument('--model', type=str, default='emitter_encoder', 
                       choices=['emitter_encoder', 'ft_transformer'])
    parser.add_argument('--train_files', nargs='+', 
                       default=['dataset/set1.xls', 'dataset/set2.xls'])
    parser.add_argument('--test_files', nargs='+', 
                       default=['raw_dt/rawdataset/s6cleaned.csv'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--output_dir', type=str, default=RESULT_DIR)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"Using model: {args.model}")
    print(f"Training files: {args.train_files}")
    print(f"Test files: {args.test_files}")
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    features = ['PW(µs)', 'Azimuth(º)', 'Elevation(º)', 'Power(dBm)', 'Freq(MHz)']
    x_train, y_train, x_test, y_test, label_mapping, scaler = load_data_with_files(
        args.train_files, args.test_files, features
    )
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(label_mapping)}")
    
    # Create GPU-optimized dataset
    train_dataset = setup_gpu_dataset(x_train, y_train, device)
    
    # Create dataloader with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * num_gpus,  # Scale batch size by number of GPUs
        shuffle=True,
        num_workers=0,  # No CPU workers needed since data is on GPU
        drop_last=True
    )
    
    for dim in EMBEDDING_DIMS_TO_TEST:
        print(f"\n===== Training {args.model} with embedding dimension {dim} =====")
        
        # Create model
        if args.model == 'emitter_encoder':
            model = EmitterEncoder(x_train.shape[1], dim).to(device)
        elif args.model == 'ft_transformer':
            model = FTTransformer(x_train.shape[1], dim).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        # Wrap model with DataParallel for multi-GPU
        if num_gpus > 1:
            model = DataParallel(model)
            print(f"Using DataParallel on {num_gpus} GPUs")
        
        # Create configuration
        config = {
            'model': {
                'type': args.model,
                'embed_dim': dim,
                'input_dim': x_train.shape[1]
            },
            'loss': {
                'type': 'triplet_margin',
                'margin': MARGIN
            },
            'training': {
                'batch_size': args.batch_size * num_gpus,  # Total batch size across all GPUs
                'learning_rate': BASE_LR * num_gpus,  # Scale learning rate
                'epochs': args.epochs,
                'weight_decay': 1e-4,
                'scheduler': 'cosine'
            },
            'data': {
                'train_files': args.train_files,
                'test_files': args.test_files,
                'features': features
            }
        }
        
        # Setup training
        criterion = TripletLoss(margin=MARGIN).to(device)
        optimizer, scheduler = setup_training(model, config['training'], device)
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # Training history tracking
        training_history = []
        accuracies = []
        eval_epochs = []
        
        print(f"Starting training for {args.epochs} epochs on {num_gpus} GPUs...")
        for epoch in range(args.epochs):
            # Train one epoch
            avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Record training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Print progress every epoch
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss {avg_loss:.4f}")
            
            # Evaluate periodically
            if (epoch + 1) % CLUSTER_EVERY == 0:
                # Convert test data to GPU tensors for evaluation
                x_test_tensor = torch.tensor(x_test, device=device)
                y_test_tensor = torch.tensor(y_test, device=device)
                
                eval_results = gpu_evaluate_model(model, x_test_tensor, y_test_tensor, device=device)
                acc = eval_results.get('clustering_accuracy', 0.0)
                accuracies.append(acc * 100)
                eval_epochs.append(epoch + 1)
                print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}  "
                      f"test-clust-acc {acc*100:5.2f}%")
        
        # Final evaluation
        print("Performing final evaluation...")
        x_test_tensor = torch.tensor(x_test, device=device)
        y_test_tensor = torch.tensor(y_test, device=device)
        
        eval_results = gpu_evaluate_model(model, x_test_tensor, y_test_tensor, device=device)
        final_acc = eval_results.get('clustering_accuracy', 0.0)
        
        # Get embeddings for visualization
        model.eval()
        with torch.no_grad():
            final_emb = model(x_test_tensor).cpu().numpy()
        
        # Get cluster labels
        if 'cluster_labels' in eval_results:
            final_clusters = eval_results['cluster_labels'].cpu().numpy()
        else:
            # Fallback to CPU clustering for visualization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=len(np.unique(y_test)), n_init=10, random_state=42)
            final_clusters = kmeans.fit_predict(final_emb)
        
        # Save results
        metrics = {
            "model": args.model,
            "embedding_dim": dim,
            "test_acc": final_acc,
            "final_loss": avg_loss,
            "best_accuracy": max(accuracies) if accuracies else final_acc * 100,
            "final_accuracy": final_acc * 100
        }
        
        # Save model
        model_save_path = f"{args.output_dir}/model_{args.model}_dim_{dim}.pth"
        save_model(model.module if hasattr(model, 'module') else model, config, model_save_path, metrics)
        
        # Save results JSON
        with open(f"{args.output_dir}/result_{args.model}_dim_{dim}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[Done] {args.model} dim={dim}  acc={final_acc*100:.2f}%")
        
        # Generate visualizations
        try:
            print("Generating visualizations...")
            generate_all_visualizations(
                model=model.module if hasattr(model, 'module') else model,
                embeddings=final_emb,
                true_labels=y_test,
                cluster_labels=final_clusters,
                training_history=training_history,
                input_features=config['data']['features'],
                config=config,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nTraining complete!")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main() 