#!/usr/bin/env python3
"""
Visualization utilities for Emitter Classification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Tuple
import torch

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_plots_directory(output_dir: str) -> str:
    """Create plots directory inside output directory"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_training_history(training_history: List[Dict], plots_dir: str, config: Dict[str, Any]):
    """Plot training loss over epochs"""
    epochs = [entry['epoch'] for entry in training_history]
    losses = [entry['train_loss'] for entry in training_history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, 'b-', linewidth=2, alpha=0.8)
    plt.title(f'Training Loss Over Time\nModel: {config["model"]["type"]}, Loss: {config["loss"]["type"]}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'training_loss.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to {plot_path}")

def plot_clustering_results(embeddings: np.ndarray, 
                          true_labels: np.ndarray, 
                          cluster_labels: np.ndarray,
                          plots_dir: str, 
                          config: Dict[str, Any],
                          method: str = 'tsne'):
    """Plot clustering results using dimensionality reduction"""
    
    # Reduce dimensionality for visualization
    if method == 'tsne':
        # Adjust perplexity based on dataset size
        n_samples = embeddings.shape[0]
        if n_samples < 50:
            perplexity = min(30, n_samples - 1)
        elif n_samples < 5000:
            perplexity = 30
        else:
            perplexity = 50  # Larger perplexity for bigger datasets
        
        print(f"Using t-SNE with perplexity={perplexity} for {n_samples} samples...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        title_method = 't-SNE'
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        title_method = 'PCA'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Reducing dimensionality using {method}...")
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: True labels
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=true_labels, cmap='tab20', alpha=0.7, s=20)
    ax1.set_title(f'True Labels ({title_method})\nModel: {config["model"]["type"]}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{title_method} Component 1', fontsize=12)
    ax1.set_ylabel(f'{title_method} Component 2', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for true labels
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('True Labels', fontsize=10)
    
    # Plot 2: Predicted clusters
    scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=cluster_labels, cmap='tab20', alpha=0.7, s=20)
    ax2.set_title(f'Predicted Clusters ({title_method})\nLoss: {config["loss"]["type"]}', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{title_method} Component 1', fontsize=12)
    ax2.set_ylabel(f'{title_method} Component 2', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for predicted clusters
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Predicted Clusters', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, f'clustering_{method}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Clustering visualization saved to {plot_path}")

def plot_confusion_matrix(true_labels: np.ndarray, 
                         cluster_labels: np.ndarray,
                         plots_dir: str, 
                         config: Dict[str, Any]):
    """Plot confusion matrix between true labels and predicted clusters"""
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, cluster_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(len(np.unique(cluster_labels))),
                yticklabels=range(len(np.unique(true_labels))))
    plt.title(f'Confusion Matrix\nModel: {config["model"]["type"]}, Loss: {config["loss"]["type"]}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Clusters', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {plot_path}")

def plot_embedding_distribution(embeddings: np.ndarray, 
                              plots_dir: str, 
                              config: Dict[str, Any]):
    """Plot distribution of embedding values"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot histogram of embedding values
    axes[0].hist(embeddings.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Distribution of Embedding Values', fontweight='bold')
    axes[0].set_xlabel('Embedding Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Plot embedding norms
    norms = np.linalg.norm(embeddings, axis=1)
    axes[1].hist(norms, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribution of Embedding Norms', fontweight='bold')
    axes[1].set_xlabel('L2 Norm')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    # Plot first two dimensions
    axes[2].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
    axes[2].set_title('First Two Embedding Dimensions', fontweight='bold')
    axes[2].set_xlabel('Dimension 1')
    axes[2].set_ylabel('Dimension 2')
    axes[2].grid(True, alpha=0.3)
    
    # Plot embedding statistics
    mean_emb = np.mean(embeddings, axis=0)
    std_emb = np.std(embeddings, axis=0)
    axes[3].bar(range(len(mean_emb)), mean_emb, alpha=0.7, color='orange', 
                yerr=std_emb, capsize=3)
    axes[3].set_title('Mean and Std of Embedding Dimensions', fontweight='bold')
    axes[3].set_xlabel('Dimension')
    axes[3].set_ylabel('Value')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f'Embedding Analysis\nModel: {config["model"]["type"]}, Loss: {config["loss"]["type"]}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'embedding_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Embedding distribution plot saved to {plot_path}")

def plot_feature_importance(model: torch.nn.Module, 
                          input_features: List[str],
                          plots_dir: str, 
                          config: Dict[str, Any]):
    """Plot feature importance based on model weights (for linear models)"""
    
    # This is a simplified version - for more complex models, you'd need more sophisticated methods
    try:
        # Get the first layer weights
        first_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
        
        if first_layer is not None:
            weights = first_layer.weight.data.cpu().numpy()
            feature_importance = np.mean(np.abs(weights), axis=0)
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(input_features)), feature_importance, 
                          color='coral', alpha=0.7, edgecolor='black')
            plt.title(f'Feature Importance (First Layer Weights)\nModel: {config["model"]["type"]}', 
                      fontsize=14, fontweight='bold')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Average Absolute Weight', fontsize=12)
            plt.xticks(range(len(input_features)), input_features, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, feature_importance):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, 'feature_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature importance plot saved to {plot_path}")
            
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

def plot_model_architecture(model: torch.nn.Module, 
                           plots_dir: str, 
                           config: Dict[str, Any]):
    """Plot model architecture summary"""
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create a simple text-based architecture visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Model info text
    model_info = f"""
    Model Architecture Summary
    
    Model Type: {config['model']['type']}
    Embedding Dimension: {config['model'].get('embed_dim', 'N/A')}
    
    Parameters:
    - Total Parameters: {total_params:,}
    - Trainable Parameters: {trainable_params:,}
    
    Configuration:
    - Loss Function: {config['loss']['type']}
    - Batch Size: {config['training']['batch_size']}
    - Learning Rate: {config['training']['learning_rate']}
    - Epochs: {config['training']['epochs']}
    
    Model Structure:
    """
    
    # Add model structure info
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            model_info += f"  - {name}: {module}\n"
    
    ax.text(0.1, 0.9, model_info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.title('Model Architecture Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'model_architecture.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model architecture plot saved to {plot_path}")

def generate_all_visualizations(model: torch.nn.Module,
                               embeddings: np.ndarray,
                               true_labels: np.ndarray,
                               cluster_labels: np.ndarray,
                               training_history: List[Dict],
                               input_features: List[str],
                               config: Dict[str, Any],
                               output_dir: str):
    """Generate all visualizations and save them to plots directory"""
    
    # Create plots directory
    plots_dir = create_plots_directory(output_dir)
    
    print(f"\nGenerating visualizations in {plots_dir}...")
    
    # Generate all plots
    plot_training_history(training_history, plots_dir, config)
    plot_clustering_results(embeddings, true_labels, cluster_labels, plots_dir, config, method='tsne')
    plot_clustering_results(embeddings, true_labels, cluster_labels, plots_dir, config, method='pca')
    plot_confusion_matrix(true_labels, cluster_labels, plots_dir, config)
    plot_embedding_distribution(embeddings, plots_dir, config)
    plot_feature_importance(model, input_features, plots_dir, config)
    plot_model_architecture(model, plots_dir, config)
    
    print(f"\nAll visualizations saved to {plots_dir}")
    
    return plots_dir 