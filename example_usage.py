#!/usr/bin/env python3
"""
Example usage script for Emitter Classification Experiments
Demonstrates different training configurations and model combinations
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the description"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Command completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run example training configurations"""
    
    # make sure we're in the right directory
    if not os.path.exists('train.py'):
        print("Error: train.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    print("Emitter Classification Experiments - Example Usage")
    print("This script demonstrates different training configurations")
    
    # Example 1: Basic training with default settings
    print("\nExample 1: Basic training with default settings")
    print("This will train an emitter encoder with triplet loss")
    cmd1 = [
        'python', 'train.py',
        '--output_dir', 'results/example1_default'
    ]
    run_command(cmd1, "Basic training with default settings")
    
    # Example 2: Fast training with transformer
    print("\nExample 2: Fast training with FT-Transformer and NT-Xent loss")
    cmd2 = [
        'python', 'train.py',
        '--model', 'ft_transformer',
        '--loss', 'nt_xent',
        '--config', 'fast',
        '--embed_dim', '64',
        '--output_dir', 'results/example2_transformer'
    ]
    run_command(cmd2, "Fast training with FT-Transformer")
    
    # Example 3: Dual encoder with supervised contrastive loss
    print("\nExample 3: Dual encoder with supervised contrastive loss")
    cmd3 = [
        'python', 'train.py',
        '--model', 'dual_encoder',
        '--loss', 'supcon',
        '--embed_dim', '32',
        '--batch_size', '128',
        '--epochs', '75',
        '--output_dir', 'results/example3_dual_encoder'
    ]
    run_command(cmd3, "Dual encoder with supervised contrastive loss")
    
    # Example 4: Custom configuration with different parameters
    print("\nExample 4: Custom configuration with different parameters")
    cmd4 = [
        'python', 'train.py',
        '--model', 'emitter_encoder',
        '--loss', 'triplet',
        '--embed_dim', '16',
        '--batch_size', '256',
        '--epochs', '50',
        '--lr', '2e-3',
        '--margin', '0.5',
        '--output_dir', 'results/example4_custom'
    ]
    run_command(cmd4, "Custom configuration with different parameters")
    
    # Example 5: Evaluation of a trained model
    print("\nExample 5: Evaluating a trained model")
    print("Note: This assumes one of the previous examples completed successfully")
    
    # check if we have a best model to evaluate
    model_paths = [
        'results/example1_default/best_model.pth',
        'results/example2_transformer/best_model.pth',
        'results/example3_dual_encoder/best_model.pth',
        'results/example4_custom/best_model.pth'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            cmd5 = [
                'python', 'evaluate.py',
                '--model_path', model_path,
                '--output_dir', f'evaluation_results/{os.path.basename(os.path.dirname(model_path))}'
            ]
            run_command(cmd5, f"Evaluating {model_path}")
            break
    else:
        print("No trained models found for evaluation. Please run training examples first.")
    
    print("\n" + "="*60)
    print("Example usage completed!")
    print("Check the results/ directory for training outputs")
    print("Check the evaluation_results/ directory for evaluation outputs")
    print("="*60)

if __name__ == "__main__":
    main() 