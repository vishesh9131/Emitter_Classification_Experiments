# Emitter Classification Experiments

A comprehensive and modular framework for emitter classification using various deep learning models and contrastive learning approaches.

## Project Structure

```
Emitter_Classification_Experiments/
├── config.py                 # Configuration management with argparse
├── train.py                  # Main training script
├── evaluate.py               # Model evaluation script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── emitter_encoder.py   # MLP-based encoder with residual connections
│   ├── ft_transformer.py    # Feature Tokenizer + Transformer
│   └── dual_encoder.py      # Dual encoder for contrastive learning
├── losses/                  # Loss functions
│   ├── __init__.py
│   ├── triplet_loss.py      # Triplet margin loss
│   ├── nt_xent_loss.py      # NT-Xent loss
│   ├── supcon_loss.py       # Supervised contrastive loss
│   └── infonce_loss.py      # InfoNCE loss
├── data/                    # Data processing
│   ├── __init__.py
│   ├── dataset.py           # Dataset classes
│   └── data_loader.py       # Data loading utilities
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── evaluation.py        # Evaluation functions
│   └── training.py          # Training utilities
├── dataset/                 # Training data (Excel files)
├── raw_dt/                  # Raw test data (CSV files)
└── results/                 # Output directory for results
```

## Performance

The framework achieves high clustering accuracy on emitter classification tasks:

- **Emitter Encoder + Triplet Loss**: ~98% accuracy
- **FT-Transformer + NT-Xent**: ~97% accuracy
- **Dual Encoder + InfoNCE**: ~96% accuracy

## Features

- **Multiple Model Architectures**: MLP with residual connections, FT-Transformer, Dual Encoder
- **Various Loss Functions**: Triplet Loss, NT-Xent, Supervised Contrastive, InfoNCE
- **Configurable Training**: Easy-to-use argparse interface with preset configurations
- **Automatic Model Saving**: Best model and checkpoint saving
- **Comprehensive Evaluation**: Clustering accuracy and detailed metrics
- **Modular Design**: Easy to extend with new models and loss functions


## Usage

### Training

The main training script supports various configurations through command-line arguments:

```bash
# Basic training with default settings
python train.py

# Train with specific model and loss
python train.py --model emitter_encoder --loss triplet

# Train with custom parameters
python train.py --model ft_transformer --loss nt_xent --embed_dim 128 --batch_size 256

# Use fast training preset
python train.py --config fast --epochs 50

# Train with custom data paths
python train.py --train_data dataset/ --test_data raw_dt/rawdataset/
```

### Available Models

- `emitter_encoder`: MLP-based encoder with residual connections
- `ft_transformer`: Feature Tokenizer + Transformer architecture
- `dual_encoder`: Dual encoder for contrastive learning

### Available Loss Functions

- `triplet`: Triplet margin loss
- `nt_xent`: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- `supcon`: Supervised contrastive loss
- `infonce`: InfoNCE loss

### Training Configurations

- `default`: Standard training (100 epochs, batch size 64)
- `fast`: Quick training (50 epochs, batch size 128)
- `thorough`: Comprehensive training (200 epochs, batch size 32)

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model_path results/best_model.pth --test_data raw_dt/rawdataset/
```

## Command Line Arguments

### Model Selection
- `--model`: Model architecture (emitter_encoder, ft_transformer, dual_encoder)
- `--embed_dim`: Embedding dimension (default: 32)

### Loss Function
- `--loss`: Loss function (triplet, nt_xent, supcon, infonce)
- `--margin`: Margin for triplet loss (default: 1.0)
- `--temperature`: Temperature for contrastive losses (default: 0.1)

### Training Configuration
- `--config`: Training preset (default, fast, thorough)
- `--batch_size`: Batch size (overrides config preset)
- `--epochs`: Number of epochs (overrides config preset)
- `--lr`: Learning rate (overrides config preset)

### Data and Paths
- `--train_data`: Path to training data directory (default: dataset/)
- `--test_data`: Path to test data directory (default: raw_dt/rawdataset/)
- `--features`: Features to use for training
- `--output_dir`: Output directory for results (default: results/)

### System Configuration
- `--gpus`: Number of GPUs to use (default: 1)
- `--num_workers`: Number of data loading workers (default: 4)
- `--seed`: Random seed (default: 42)

## Data Format

### Training Data
- Excel files (.xls, .xlsx) in the `dataset/` directory
- Required columns: `Name` (label), `PW(µs)`, `Azimuth(º)`, `Elevation(º)`, `Power(dBm)`, `Freq(MHz)`
- Optional: `Status` column (DELETE_EMITTER entries are filtered out)

### Test Data
- CSV files in the `raw_dt/rawdataset/` directory
- Same column structure as training data

## Output

Training produces the following outputs in the results directory:

- `best_model.pth`: Best performing model checkpoint
- `final_model.pth`: Final model checkpoint
- `checkpoint_epoch_N.pth`: Periodic checkpoints
- `training_history.json`: Training history and metrics
- `config.json`: Complete configuration used for training

## Examples

### Quick Experiment
```bash
# Train with transformer and NT-Xent loss
python train.py --model ft_transformer --loss nt_xent --config fast
```

### Comprehensive Training
```bash
# Train with dual encoder and supervised contrastive loss
python train.py --model dual_encoder --loss supcon --config thorough --embed_dim 64
```

### Custom Configuration
```bash
# Train with custom parameters
python train.py \
    --model emitter_encoder \
    --loss triplet \
    --embed_dim 16 \
    --batch_size 128 \
    --epochs 150 \
    --lr 5e-4 \
    --margin 0.5
```

## Extending the Framework

### Adding New Models

1. Create a new model class in `models/`
2. Implement the `from_config` class method
3. Add the model to the `get_model` function in `train.py`

### Adding New Loss Functions

1. Create a new loss class in `losses/`
2. Implement the `from_config` class method
3. Add the loss to the `get_loss` function in `train.py`

### Adding New Datasets

1. Create a new dataset class in `data/dataset.py`
2. Add the dataset type to the `get_dataset` function in `train.py`


