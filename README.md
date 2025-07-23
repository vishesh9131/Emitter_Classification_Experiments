# Emitter_Classification_Experiments

Repository for Experiments conducted for Emitter Classification Project

## Jupyter Notebooks Overview

### Harshwardhan/Embedding/Embedding.ipynb

- **Purpose**: Emitter classification using Siamese Triplet Loss with Artificial Neural Networks
- **Conclusion**: 98% Accurate
- **Key Features**:

  - Loads and preprocesses radar emitter data from CSV files (CombinedDeinterleaved.csv)
  - Implements EmitterEncoder model with embedding layer
  - Uses features: Pulse Width (PW), Azimuth, Elevation, Power, and Frequency
  - Trains model with triplet loss for learning discriminative embeddings
  - Saves trained model weights for inference
  - Includes data normalization and train/test splitting

### Harshwardhan/EDA/LabelChecking.ipynb

- **Purpose**: To check whether we should consider same label from different sets as one
- **Key Features**:
  - Statistical comparison across 3 different datasets
  - Generates distribution plots (histograms, box plots, violin plots)
  - Calculates descriptive statistics and variability metrics
  - Color-coded visualizations for easy comparison
  - Helps identify data quality and consistency issues
- **Conclusion**: Since the distribution varies across sets, we can consider them to be different. Ex - Label F from set 3 would be a different label than Label F from set 5

### Harshwardhan/Generating data/VAEs.ipynb

- **Purpose**: Synthetic data generation using Variational Autoencoders (VAEs)
- **Key Features**:
  - Implements VAE architecture for generating synthetic radar emitter data
  - Uses 4-dimensional input features
  - Includes hyperparameter optimization with Optuna
  - Generates new samples from learned latent space
  - Compares original vs reconstructed data quality
  - Configurable sample generation (N=2000+ samples)

## Data Files

- **CombinedDeinterleaved.csv**: Main dataset containing radar emitter parameters
- **trained_model_full.pth**: Saved PyTorch model weights from embedding training

## Usage

Each notebook can be run independently. Ensure CSV data files are in the correct paths before
