#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import os
import torch
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import get_parser, get_config
        print("✓ config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from models import EmitterEncoder, FTTransformer, DualEncoder
        print("✓ models module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from losses import TripletLoss, NTXentLoss, SupConLoss, InfoNCELoss
        print("✓ losses module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import losses: {e}")
        return False
    
    try:
        from data import TripletPDWDataset, PairPDWDataset, PDWDataset, load_data
        print("✓ data module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data: {e}")
        return False
    
    try:
        from utils import setup_training, save_model, load_model, evaluate_model, set_seed
        print("✓ utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    return True

def test_models():
    """Test model creation and forward pass"""
    print("\nTesting models...")
    
    input_dim = 5
    embed_dim = 32
    batch_size = 4
    
    # test data
    x = torch.randn(batch_size, input_dim)
    
    try:
        # test emitter encoder
        model = EmitterEncoder(input_dim, embed_dim)
        output = model(x)
        assert output.shape == (batch_size, embed_dim)
        print("✓ EmitterEncoder works correctly")
    except Exception as e:
        print(f"✗ EmitterEncoder failed: {e}")
        return False
    
    try:
        # test ft transformer
        model = FTTransformer(input_dim, embed_dim)
        output = model(x)
        assert output.shape == (batch_size, embed_dim)
        print("✓ FTTransformer works correctly")
    except Exception as e:
        print(f"✗ FTTransformer failed: {e}")
        return False
    
    try:
        # test dual encoder
        model = DualEncoder(input_dim, embed_dim)
        output = model(x)
        assert output.shape == (batch_size, embed_dim)
        print("✓ DualEncoder works correctly")
    except Exception as e:
        print(f"✗ DualEncoder failed: {e}")
        return False
    
    return True

def test_losses():
    """Test loss functions"""
    print("\nTesting loss functions...")
    
    batch_size = 4
    embed_dim = 32
    
    # test data
    anchor = torch.randn(batch_size, embed_dim)
    positive = torch.randn(batch_size, embed_dim)
    negative = torch.randn(batch_size, embed_dim)
    labels = torch.randint(0, 3, (batch_size,))
    
    try:
        # test triplet loss
        loss_fn = TripletLoss(margin=1.0)
        loss = loss_fn(anchor, positive, negative)
        assert loss.item() >= 0
        print("✓ TripletLoss works correctly")
    except Exception as e:
        print(f"✗ TripletLoss failed: {e}")
        return False
    
    try:
        # test nt-xent loss
        loss_fn = NTXentLoss(temperature=0.1)
        loss = loss_fn(anchor, positive)
        assert loss.item() >= 0
        print("✓ NTXentLoss works correctly")
    except Exception as e:
        print(f"✗ NTXentLoss failed: {e}")
        return False
    
    try:
        # test supcon loss
        loss_fn = SupConLoss(temperature=0.07)
        loss = loss_fn(anchor, labels)
        assert loss.item() >= 0
        print("✓ SupConLoss works correctly")
    except Exception as e:
        print(f"✗ SupConLoss failed: {e}")
        return False
    
    try:
        # test infonce loss
        loss_fn = InfoNCELoss(temperature=0.07)
        loss = loss_fn(anchor, positive)
        assert loss.item() >= 0
        print("✓ InfoNCELoss works correctly")
    except Exception as e:
        print(f"✗ InfoNCELoss failed: {e}")
        return False
    
    return True

def test_datasets():
    """Test dataset creation"""
    print("\nTesting datasets...")
    
    # test data
    features = np.random.randn(100, 5).astype(np.float32)
    labels = np.random.randint(0, 3, (100,))
    
    try:
        # test basic dataset
        dataset = PDWDataset(features, labels)
        assert len(dataset) == 100
        x, y = dataset[0]
        assert x.shape == (5,)
        print("✓ PDWDataset works correctly")
    except Exception as e:
        print(f"✗ PDWDataset failed: {e}")
        return False
    
    try:
        # test triplet dataset
        dataset = TripletPDWDataset(features, labels)
        assert len(dataset) == 100
        anchor, positive, negative = dataset[0]
        assert anchor.shape == (5,)
        print("✓ TripletPDWDataset works correctly")
    except Exception as e:
        print(f"✗ TripletPDWDataset failed: {e}")
        return False
    
    try:
        # test pair dataset
        dataset = PairPDWDataset(features, labels)
        assert len(dataset) == 100
        anchor, positive = dataset[0]
        assert anchor.shape == (5,)
        print("✓ PairPDWDataset works correctly")
    except Exception as e:
        print(f"✗ PairPDWDataset failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration system"""
    print("\nTesting configuration system...")
    
    try:
        from config import get_parser, get_config
        
        # test parser creation
        parser = get_parser()
        assert parser is not None
        print("✓ Argument parser created successfully")
        
        # test with default arguments
        args = parser.parse_args([])
        config = get_config(args)
        assert 'model' in config
        assert 'loss' in config
        assert 'training' in config
        print("✓ Configuration system works correctly")
        
    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Emitter Classification Experiments - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_models,
        test_losses,
        test_datasets,
        test_config
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Installation is working correctly.")
        print("You can now run training with: python train.py")
    else:
        print("✗ Some tests failed. Please check your installation.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 