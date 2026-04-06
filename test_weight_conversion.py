#!/usr/bin/env python3
"""
Test Script: Verify Keras to PyTorch Weight Conversion
=======================================================

This script helps you verify that your weight conversion worked correctly
by comparing model outputs before and after loading weights.
"""

import torch
import numpy as np
import argparse
from meta import Meta


def create_test_input(shape=(1, 2, 64, 272)):
    """
    Create a test input tensor for verification.
    
    Args:
        shape: Input shape (batch, channels, height, width)
        
    Returns:
        Random test tensor
    """
    return torch.randn(shape)


def test_weight_loading(checkpoint_path, config):
    """
    Test if weights load correctly and change model behavior.
    
    Args:
        checkpoint_path: Path to converted weights
        config: Model architecture configuration
    """
    print("="*70)
    print("Weight Loading Verification Test")
    print("="*70)
    
    # Create dummy arguments for Meta model
    class DummyArgs:
        def __init__(self):
            self.meta_lr = 5e-4
            self.update_lr = 1e-4
            self.update_step = 3
            self.scheduler_factor = 0.5
            self.scheduler_patience = 8
            self.scheduler_min_lr = 1e-7
            self.max_grad_norm = 0.75
            self.k_spt = 5
            self.k_qry = 5
            self.batchsz = 8
    
    args = DummyArgs()
    
    print("\n1. Creating model with random initialization...")
    model_random = Meta(args, config)
    
    # Get a few random weight values
    random_weights = [p.clone() for p in list(model_random.parameters())[:3]]
    
    print("\n2. Creating second model and loading pre-trained weights...")
    model_pretrained = Meta(args, config)
    
    # Load pretrained weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        missing, unexpected = model_pretrained.load_state_dict(state_dict, strict=False)
        
        print(f"   ✓ Loaded {len(state_dict)} parameters")
        if missing:
            print(f"   - Missing: {len(missing)} parameters")
        if unexpected:
            print(f"   - Unexpected: {len(unexpected)} parameters")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Get same weights from pretrained model
    pretrained_weights = [p.clone() for p in list(model_pretrained.parameters())[:3]]
    
    print("\n3. Comparing weights...")
    weights_changed = False
    for i, (w_rand, w_pre) in enumerate(zip(random_weights, pretrained_weights)):
        if w_rand.shape == w_pre.shape:
            diff = torch.abs(w_rand - w_pre).mean().item()
            if diff > 1e-6:
                weights_changed = True
                print(f"   Layer {i}: Mean difference = {diff:.6f} ✓ (weights changed)")
            else:
                print(f"   Layer {i}: Mean difference = {diff:.6f} ✗ (weights unchanged!)")
        else:
            print(f"   Layer {i}: Shape mismatch {w_rand.shape} vs {w_pre.shape}")
    
    if not weights_changed:
        print("\n⚠ WARNING: Weights don't appear to have changed!")
        print("   This suggests the loading may have failed.")
        return False
    
    print("\n4. Testing model outputs...")
    model_random.eval()
    model_pretrained.eval()
    
    # Create test input
    test_input = create_test_input()
    
    with torch.no_grad():
        output_random = model_random.net(test_input, vars=None, bn_training=False)
        output_pretrained = model_pretrained.net(test_input, vars=None, bn_training=False)
    
    output_diff = torch.abs(output_random - output_pretrained).mean().item()
    print(f"   Mean output difference: {output_diff:.6f}")
    
    if output_diff > 1e-6:
        print("   ✓ Outputs are different (as expected with different weights)")
    else:
        print("   ✗ Outputs are identical (unexpected!)")
        return False
    
    print("\n" + "="*70)
    print("✓ VERIFICATION SUCCESSFUL!")
    print("="*70)
    print("The pre-trained weights loaded correctly and affect model behavior.")
    print("You can now use these weights for MAML training.")
    print("="*70)
    
    return True


def compare_model_architectures(keras_summary_file=None):
    """
    Helper to compare Keras and PyTorch architectures.
    
    Args:
        keras_summary_file: Text file with Keras model.summary() output
    """
    print("\n" + "="*70)
    print("Architecture Comparison Tips")
    print("="*70)
    
    print("""
To ensure weights transfer correctly, verify that:

1. **Number of layers match**
   - Keras: Count Conv2D and BatchNormalization layers
   - PyTorch: Count conv2d and bn entries in config

2. **Layer parameters match**
   - Kernel sizes (e.g., 3x3, 5x5)
   - Number of filters/channels
   - Activation functions

3. **Layer order matches**
   - First layer should have same input/output dimensions
   - Last layer should have same output shape

**Keras Conv2D example:**
  Conv2D(64, (3, 3), padding='same')  # 64 filters, 3x3 kernel

**PyTorch equivalent:**
  ('conv2d', [64, in_channels, 3, 3, 1, 1])  # [out, in, h, w, stride, padding]

**Keras BatchNormalization:**
  BatchNormalization()

**PyTorch equivalent:**
  ('bn', [num_features])
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Test Keras to PyTorch weight conversion'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to converted PyTorch checkpoint')
    parser.add_argument('--compare_architecture', action='store_true',
                        help='Show architecture comparison guide')
    
    args = parser.parse_args()
    
    # Example configuration (should match your actual model)
    # This is the SRCNN + DNCNN architecture from MAML_trainer_with_tracking.py
    config = [
        # SRCNN Layer 1
        ('conv2d', [64, 2, 9, 9, 1, 4]),
        ('tanh', [True]),
        
        # SRCNN Layer 2
        ('conv2d', [32, 64, 1, 1, 1, 0]),
        ('tanh', [True]),
        
        # SRCNN Layer 3
        ('conv2d', [2, 32, 5, 5, 1, 2]),
        ('tanh', [True]),
        
        # DNCNN Layers
        ('conv2d', [64, 2, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        # Additional DNCNN layers...
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('tanh', [True]),
        ('bn', [64]),
        
        # Final layer
        ('conv2d', [2, 64, 3, 3, 1, 1])
    ]
    
    if args.compare_architecture:
        compare_model_architectures()
    else:
        success = test_weight_loading(args.checkpoint, config)
        
        if not success:
            print("\n⚠ Test failed. Possible issues:")
            print("  1. Checkpoint file is corrupted or wrong format")
            print("  2. Model architecture doesn't match")
            print("  3. Conversion script had errors")
            print("\nTry:")
            print("  - Re-run the conversion script")
            print("  - Check that Keras model loads correctly")
            print("  - Verify architecture matches between Keras and PyTorch")


if __name__ == '__main__':
    main()

