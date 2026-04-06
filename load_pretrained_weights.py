#!/usr/bin/env python3
"""
Helper Script: Load Pre-trained Weights into MAML
==================================================

This script demonstrates how to initialize your MAML model with 
pre-trained ChannelNet weights converted from Keras.
"""

import torch
import argparse
from meta import Meta


def load_pretrained_weights_into_maml(maml_model, checkpoint_path, strict=False):
    """
    Load pre-trained weights from a checkpoint into MAML model.
    
    Args:
        maml_model: Your MAML/Meta model instance
        checkpoint_path: Path to the converted PyTorch checkpoint
        strict: If False, allows partial loading (recommended for transfer learning)
        
    Returns:
        maml_model with loaded weights
        
    Explanation:
        - strict=False allows loading even if some layers don't match perfectly
        - This is useful when architectures are similar but not identical
        - Missing or extra keys will be reported but won't cause errors
    """
    print(f"Loading pre-trained weights from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # The state dictionary might be nested
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"Source: {checkpoint.get('source', 'Unknown')}")
    else:
        state_dict = checkpoint
    
    print(f"Checkpoint contains {len(state_dict)} parameters")
    
    # Try to load weights
    try:
        # Load with strict=False to allow partial matching
        missing_keys, unexpected_keys = maml_model.load_state_dict(
            state_dict, 
            strict=strict
        )
        
        if missing_keys:
            print(f"\n⚠ Warning: {len(missing_keys)} keys in model not found in checkpoint:")
            for key in missing_keys[:5]:  # Show first 5
                print(f"  - {key}")
            if len(missing_keys) > 5:
                print(f"  ... and {len(missing_keys) - 5} more")
        
        if unexpected_keys:
            print(f"\n⚠ Warning: {len(unexpected_keys)} keys in checkpoint not used:")
            for key in unexpected_keys[:5]:
                print(f"  - {key}")
            if len(unexpected_keys) > 5:
                print(f"  ... and {len(unexpected_keys) - 5} more")
        
        if not missing_keys and not unexpected_keys:
            print("✓ All weights loaded successfully!")
        else:
            print("✓ Weights loaded with partial matching (this is normal for transfer learning)")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: manually match layers
        model_state = maml_model.state_dict()
        loaded_count = 0
        
        for name, param in state_dict.items():
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    loaded_count += 1
                else:
                    print(f"  Shape mismatch for {name}: "
                          f"model {model_state[name].shape} vs checkpoint {param.shape}")
        
        print(f"✓ Manually loaded {loaded_count}/{len(model_state)} parameters")
    
    return maml_model


def verify_weight_loading(model1, model2, num_samples=5):
    """
    Verify that weights were loaded by comparing parameter values.
    
    Args:
        model1: Model before loading
        model2: Model after loading
        num_samples: Number of parameters to check
    """
    print("\nVerifying weight transfer...")
    
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())
    
    different_count = 0
    for i in range(min(num_samples, len(params1), len(params2))):
        if not torch.equal(params1[i], params2[i]):
            different_count += 1
    
    if different_count > 0:
        print(f"✓ Weights changed! {different_count}/{num_samples} sampled parameters are different.")
        print("  This confirms that pre-trained weights were loaded.")
    else:
        print("⚠ Warning: Weights appear unchanged. Loading may have failed.")


def example_usage():
    """
    Example of how to use this in your MAML training script.
    """
    print("\n" + "="*70)
    print("EXAMPLE USAGE IN YOUR MAML SCRIPT")
    print("="*70)
    
    example_code = """
# In your MAML_trainer_with_tracking.py, modify the main() function:

def main(args):
    # ... (existing setup code) ...
    
    # Create MAML model
    maml = MetaWithTracking(args, config, loss_tracker)
    
    # NEW: Load pre-trained ChannelNet weights if provided
    if args.pretrained_weights is not None:
        print("\\nInitializing with pre-trained ChannelNet weights...")
        maml = load_pretrained_weights_into_maml(
            maml, 
            args.pretrained_weights, 
            strict=False
        )
        print("✓ Pre-trained weights loaded!\\n")
    else:
        print("\\nUsing random initialization (no pre-trained weights)\\n")
    
    maml.to(device)
    
    # ... (continue with training) ...


# Then add this argument to your argument parser:
parser.add_argument('--pretrained_weights', type=str, default=None,
                    help='Path to pre-trained weights (converted from Keras)')

# Usage:
# python MAML_trainer_with_tracking.py --pretrained_weights keras_to_pytorch_weights.pth.tar
"""
    print(example_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test loading pre-trained weights'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to converted PyTorch checkpoint')
    parser.add_argument('--config', type=str, 
                        help='Path to model config (optional)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Testing Pre-trained Weight Loading")
    print("="*70)
    
    # This is a test - you'd normally do this in your training script
    print("\nNote: This is a demonstration. To actually use pre-trained weights,")
    print("integrate the load_pretrained_weights_into_maml() function into")
    print("your MAML_trainer_with_tracking.py script.")
    
    example_usage()

