#!/usr/bin/env python3
"""
Keras Weights-Only to PyTorch Converter
========================================

This script handles Keras weights-only files (*.weights.h5) and converts them
to PyTorch format WITHOUT needing the full model architecture.

Use this when you have .weights.h5 files instead of full model files.
"""

import numpy as np
import torch
import h5py
import argparse
from collections import OrderedDict
import os


def load_weights_from_h5(h5_path):
    """
    Load weights directly from HDF5 weights file.
    
    Args:
        h5_path: Path to .weights.h5 file
        
    Returns:
        Dictionary of layer weights
    """
    weights_dict = {}
    
    print(f"Loading weights from: {h5_path}")
    print("-" * 70)
    
    with h5py.File(h5_path, 'r') as f:
        # The structure depends on how weights were saved
        # Common structures: f['model_weights'] or direct layer groups
        
        def explore_group(group, prefix=''):
            """Recursively explore HDF5 group structure."""
            for key in group.keys():
                item = group[key]
                current_path = f"{prefix}/{key}" if prefix else key
                
                if isinstance(item, h5py.Group):
                    # It's a group, explore recursively
                    explore_group(item, current_path)
                elif isinstance(item, h5py.Dataset):
                    # It's a dataset (actual weights)
                    weights_dict[current_path] = np.array(item)
                    print(f"  Found: {current_path} - Shape: {weights_dict[current_path].shape}")
        
        # Start exploration from root
        explore_group(f)
    
    print("-" * 70)
    print(f"Total weight arrays found: {len(weights_dict)}")
    return weights_dict


def organize_weights_by_layer(weights_dict):
    """
    Organize flat weight dictionary into layer-based structure.
    
    Args:
        weights_dict: Flat dictionary of all weights
        
    Returns:
        Dictionary organized by layers
    """
    organized = {}
    
    # Group weights by layer
    for key, value in weights_dict.items():
        # Extract layer name (before the last '/')
        parts = key.split('/')
        
        # Find the layer name
        layer_name = None
        for part in parts:
            if 'conv' in part.lower() or 'dense' in part.lower() or 'batch' in part.lower():
                layer_name = part
                break
        
        if layer_name is None:
            layer_name = parts[-2] if len(parts) > 1 else 'unknown'
        
        if layer_name not in organized:
            organized[layer_name] = []
        
        organized[layer_name].append((key, value))
    
    return organized


def convert_conv2d_weights(weight_array):
    """
    Convert Keras Conv2D weights to PyTorch format.
    
    Keras: (H, W, in, out) → PyTorch: (out, in, H, W)
    """
    if len(weight_array.shape) == 4:
        # This is a convolutional kernel
        return torch.from_numpy(np.transpose(weight_array, (3, 2, 0, 1)))
    else:
        # This is a bias or other 1D parameter
        return torch.from_numpy(weight_array)


def create_pytorch_state_dict_from_weights(weights_dict, layer_prefix='net.vars', add_net_prefix=True):
    """
    Create PyTorch state dictionary from raw weights.
    
    Args:
        weights_dict: Dictionary of weight arrays from HDF5
        layer_prefix: Prefix for parameter names (default: 'net.vars')
        add_net_prefix: Whether to add 'net.' prefix for MAML compatibility
        
    Returns:
        PyTorch state dictionary
    """
    state_dict = OrderedDict()
    
    print("\nConverting to PyTorch format...")
    print("=" * 70)
    
    # Organize weights by layer
    organized = organize_weights_by_layer(weights_dict)
    
    param_idx = 0
    bn_idx = 0
    
    # Filter out optimizer variables (we only want model weights)
    model_layers = {k: v for k, v in organized.items() 
                    if not k.startswith('vars') and 'optimizer' not in k.lower()}
    
    # Sort layers by their numeric index (extract number from layer name)
    def extract_layer_number(layer_name):
        """Extract numeric index from layer name for proper sorting."""
        import re
        # Find numbers in the layer name (e.g., batch_normalization_5, conv2d_10)
        # Pattern: layer_name_NUMBER
        match = re.search(r'_(\d+)$', layer_name)
        if match:
            return int(match.group(1))
        # No number suffix means it's layer 0 (e.g., "conv2d", "batch_normalization")
        return 0
    
    # Sort by: layer type (conv before bn) and then by layer number
    def sort_key(layer_name):
        num = extract_layer_number(layer_name)
        # Conv layers before BN layers at same index
        if 'conv' in layer_name.lower():
            return (num, 0)  # conv first
        elif 'batch' in layer_name.lower() or 'bn' in layer_name.lower():
            return (num, 1)  # bn second
        else:
            return (num, 2)  # others last
    
    # Sort layers to maintain correct order
    for layer_name in sorted(model_layers.keys(), key=sort_key):
        layer_weights = model_layers[layer_name]
        
        print(f"\nProcessing: {layer_name}")
        
        is_conv = 'conv' in layer_name.lower()
        is_bn = 'batch' in layer_name.lower() or 'bn' in layer_name.lower()
        
        if is_bn:
            # BatchNorm layer - MAML stores differently than standard PyTorch:
            # - weight and bias go in 'vars' (trainable parameters)
            # - running_mean and running_var go in 'vars_bn' (non-trainable stats)
            
            # Sort the weights to ensure correct order
            bn_params = {}
            for key, weight in layer_weights:
                full_path = key.split('/')
                # The last part after 'vars' indicates parameter type
                if 'vars' in full_path:
                    var_idx = full_path.index('vars')
                    if var_idx + 1 < len(full_path):
                        param_idx_str = full_path[var_idx + 1]
                        bn_params[int(param_idx_str)] = weight
            
            # Map to MAML's BatchNorm structure
            # Keras typically stores as: [gamma=0, beta=1, moving_mean=2, moving_variance=3]
            if len(bn_params) >= 4:
                prefix = 'net.' if add_net_prefix else ''
                
                # Weight and bias go in vars (trainable)
                state_dict[f'{prefix}vars.{param_idx}'] = torch.from_numpy(bn_params[0])  # gamma/weight
                print(f"  → {prefix}vars.{param_idx} (BN weight): {tuple(bn_params[0].shape)}")
                param_idx += 1
                
                state_dict[f'{prefix}vars.{param_idx}'] = torch.from_numpy(bn_params[1])  # beta/bias
                print(f"  → {prefix}vars.{param_idx} (BN bias): {tuple(bn_params[1].shape)}")
                param_idx += 1
                
                # Running stats go in vars_bn (non-trainable)
                state_dict[f'{prefix}vars_bn.{bn_idx}'] = torch.from_numpy(bn_params[2])  # running_mean
                print(f"  → {prefix}vars_bn.{bn_idx} (running_mean): {tuple(bn_params[2].shape)}")
                bn_idx += 1
                
                state_dict[f'{prefix}vars_bn.{bn_idx}'] = torch.from_numpy(bn_params[3])  # running_var
                print(f"  → {prefix}vars_bn.{bn_idx} (running_var): {tuple(bn_params[3].shape)}")
                bn_idx += 1
            
        elif is_conv:
            # Convolutional layer - should have kernel and bias
            conv_params = {}
            for key, weight in layer_weights:
                full_path = key.split('/')
                if 'vars' in full_path:
                    var_idx = full_path.index('vars')
                    if var_idx + 1 < len(full_path):
                        param_idx_str = full_path[var_idx + 1]
                        conv_params[int(param_idx_str)] = weight
            
            # Process in order: kernel (0), then bias (1)
            for idx in sorted(conv_params.keys()):
                weight = conv_params[idx]
                prefix = 'net.' if add_net_prefix else ''
                
                if len(weight.shape) == 4:
                    # This is a conv kernel
                    converted = convert_conv2d_weights(weight)
                    pytorch_key = f'{prefix}vars.{param_idx}'
                    state_dict[pytorch_key] = converted
                    print(f"  → {pytorch_key} (kernel): {tuple(converted.shape)}")
                else:
                    # This is a bias
                    pytorch_key = f'{prefix}vars.{param_idx}'
                    state_dict[pytorch_key] = torch.from_numpy(weight)
                    print(f"  → {pytorch_key} (bias): {tuple(weight.shape)}")
                
                param_idx += 1
    
    print("\n" + "=" * 70)
    print(f"Total PyTorch parameters created: {len(state_dict)}")
    
    return state_dict


def save_pytorch_checkpoint(state_dict, output_path, source_file):
    """
    Save converted weights as PyTorch checkpoint.
    
    Args:
        state_dict: PyTorch state dictionary
        output_path: Where to save
        source_file: Original weights file path
    """
    checkpoint = {
        'state_dict': state_dict,
        'source': 'Converted from Keras weights-only file',
        'keras_weights_file': source_file,
        'num_parameters': len(state_dict)
    }
    
    torch.save(checkpoint, output_path)
    print(f"\n✓ PyTorch checkpoint saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Keras weights-only files to PyTorch format'
    )
    parser.add_argument('--weights_file', type=str, required=True,
                        help='Path to Keras .weights.h5 file')
    parser.add_argument('--output', type=str, default='converted_weights.pth.tar',
                        help='Output path for PyTorch checkpoint')
    parser.add_argument('--show_structure', action='store_true',
                        help='Show detailed HDF5 structure')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights_file):
        print(f"Error: Weights file not found: {args.weights_file}")
        return
    
    print("=" * 70)
    print("Keras Weights-Only to PyTorch Converter")
    print("=" * 70)
    print(f"Input:  {args.weights_file}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # Load weights from HDF5
    weights_dict = load_weights_from_h5(args.weights_file)
    
    if len(weights_dict) == 0:
        print("\n⚠ Warning: No weights found in file!")
        print("The file might be empty or have an unexpected structure.")
        return
    
    # Convert to PyTorch format (with 'net.' prefix for MAML compatibility)
    state_dict = create_pytorch_state_dict_from_weights(weights_dict, layer_prefix='net.vars', add_net_prefix=True)
    
    # Save checkpoint
    save_pytorch_checkpoint(state_dict, args.output, args.weights_file)
    
    print("\n" + "=" * 70)
    print("Conversion Complete!")
    print("=" * 70)
    print(f"You can now use these weights in MAML:")
    print(f"  python MAML_trainer_with_tracking.py \\")
    print(f"      --pretrained_weights {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()

