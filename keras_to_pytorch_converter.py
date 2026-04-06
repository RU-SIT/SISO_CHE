#!/usr/bin/env python3
"""
Keras/TensorFlow to PyTorch Weight Converter
============================================

This script helps transfer pre-trained weights from a Keras ChannelNet model
to a PyTorch MAML model for better initialization.

Key Concepts:
- Keras Conv2D weights shape: (kernel_h, kernel_w, in_channels, out_channels)
- PyTorch Conv2d weights shape: (out_channels, in_channels, kernel_h, kernel_w)
- We need to transpose and reorder dimensions when converting
"""

import numpy as np
import torch
import os
import h5py
import argparse
from collections import OrderedDict


def load_keras_model_weights(keras_model_path):
    """
    Load weights from a Keras model file (.h5 or .keras format).
    
    Args:
        keras_model_path: Path to the saved Keras model
        
    Returns:
        Dictionary mapping layer names to weight arrays
    """
    try:
        # Try importing TensorFlow/Keras
        import tensorflow as tf
        from tensorflow import keras
        
        print(f"Loading Keras model from: {keras_model_path}")
        model = keras.models.load_model(keras_model_path, compile=False)
        
        # Extract weights layer by layer
        keras_weights = {}
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if len(layer_weights) > 0:
                keras_weights[layer.name] = {
                    'weights': layer_weights,
                    'layer_type': layer.__class__.__name__
                }
                print(f"  - {layer.name} ({layer.__class__.__name__}): {len(layer_weights)} arrays")
        
        return keras_weights, model
        
    except ImportError:
        print("TensorFlow not available. Trying to load weights from HDF5 directly...")
        return load_weights_from_h5(keras_model_path)


def load_weights_from_h5(h5_path):
    """
    Alternative method: Load weights directly from HDF5 file without TensorFlow.
    
    Args:
        h5_path: Path to .h5 file
        
    Returns:
        Dictionary of weights
    """
    keras_weights = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Navigate the HDF5 structure
        if 'model_weights' in f.keys():
            model_weights = f['model_weights']
        else:
            model_weights = f
            
        # Extract weights from each layer
        for layer_name in model_weights.keys():
            layer_group = model_weights[layer_name]
            
            # Get weight names in this layer
            if isinstance(layer_group, h5py.Group):
                weight_names = list(layer_group.keys())
                if len(weight_names) > 0:
                    weights = []
                    for weight_name in weight_names:
                        weights.append(np.array(layer_group[weight_name]))
                    
                    keras_weights[layer_name] = {
                        'weights': weights,
                        'layer_type': 'Unknown'
                    }
                    print(f"  - {layer_name}: {len(weights)} arrays")
    
    return keras_weights, None


def convert_conv2d_weights(keras_weight):
    """
    Convert Keras Conv2D weights to PyTorch Conv2d format.
    
    Keras Conv2D: (kernel_h, kernel_w, in_channels, out_channels)
    PyTorch Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
    
    Args:
        keras_weight: Numpy array from Keras
        
    Returns:
        PyTorch tensor with correct dimension order
    """
    # Transpose: (H, W, in, out) -> (out, in, H, W)
    pytorch_weight = np.transpose(keras_weight, (3, 2, 0, 1))
    return torch.from_numpy(pytorch_weight)


def convert_batchnorm_weights(keras_weights_list):
    """
    Convert Keras BatchNormalization weights to PyTorch BatchNorm format.
    
    Keras BN typically has: [gamma, beta, moving_mean, moving_variance]
    PyTorch BN expects: weight, bias, running_mean, running_var
    
    Args:
        keras_weights_list: List of numpy arrays [gamma, beta, mean, var]
        
    Returns:
        Dictionary with PyTorch parameter names
    """
    bn_dict = {}
    
    if len(keras_weights_list) >= 4:
        bn_dict['weight'] = torch.from_numpy(keras_weights_list[0])  # gamma
        bn_dict['bias'] = torch.from_numpy(keras_weights_list[1])    # beta
        bn_dict['running_mean'] = torch.from_numpy(keras_weights_list[2])
        bn_dict['running_var'] = torch.from_numpy(keras_weights_list[3])
    elif len(keras_weights_list) == 2:
        # Sometimes only gamma and beta are saved
        bn_dict['weight'] = torch.from_numpy(keras_weights_list[0])
        bn_dict['bias'] = torch.from_numpy(keras_weights_list[1])
    
    return bn_dict


def create_pytorch_state_dict(keras_weights, layer_mapping=None):
    """
    Create a PyTorch state dictionary from Keras weights.
    
    Args:
        keras_weights: Dictionary of Keras weights from load_keras_model_weights()
        layer_mapping: Optional manual mapping from Keras layer names to PyTorch parameter names
        
    Returns:
        PyTorch state dictionary ready to load
    """
    pytorch_state_dict = OrderedDict()
    
    # Convert each layer's weights
    conv_idx = 0
    bn_idx = 0
    
    for keras_layer_name, layer_info in keras_weights.items():
        weights = layer_info['weights']
        layer_type = layer_info['layer_type']
        
        print(f"\nProcessing {keras_layer_name} ({layer_type})...")
        
        # Handle Conv2D layers
        if 'Conv' in layer_type and len(weights) > 0:
            # Convert convolutional weights
            conv_weight = convert_conv2d_weights(weights[0])
            pytorch_state_dict[f'vars.{conv_idx}'] = conv_weight
            print(f"  → vars.{conv_idx} (Conv weight): {conv_weight.shape}")
            conv_idx += 1
            
            # Convert bias if present
            if len(weights) > 1:
                conv_bias = torch.from_numpy(weights[1])
                pytorch_state_dict[f'vars.{conv_idx}'] = conv_bias
                print(f"  → vars.{conv_idx} (Conv bias): {conv_bias.shape}")
                conv_idx += 1
        
        # Handle BatchNormalization layers
        elif 'BatchNorm' in layer_type or 'batch_norm' in keras_layer_name.lower():
            bn_params = convert_batchnorm_weights(weights)
            
            for param_name, param_value in bn_params.items():
                pytorch_state_dict[f'vars_bn.{bn_idx}.{param_name}'] = param_value
                print(f"  → vars_bn.{bn_idx}.{param_name}: {param_value.shape}")
            
            bn_idx += 1
    
    print(f"\nTotal PyTorch parameters created: {len(pytorch_state_dict)}")
    return pytorch_state_dict


def save_pytorch_checkpoint(state_dict, output_path, metadata=None):
    """
    Save the converted weights as a PyTorch checkpoint.
    
    Args:
        state_dict: PyTorch state dictionary
        output_path: Where to save the .pth.tar file
        metadata: Additional information to save (optional)
    """
    checkpoint = {
        'state_dict': state_dict,
        'source': 'Converted from Keras ChannelNet',
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, output_path)
    print(f"\n✓ PyTorch checkpoint saved to: {output_path}")


def main():
    """
    Main conversion function.
    
    Usage:
        python keras_to_pytorch_converter.py --keras_model path/to/model.h5 \
                                               --output converted_weights.pth.tar
    """
    parser = argparse.ArgumentParser(
        description='Convert Keras ChannelNet weights to PyTorch MAML format'
    )
    parser.add_argument('--keras_model', type=str, required=True,
                        help='Path to Keras model file (.h5 or .keras)')
    parser.add_argument('--output', type=str, default='keras_to_pytorch_weights.pth.tar',
                        help='Output path for PyTorch checkpoint')
    parser.add_argument('--show_architecture', action='store_true',
                        help='Show detailed architecture information')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.keras_model):
        print(f"Error: Keras model file not found: {args.keras_model}")
        return
    
    print("="*70)
    print("Keras to PyTorch Weight Converter")
    print("="*70)
    
    # Step 1: Load Keras weights
    keras_weights, keras_model = load_keras_model_weights(args.keras_model)
    
    if args.show_architecture and keras_model is not None:
        print("\n" + "="*70)
        print("Keras Model Architecture:")
        print("="*70)
        keras_model.summary()
    
    # Step 2: Convert to PyTorch format
    print("\n" + "="*70)
    print("Converting to PyTorch format...")
    print("="*70)
    pytorch_state_dict = create_pytorch_state_dict(keras_weights)
    
    # Step 3: Save PyTorch checkpoint
    save_pytorch_checkpoint(
        pytorch_state_dict,
        args.output,
        metadata={
            'keras_source': args.keras_model,
            'num_parameters': len(pytorch_state_dict)
        }
    )
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"You can now load these weights in your MAML model using:")
    print(f"  checkpoint = torch.load('{args.output}')")
    print(f"  maml.load_state_dict(checkpoint['state_dict'], strict=False)")


if __name__ == '__main__':
    main()

