#!/usr/bin/env python3
"""Diagnose architecture mismatch between checkpoint and MAML model."""

import torch
from meta import Meta
import argparse

# Your current MAML config (DNCNN only, SRCNN commented out)
config = [
    # DNCNN Layer 1: 3x3 conv, in=2, out=64
    ('conv2d', [64, 2, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    # DNCNN Layers 2-19: 18 layers of 3x3 conv, in=64, out=64 with BN
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    ('conv2d', [64, 64, 3, 3, 1, 1]),
    ('tanh', [True]),
    ('bn', [64]),
    
    # DNCNN Layer 20: 3x3 conv, in=64, out=2 (final output)
    ('conv2d', [2, 64, 3, 3, 1, 1])
]

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
        self.n_way = 5
        self.batchsz = 8

print("="*80)
print("ARCHITECTURE DIAGNOSIS")
print("="*80)

# Create MAML model
print("\n1️⃣ Creating MAML model from config...")
try:
    args = DummyArgs()
    # Just create the learner part without full Meta initialization
    from learner import Learner
    learner = Learner(config)  # Only pass config
    
    print(f"✓ Model created successfully")
    
    # Get model parameters
    model_params = list(learner.state_dict().keys())
    print(f"  Total parameters: {len(model_params)}")
    
    print(f"\n📋 First 20 MAML model parameters:")
    for i, key in enumerate(model_params[:20]):
        param_shape = learner.state_dict()[key].shape
        print(f"  {i:2d}. {key:25s} shape: {tuple(param_shape)}")
    
except Exception as e:
    print(f"❌ Error creating model: {e}")
    print("\nTrying alternative approach with Meta class...")
    
    # Try with simplified Meta
    import sys
    sys.exit(1)

# Load checkpoint
print(f"\n2️⃣ Loading checkpoint...")
checkpoint_path = "DNCNN_pretrained_init.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

# Get checkpoint parameters
ckpt_params = list(state_dict.keys())
print(f"✓ Checkpoint loaded")
print(f"  Total parameters: {len(ckpt_params)}")

print(f"\n📋 First 20 Checkpoint parameters:")
for i, key in enumerate(ckpt_params[:20]):
    param_shape = state_dict[key].shape
    print(f"  {i:2d}. {key:25s} shape: {tuple(param_shape)}")

# Compare
print(f"\n3️⃣ Comparison:")
print(f"  MAML model params: {len(model_params)}")
print(f"  Checkpoint params: {len(ckpt_params)}")
print(f"  Match: {'✓ YES' if len(model_params) == len(ckpt_params) else '❌ NO'}")

# Find mismatches
print(f"\n4️⃣ Finding mismatches in first 40 parameters (conv layers):")
conv_model = [k for k in model_params if 'vars.' in k and 'vars_bn' not in k][:40]
conv_ckpt = [k for k in ckpt_params if 'vars.' in k and 'vars_bn' not in k][:40]

mismatches = []
for i in range(min(len(conv_model), len(conv_ckpt))):
    model_key = conv_model[i]
    ckpt_key = conv_ckpt[i]
    model_shape = learner.state_dict()[model_key].shape
    ckpt_shape = state_dict[ckpt_key].shape
    
    if model_shape != ckpt_shape:
        mismatches.append((i, model_key, model_shape, ckpt_key, ckpt_shape))
        if len(mismatches) <= 5:  # Show first 5
            print(f"  ❌ Index {i}: {model_key} {model_shape} vs {ckpt_key} {ckpt_shape}")

if not mismatches:
    print("  ✓ All shapes match!")
else:
    print(f"\n  Total mismatches: {len(mismatches)}")

print("\n" + "="*80)

