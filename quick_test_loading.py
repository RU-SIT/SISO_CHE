#!/usr/bin/env python3
"""Quick test to verify DNCNN weights load into MAML correctly."""

import torch
import argparse
from meta import Meta

# DNCNN-only config (matching your edited MAML_trainer_with_tracking.py)
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

print("="*70)
print("Quick Weight Loading Test")
print("="*70)

# Create MAML model
args = DummyArgs()
maml = Meta(args, config)

print(f"\nMAML model created")
print(f"Total parameters in MAML: {len(list(maml.state_dict().keys()))}")

# List first few parameter names
print("\nFirst 10 MAML parameter names:")
for i, key in enumerate(list(maml.state_dict().keys())[:10]):
    print(f"  {i+1}. {key}")

# Load checkpoint
checkpoint_path = "DNCNN_pretrained_init.pth.tar"
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

print(f"Checkpoint contains: {len(state_dict)} parameters")
print("\nFirst 10 checkpoint parameter names:")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {i+1}. {key}")

# Try to load
print("\n" + "="*70)
print("Attempting to load weights...")
print("="*70)

missing_keys, unexpected_keys = maml.load_state_dict(state_dict, strict=False)

loaded_count = len(state_dict) - len(unexpected_keys)
total_params = len(maml.state_dict())

print(f"\n✓ Successfully loaded {loaded_count}/{total_params} parameters")

if missing_keys:
    print(f"\n⚠ Missing {len(missing_keys)} parameters:")
    for key in missing_keys[:5]:
        print(f"  - {key}")
    if len(missing_keys) > 5:
        print(f"  ... and {len(missing_keys) - 5} more")

if unexpected_keys:
    print(f"\n⚠ Unexpected {len(unexpected_keys)} parameters:")
    for key in unexpected_keys[:5]:
        print(f"  - {key}")
    if len(unexpected_keys) > 5:
        print(f"  ... and {len(unexpected_keys) - 5} more")

if not missing_keys and not unexpected_keys:
    print("\n🎉 PERFECT MATCH! All weights loaded successfully!")
    print("✓ Your MAML architecture matches the DNCNN weights perfectly!")
elif loaded_count / total_params > 0.9:
    print(f"\n✓ GOOD! {loaded_count/total_params*100:.1f}% of weights loaded.")
    print("This is sufficient for transfer learning.")
elif loaded_count / total_params > 0.5:
    print(f"\n⚠ OK: {loaded_count/total_params*100:.1f}% of weights loaded.")
    print("Partial transfer learning will work, but not optimal.")
else:
    print(f"\n❌ PROBLEM: Only {loaded_count/total_params*100:.1f}% of weights loaded.")
    print("Architecture mismatch is too large.")

print("\n" + "="*70)

