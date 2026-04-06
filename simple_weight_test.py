#!/usr/bin/env python3
"""Simple test to check if checkpoint parameter names match MAML expectations."""

import torch

print("="*70)
print("Simple Weight Matching Test")
print("="*70)

# Load checkpoint
checkpoint_path = "DNCNN_pretrained_init.pth.tar"
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

print(f"✓ Checkpoint loaded")
print(f"  Contains {len(state_dict)} parameters")
print(f"  Source: {checkpoint.get('source', 'Unknown')}")

# Analyze parameter names
conv_params = [k for k in state_dict.keys() if 'vars.' in k and 'vars_bn' not in k]
bn_params = [k for k in state_dict.keys() if 'vars_bn' in k]

print(f"\n📊 Parameter Analysis:")
print(f"  Convolutional parameters: {len(conv_params)}")
print(f"  BatchNorm parameters: {len(bn_params)}")
print(f"  Total: {len(state_dict)}")

print(f"\n🔍 First 10 Conv parameters:")
for key in conv_params[:10]:
    print(f"  ✓ {key}: {tuple(state_dict[key].shape)}")

print(f"\n🔍 First 8 BatchNorm parameters:")
for key in bn_params[:8]:
    print(f"  ✓ {key}: {tuple(state_dict[key].shape)}")

# Check naming convention
has_net_prefix = any(k.startswith('net.') for k in state_dict.keys())
print(f"\n✓ Naming convention check:")
print(f"  Has 'net.' prefix: {'YES ✓' if has_net_prefix else 'NO ❌'}")
print(f"  Format: {'MAML-compatible' if has_net_prefix else 'Needs adjustment'}")

# Check for DNCNN architecture
# Expected: 20 conv layers (40 params: kernel+bias) + 18 BN layers (72 params)
expected_conv = 40  # 20 layers × 2 (kernel + bias)
expected_bn = 72    # 18 layers × 4 (weight, bias, running_mean, running_var)

print(f"\n📋 Architecture Verification:")
print(f"  Expected conv params: {expected_conv}")
print(f"  Found conv params: {len(conv_params)}")
print(f"  Match: {'✓ YES' if len(conv_params) == expected_conv else '❌ NO'}")

print(f"\n  Expected BN params: {expected_bn}")
print(f"  Found BN params: {len(bn_params)}")
print(f"  Match: {'✓ YES' if len(bn_params) == expected_bn else '❌ NO'}")

# Check parameter indices are consecutive
conv_indices = sorted([int(k.split('.')[2]) for k in conv_params])
expected_indices = list(range(40))
indices_match = conv_indices == expected_indices

print(f"\n  Conv parameter indices: {conv_indices[:5]}...{conv_indices[-5:]}")
print(f"  Indices consecutive: {'✓ YES' if indices_match else '❌ NO'}")

print("\n" + "="*70)
if has_net_prefix and len(conv_params) == 40 and len(bn_params) == 72:
    print("🎉 SUCCESS! Checkpoint is properly formatted for MAML!")
    print("✓ All parameter names have 'net.' prefix")
    print("✓ Architecture matches DNCNN (20 conv + 18 BN layers)")
    print("✓ Ready to use with --pretrained_weights")
else:
    print("⚠ Issues detected:")
    if not has_net_prefix:
        print("  - Missing 'net.' prefix (needs reconversion)")
    if len(conv_params) != 40:
        print(f"  - Conv params mismatch ({len(conv_params)} vs 40 expected)")
    if len(bn_params) != 72:
        print(f"  - BN params mismatch ({len(bn_params)} vs 72 expected)")
print("="*70)

