# Transferring Keras ChannelNet Weights to PyTorch MAML

## Overview

This guide explains how to use your pre-trained **TensorFlow/Keras ChannelNet** model to initialize your **PyTorch MAML** model, instead of starting from random weights. This is called **transfer learning across frameworks**.

---

## Why Transfer Learning?

### Benefits:
1. **Faster Convergence**: Pre-trained weights provide a better starting point than random initialization
2. **Better Performance**: The model already "knows" useful features from ChannelNet training
3. **Less Data Needed**: Fine-tuning pre-trained weights often requires fewer samples
4. **Reduced Training Time**: You start closer to a good solution

### The Challenge:
- **ChannelNet** is in TensorFlow/Keras
- **MAML** is in PyTorch
- Different frameworks have different:
  - Weight formats (dimension ordering)
  - Naming conventions
  - Internal representations

### The Solution:
Convert the weights from Keras format → NumPy arrays → PyTorch tensors

---

## Step-by-Step Process

### Step 1: Prepare Your Keras Model

First, make sure you have your trained ChannelNet model saved. It should be in one of these formats:
- `.h5` file (HDF5 format)
- `.keras` file (newer Keras format)
- SavedModel directory

**Example:**
```bash
# Your Keras model should be saved like this:
your_trained_channelnet_model.h5
```

---

### Step 2: Convert Keras Weights to PyTorch Format

Use the provided conversion script:

```bash
python keras_to_pytorch_converter.py \
    --keras_model path/to/your/channelnet_model.h5 \
    --output converted_channelnet_weights.pth.tar \
    --show_architecture
```

**What this does:**
1. Loads your Keras model
2. Extracts all weight matrices (conv layers, batch norm, etc.)
3. Converts dimension ordering:
   - Keras Conv2D: `(height, width, in_channels, out_channels)`
   - PyTorch Conv2d: `(out_channels, in_channels, height, width)`
4. Saves as PyTorch checkpoint file (`.pth.tar`)

**Output:**
- `converted_channelnet_weights.pth.tar` - Ready to use with PyTorch!

---

### Step 3: Train MAML with Pre-trained Weights

Now use the converted weights to initialize your MAML training:

```bash
python MAML_trainer_with_tracking.py \
    --pretrained_weights converted_channelnet_weights.pth.tar \
    --root Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak \
    --device cuda:0 \
    --epoch 5000 \
    --n_way 5 \
    --k_spt 5 \
    --k_qry 5 \
    --meta_lr 5e-4 \
    --update_lr 1e-4
```

**What happens:**
1. MAML model is created
2. Pre-trained ChannelNet weights are loaded into it
3. Training starts from these weights instead of random values
4. Inner loop adaptations build on top of the pre-trained features

---

## Technical Details

### Weight Dimension Conversion

**Convolutional Layers:**
- Keras stores as: `(kernel_h, kernel_w, in_channels, out_channels)`
- PyTorch expects: `(out_channels, in_channels, kernel_h, kernel_w)`
- Solution: Transpose using `np.transpose(keras_weight, (3, 2, 0, 1))`

**Batch Normalization:**
- Keras: `[gamma, beta, moving_mean, moving_variance]`
- PyTorch: `[weight, bias, running_mean, running_var]`
- These map directly with simple renaming

### Handling Architecture Mismatches

**Q: What if my ChannelNet and MAML architectures are slightly different?**

A: The code uses `strict=False` when loading weights:
- **Matching layers**: Weights are transferred
- **Extra layers in ChannelNet**: Ignored
- **Missing layers in MAML**: Use random initialization
- **Best practice**: Keep architectures as similar as possible

---

## Example Workflow

### Complete Example from Start to Finish:

```bash
# Step 1: Train ChannelNet in Keras (you've already done this)
# Result: channelnet_trained_model.h5

# Step 2: Convert to PyTorch
python keras_to_pytorch_converter.py \
    --keras_model channelnet_trained_model.h5 \
    --output channelnet_pytorch_init.pth.tar

# Step 3: Train MAML with pre-trained initialization
python MAML_trainer_with_tracking.py \
    --pretrained_weights channelnet_pytorch_init.pth.tar \
    --epoch 5000 \
    --meta_lr 5e-4 \
    --update_lr 1e-4 \
    --enable_early_stopping \
    --early_stop_patience 30
```

---

## Verification

### How to verify weights were loaded correctly:

Check the console output during training:

```
======================================================================
Loading pre-trained weights from: channelnet_pytorch_init.pth.tar
======================================================================
Source: Converted from Keras ChannelNet
Original Keras model: channelnet_trained_model.h5
Checkpoint contains 42 parameters
MAML model has 42 parameters

✓ Successfully loaded 42/42 parameters

✓ Perfect match! All weights loaded successfully!
======================================================================
```

### Signs of successful transfer:
1. **Lower initial loss**: First epoch loss should be much lower than random init
2. **Faster convergence**: Training should reach good performance sooner
3. **Better final performance**: Final accuracy/loss should improve

---

## Troubleshooting

### Problem 1: TensorFlow not installed

**Error:** `ImportError: No module named 'tensorflow'`

**Solution:**
```bash
pip install tensorflow
# or if you need GPU support:
pip install tensorflow-gpu
```

### Problem 2: Shape mismatch

**Error:** `Shape mismatch for layer X`

**Cause:** Your ChannelNet and MAML architectures are different

**Solutions:**
1. Check layer configurations match (kernel sizes, channels, etc.)
2. Use `strict=False` (already default in the code)
3. Manually adjust architectures to match

### Problem 3: Weights don't seem to load

**Check:**
1. Is the checkpoint path correct?
2. Does the file exist?
3. Check console output for loading errors
4. Compare first epoch loss with/without pre-training

---

## Understanding the Code

### Key Functions

#### 1. `load_keras_model_weights()`
```python
# Loads your Keras model and extracts all weight matrices
# Returns: Dictionary with layer names and their weights
```

#### 2. `convert_conv2d_weights()`
```python
# Converts Keras Conv2D format to PyTorch Conv2d format
# Handles dimension transposition: (H,W,in,out) → (out,in,H,W)
```

#### 3. `load_pretrained_weights()`
```python
# Loads converted weights into your MAML model
# Handles mismatches gracefully with strict=False
```

---

## Advanced Usage

### Custom Layer Mapping

If automatic conversion doesn't work, you can manually specify layer mapping:

```python
# In keras_to_pytorch_converter.py, modify:
layer_mapping = {
    'keras_conv1': 'net.vars.0',
    'keras_conv2': 'net.vars.2',
    # ... add your mappings
}
```

### Partial Weight Transfer

Load only specific layers:

```python
checkpoint = torch.load('converted_weights.pth.tar')
state_dict = checkpoint['state_dict']

# Load only first 3 conv layers
model_dict = maml.state_dict()
partial_dict = {k: v for k, v in state_dict.items() 
                if k.startswith('vars.0') or k.startswith('vars.2')}
model_dict.update(partial_dict)
maml.load_state_dict(model_dict)
```

---

## Comparison: Random vs Pre-trained Initialization

### Expected Results:

| Metric | Random Init | Pre-trained Init | Improvement |
|--------|-------------|------------------|-------------|
| Initial Loss | ~0.5-1.0 | ~0.05-0.2 | 5-10x better |
| Epochs to Converge | 2000-3000 | 500-1000 | 2-3x faster |
| Final Performance | Baseline | +10-20% | Better generalization |

---

## References

### Scientific Background:

1. **Transfer Learning**: Using knowledge from one task to improve learning on another
2. **Meta-Learning (MAML)**: Learning to learn - adapting quickly to new tasks
3. **Channel Estimation**: Predicting wireless channel properties from pilot signals

### Why This Works:

- ChannelNet learns general features about wireless channels
- MAML adapts these features to specific channel conditions
- Pre-training provides a better "prior" for meta-learning

---

## Questions?

**Q: Will this work if ChannelNet and MAML have different architectures?**
A: Yes, partially. Matching layers will transfer, others use random initialization.

**Q: Should I use a lower learning rate with pre-trained weights?**
A: Yes, often a good idea. Try reducing meta_lr by 2-5x.

**Q: Can I fine-tune only certain layers?**
A: Yes! Freeze early layers and only update later ones:
```python
for param in maml.net.vars[:10]:  # Freeze first 10 parameters
    param.requires_grad = False
```

**Q: How do I know if transfer learning helped?**
A: Compare training curves with and without pre-trained weights. You should see:
- Lower initial loss
- Faster convergence
- Better final performance

---

## Summary

✅ **What we did:**
1. Created a converter: Keras → PyTorch
2. Modified MAML trainer to load pre-trained weights
3. Added command-line option `--pretrained_weights`

✅ **Benefits:**
- Faster training
- Better performance
- More stable learning

✅ **How to use:**
```bash
# Convert
python keras_to_pytorch_converter.py --keras_model model.h5 --output init.pth.tar

# Train
python MAML_trainer_with_tracking.py --pretrained_weights init.pth.tar
```

🎓 **Key Concept**: Transfer learning across frameworks is possible because neural network knowledge is stored in numerical weight matrices, not in framework-specific code!

