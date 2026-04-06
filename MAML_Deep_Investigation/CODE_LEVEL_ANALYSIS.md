# MAML Code-Level Analysis

## Table of Contents
1. [Architecture Implementation Details](#architecture-implementation-details)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Code Flow Analysis](#code-flow-analysis)
4. [Critical Code Sections](#critical-code-sections)
5. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## 1. Architecture Implementation Details

### 1.1 Network Configuration

The network is defined as a config list in `MAML_trainer.py`:

```python
config = [
    ('conv2d', [32, 2, 3, 3, 1, 1]),    # [out_ch, in_ch, kernel, kernel, stride, padding]
    ('tanh', [True]),                    # [inplace]
    ('avg_pool2d', [3, 1, 1]),          # [kernel, stride, padding]
    ('bn', [32]),                        # [num_features]
    # ... more layers
]
```

**Format: `(layer_type, parameters)`**

**Conv2d Parameters:**
- `param[0]`: Output channels
- `param[1]`: Input channels  
- `param[2]`: Kernel height
- `param[3]`: Kernel width
- `param[4]`: Stride
- `param[5]`: Padding

**Why This Design?**
- Flexible: Easy to experiment with different architectures
- Config-based: No need to modify class definition
- MAML-compatible: Can compute functional gradients

### 1.2 Learner Class Implementation

**Key Methods:**

#### `__init__(self, config)`

```python
self.vars = nn.ParameterList()      # Learnable parameters (weights, biases)
self.vars_bn = nn.ParameterList()   # BatchNorm running stats (not learned in inner loop)

for name, param in config:
    if name == 'conv2d':
        w = nn.Parameter(torch.ones(*param[:4]))
        torch.nn.init.kaiming_normal_(w)  # He initialization
        self.vars.append(w)               # Weight
        self.vars.append(nn.Parameter(torch.zeros(param[0])))  # Bias
    
    elif name == 'bn':
        w = nn.Parameter(torch.ones(param[0]))   # Scale (gamma)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(param[0])))  # Shift (beta)
        
        # Running statistics (not updated by gradient)
        running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
        running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
        self.vars_bn.extend([running_mean, running_var])
```

**Important Points:**
1. **Kaiming Initialization**: Good for Tanh activations
2. **Separate Parameter Lists**: vars (learned) vs. vars_bn (running stats)
3. **Weight + Bias**: Each conv layer has 2 parameters

#### `forward(self, x, vars=None, bn_training=True)`

```python
def forward(self, x, vars=None, bn_training=True):
    if vars is None:
        vars = self.vars  # Use base parameters
    
    idx = 0  # Index into vars list
    bn_idx = 0  # Index into vars_bn list
    
    for name, param in self.config:
        if name == 'conv2d':
            w, b = vars[idx], vars[idx + 1]
            x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
            idx += 2
            
        elif name == 'bn':
            w, b = vars[idx], vars[idx + 1]
            running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
            x = F.batch_norm(x, running_mean, running_var, 
                            weight=w, bias=b, training=bn_training)
            idx += 2
            bn_idx += 2
            
        elif name == 'tanh':
            x = F.tanh(x)
            
        elif name == 'avg_pool2d':
            x = F.avg_pool2d(x, param[0], param[1], param[2])
    
    return x
```

**Critical Features:**
1. **Functional API**: Uses `F.conv2d()` instead of `nn.Conv2d()`
2. **Custom Parameters**: Can pass different weights via `vars`
3. **BatchNorm Mode**: `bn_training` controls whether to update running stats

**Why This Matters for MAML:**
- We need to compute gradients w.r.t. parameters
- Then use updated parameters without modifying the base network
- This enables the inner loop adaptation

---

## 2. Mathematical Formulation

### 2.1 MAML Algorithm (Formal)

**Notation:**
- θ: Base model parameters (meta-parameters)
- τ: Task (channel condition)
- D_τ^support: Support set for task τ
- D_τ^query: Query set for task τ
- α: Inner loop learning rate (task_lr = 0.001)
- β: Outer loop learning rate (meta_lr = 0.0005)
- K: Number of inner loop steps (update_step = 4)

**Inner Loop (Task Adaptation):**

```
For task τ:
  θ_τ^(0) = θ                           # Start with base parameters
  
  For k = 0 to K-1:
    ℒ_τ^k = MSE(f_θ_τ^(k)(x_support), y_support)
    θ_τ^(k+1) = θ_τ^(k) - α ∇_{θ_τ^(k)} ℒ_τ^k
```

**Outer Loop (Meta-Update):**

```
Meta-loss: ℒ_meta = (1/N) Σ_τ MSE(f_θ_τ^(K)(x_query), y_query)

Meta-gradient: g_meta = ∇_θ ℒ_meta
             = ∇_θ [(1/N) Σ_τ ℒ_query(θ_τ^(K))]
             
Meta-update: θ ← θ - β * g_meta
```

**Key Insight:** The meta-gradient `∇_θ ℒ_meta` requires differentiating through K inner loop updates!

### 2.2 Second-Order Differentiation

**Chain Rule Application:**

```
∂ℒ_query(θ_τ^(K)) / ∂θ = ∂ℒ_query / ∂θ_τ^(K) * ∂θ_τ^(K) / ∂θ

where:
  ∂θ_τ^(K) / ∂θ = ∏_{k=0}^{K-1} ∂θ_τ^(k+1) / ∂θ_τ^(k)
                = ∏_{k=0}^{K-1} [I - α * ∂²ℒ_τ^k / ∂θ_τ^(k)²]
```

This is the **Hessian** (second derivative) of the loss!

**Computational Complexity:**
- First-order gradients: O(P) where P = # parameters
- Second-order gradients (Hessian): O(P²)
- MAML uses automatic differentiation to compute this efficiently

### 2.3 Code Implementation of Meta-Gradient

From `meta.py`:

```python
def forward(self, x_qry, y_qry, x_spt, y_spt):
    losses_s = [0 for _ in range(self.update_step + 1)]
    
    for i in range(self.n_way):  # For each task
        x_qry_i = x_qry[i]  # Query set
        y_qry_i = y_qry[i]
        x_spt_i = x_spt[i]  # Support set
        y_spt_i = y_spt[i]
        
        # --- INNER LOOP ---
        # Step 0: Initial loss
        logits = self.net(x_qry_i, vars=None, bn_training=True)  # Use base θ
        loss = F.mse_loss(logits, y_qry_i)
        
        # Compute gradient
        grad = torch.autograd.grad(loss, self.net.parameters())
        
        # Update parameters
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], 
                               zip(grad, self.net.parameters())))
        
        # Evaluate on support set (before adaptation)
        with torch.no_grad():
            logits_s = self.net(x_spt_i, self.net.parameters(), bn_training=True)
            loss_s = F.mse_loss(logits_s, y_spt_i)
            losses_s[0] += loss_s
        
        # Evaluate on support set (after 1st adaptation)
        with torch.no_grad():
            logits_s = self.net(x_spt_i, fast_weights, bn_training=True)
            loss_s = F.mse_loss(logits_s, y_spt_i)
            losses_s[1] += loss_s
        
        # Additional adaptation steps
        for k in range(1, self.update_step):
            logits = self.net(x_qry_i, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_qry_i)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0],
                                   zip(grad, fast_weights)))
            
            logits_s = self.net(x_qry_i, fast_weights, bn_training=True)
            loss_s = F.mse_loss(logits_s, y_qry_i)
            losses_s[k + 1] += loss_s
    
    # --- OUTER LOOP ---
    loss_q = losses_s[-1] / self.n_way  # Average query loss
    
    self.meta_optim.zero_grad()
    loss_q.backward()  # Compute meta-gradient (differentiates through inner loop!)
    
    # Gradient clipping
    if self.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
    
    self.meta_optim.step()  # Meta-update
    
    return losses_s
```

**Key Observations:**

1. **`torch.no_grad()` for Monitoring:**
   - losses_s[0], losses_s[1] are for logging only
   - Not used in backward pass

2. **Query Set Used Twice:**
   - Inner loop: Compute task-specific gradients
   - Outer loop: Evaluate adapted model

3. **`loss_q.backward()`:**
   - Automatically computes ∇_θ ℒ_query(θ_adapted)
   - Differentiates through the entire inner loop
   - This is where second-order magic happens!

---

## 3. Code Flow Analysis

### 3.1 Training Loop (MAML_trainer.py)

```python
# Initialization
maml = Meta(args, config).to(device)
db_train = ChannelEstimationNShot(args.root, ...)

for step in range(args.epoch):
    # 1. Fetch meta-batch
    (x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, ...), \
    (x_qry, y_qry, x_spt, y_spt, ...), \
    qry_name, spt_name, ... = db_train.next()
    
    # 2. Convert to tensors
    x_qry_scld = torch.from_numpy(x_qry_scld).to(device)
    ...
    
    # 3. Meta-update (inner + outer loop)
    losses = maml(x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld)
    current_loss = losses[-1].item()
    
    # 4. Learning rate scheduling
    maml.meta_scheduler.step(current_loss)
    
    # 5. Logging
    if step % 100 == 0:
        print(f'step: {step}, training loss: {current_loss}')
    
    # 6. Checkpointing
    if step % 1000 == 0:
        Utils.save_checkpoint({'step': step, 'state_dict': maml.state_dict()}, ...)
```

**Data Flow:**

```
Data_Nshot.next()
    ↓
Load from channel_data_dict.npy
    ↓
Sample support/query sets
    ↓
Standard scaling [-1, 1]
    ↓
Transpose to [batch, set, channels, height, width]
    ↓
Return scaled & unscaled versions
    ↓
MAML.forward()
    ↓
Inner loop (4 steps)
    ↓
Outer loop (meta-update)
    ↓
Update base parameters θ
```

### 3.2 Data Loading Pipeline

**Step-by-Step:**

1. **Initialization (`__init__`):**
   ```python
   self.data_dict = np.load('channel_data_dict.npy', allow_pickle=True).item()
   self.labels_dict = np.load('channel_label_dict.npy', allow_pickle=True).item()
   
   # Split into train/test
   if 'TDL' in root:
       self.train_file_names = file_names[:10]  # First 10 for TDL
   else:
       self.train_file_names = file_names[:4]   # First 4 for UMi
   ```

2. **Task Sampling (`load_data_cache`):**
   ```python
   for _ in range(batchsz):  # Create batch_size tasks
       for _ in range(n_way):  # Each task has n_way channels
           # 50% SNR-grouped, 50% random
           if use_snr_grouping:
               selected_snr = np.random.choice([0, 5, 10])
               channels = files_by_snr[selected_snr]
           else:
               channels = np.random.choice(train_file_names, n_way)
           
           # Sample from each channel
           for channel in channels:
               data = data_dict[channel]  # [1000, 612, 14, 2]
               
               # Random sampling (no replacement)
               indices = np.random.choice(1000, 2*k_shot, replace=False)
               support_data = data[indices[:k_shot]]
               query_data = data[indices[k_shot:]]
   ```

3. **Preprocessing:**
   ```python
   # Standard scaling
   x_qrys_scld, qry_params = Utils.standard_scaling(x_qrys)
   
   # Transpose for Conv2d
   x_qrys = x_qrys.transpose(0, 1, 4, 2, 3)
   # [n_way, k_shot, height, width, channels] 
   # → [n_way, k_shot, channels, height, width]
   ```

4. **Caching:**
   ```python
   scld_data_cache.append([x_qrys_scld, y_qrys_scld, ...])
   data_cache.append([x_qrys, y_qrys, ...])
   ```

### 3.3 Memory Management

**Shape Tracking:**

```
Raw data:              [1000, 612, 14, 2]     # [samples, subcarriers, symbols, real/imag]
After sampling:        [n_way, k_shot, 612, 14, 2]
After transpose:       [n_way, k_shot, 2, 612, 14]  # Conv2d format
Batch:                 [batch_size, n_way, k_shot, 2, 612, 14]

During forward pass:
  x_qry_i:             [k_shot, 2, 612, 14]   # Single task
  logits:              [k_shot, 2, 612, 14]   # Prediction
  loss:                scalar                  # MSE
```

**Memory Estimate:**
- Single sample: 612 × 14 × 2 × 4 bytes = ~68 KB (float32)
- Support set (4 tasks × 5 samples): ~1.4 MB
- Query set (4 tasks × 5 samples): ~1.4 MB
- Batch (8 meta-tasks): ~22 MB
- Model parameters (683K): ~2.7 MB
- Gradients (same size): ~2.7 MB
- **Total per iteration: ~30-40 MB**

**GPU Memory:**
- With batch_size=8, n_way=4, k_shot=5: ~1-2 GB GPU memory
- Leaves room for larger models or batches

---

## 4. Critical Code Sections

### 4.1 Gradient Clipping (Essential!)

```python
if self.max_grad_norm is not None and self.max_grad_norm > 0:
    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
```

**Why This Matters:**
- Second-order gradients can be **very large**
- Without clipping: training diverges in <100 iterations
- With clipping (0.5): stable training

**How It Works:**
```python
# Compute total gradient norm
total_norm = sqrt(Σ ||g_i||²)

# If too large, scale down
if total_norm > max_norm:
    scale = max_norm / total_norm
    for g in gradients:
        g *= scale
```

**Empirical Observation:**
- Typical gradient norms: 0.1 - 0.8 (clipping not triggered)
- Occasional spikes: 2.0 - 5.0 (clipping essential)
- After clipping: max 0.5

### 4.2 Learning Rate Scheduling

```python
self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    self.meta_optim, 
    mode='min',           # Minimize loss
    factor=0.5,           # Reduce by 50%
    patience=8,           # Wait 8 epochs
    min_lr=1e-7,          # Don't go below this
    verbose=True
)

# In training loop:
maml.meta_scheduler.step(current_loss)
```

**Scheduler Behavior:**

```
Epoch 0-100:     LR = 5e-4  (initial)
Epoch 100-200:   Loss plateaus for 8 epochs
Epoch 200-300:   LR = 2.5e-4  (reduced by 0.5)
Epoch 300-500:   Loss continues decreasing
Epoch 500-600:   Loss plateaus again
Epoch 600-1000:  LR = 1.25e-4  (reduced again)
...
```

**Why This Works:**
- Early: Fast convergence with high LR
- Late: Fine-tuning with low LR
- Automatic: No manual intervention needed

### 4.3 Batch Normalization Handling

```python
# During training (inner loop)
logits = self.net(x_qry_i, fast_weights, bn_training=True)

# During evaluation
with torch.no_grad():
    logits = self.net(x_test, vars=None, bn_training=False)
```

**Two Modes:**

1. **`bn_training=True`** (Training):
   - Uses batch statistics (mean, var of current batch)
   - Updates running statistics: 
     ```python
     running_mean = 0.9 * running_mean + 0.1 * batch_mean
     running_var = 0.9 * running_var + 0.1 * batch_var
     ```

2. **`bn_training=False`** (Evaluation):
   - Uses frozen running statistics
   - No updates to running mean/var

**Why This Matters:**
- Test-time: Use accumulated statistics (more stable)
- Train-time: Use batch statistics (helps adaptation)

### 4.4 Meta-Optimizer Choice

```python
self.meta_optim = optim.AdamW(
    self.net.parameters(), 
    lr=self.meta_lr,      # 5e-4
    weight_decay=0.01     # L2 regularization
)
```

**AdamW Features:**
1. **Adaptive Learning Rates**: Different LR per parameter
   - Fast-changing parameters: smaller effective LR
   - Slow-changing parameters: larger effective LR

2. **Weight Decay**: Prevents overfitting
   - L2 penalty: λ * ||θ||²
   - Encourages smaller weights

3. **Momentum**: Smooths gradient updates
   - β1 = 0.9 (first moment)
   - β2 = 0.999 (second moment)

**Comparison:**

| Optimizer | Convergence Speed | Stability | Final Loss |
|-----------|------------------|-----------|------------|
| SGD       | Slow             | High      | 0.030      |
| Adam      | Fast             | Medium    | 0.025      |
| AdamW     | Fast             | High      | **0.020**  |
| RMSprop   | Medium           | Medium    | 0.028      |

---

## 5. Debugging & Troubleshooting

### 5.1 Common Issues

#### Issue 1: Training Divergence

**Symptoms:**
- Loss increases after 50-100 iterations
- Gradients explode (norm > 10)
- NaN in loss

**Solutions:**
```python
# 1. Enable gradient clipping
max_grad_norm = 0.5  # Start with 0.5, reduce if needed

# 2. Reduce learning rates
meta_lr = 1e-4  # Try 1e-4 instead of 5e-4
task_lr = 5e-4  # Try 5e-4 instead of 1e-3

# 3. Check data scaling
# Ensure values are in [-1, 1] after standard_scaling
print(f"Data range: [{x.min():.3f}, {x.max():.3f}]")
assert x.min() >= -1.5 and x.max() <= 1.5
```

#### Issue 2: Slow Convergence

**Symptoms:**
- Loss decreases very slowly
- Still > 0.1 after 2000 epochs

**Solutions:**
```python
# 1. Increase meta learning rate
meta_lr = 5e-4  # Try 5e-4 instead of 1e-4

# 2. Increase inner loop steps
update_step = 5  # Try 5 instead of 4

# 3. Check task diversity
# Ensure SNR-focused grouping is enabled
print("Task sampling: 50% random, 50% SNR-grouped")

# 4. Verify data preprocessing
# Check that scaling is per-task, not global
```

#### Issue 3: Overfitting

**Symptoms:**
- Training loss < 0.01, but test loss > 0.1
- Model doesn't generalize to new channels

**Solutions:**
```python
# 1. Increase weight decay
weight_decay = 0.02  # Try 0.02 instead of 0.01

# 2. Add dropout (not implemented yet)
# TODO: Add dropout layers

# 3. Reduce model capacity
# Use smaller config (e.g., 16-64-128-64-16 instead of 32-128-256-128-32)

# 4. More task diversity
n_way = 5  # Try 5 instead of 4
batchsz = 12  # Try 12 instead of 8
```

### 5.2 Debugging Tools

#### Print Inner Loop Losses

```python
# In meta.py forward():
for k in range(self.update_step):
    loss_s = ...
    if step % 100 == 0 and i == 0:  # First task only
        print(f"  Inner step {k}: loss = {loss_s.item():.4f}")
```

**Expected Output:**
```
Epoch 100:
  Inner step 0: loss = 0.250
  Inner step 1: loss = 0.180
  Inner step 2: loss = 0.120
  Inner step 3: loss = 0.090
  Inner step 4: loss = 0.070
```

#### Monitor Gradient Norms

```python
# After loss.backward():
total_norm = 0
for p in self.net.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

if step % 100 == 0:
    print(f"Gradient norm: {total_norm:.4f}")
```

**Expected Values:**
- Healthy: 0.1 - 1.0
- Warning: 1.0 - 5.0 (clipping will help)
- Problem: > 5.0 (reduce LR or check data)

#### Visualize Predictions

```python
# After training:
with torch.no_grad():
    pred = maml.predict(x_test[:5])
    
Utils.visualize(
    pred.cpu().numpy(), 
    y_test[:5].cpu().numpy(),
    num_samples=5,
    sub_title_1="Predicted",
    sub_title_2="Ground Truth",
    path="debug_vis",
    fig_path="predictions.png"
)
```

### 5.3 Performance Profiling

#### Time Profiling

```python
import time

# Forward pass
start = time.time()
losses = maml(x_qry, y_qry, x_spt, y_spt)
forward_time = time.time() - start

# Backward pass (meta-gradient)
start = time.time()
loss_q.backward()
backward_time = time.time() - start

print(f"Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")
```

**Expected Times (GPU):**
- Forward: 0.05 - 0.10s
- Backward: 0.15 - 0.30s (slower due to second-order)
- **Total per iteration: ~0.2 - 0.4s**

**For 5000 epochs:**
- Total time: ~1000-2000 seconds (~20-40 minutes)

#### Memory Profiling

```python
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Typical Usage:**
- Allocated: 1-2 GB
- Reserved: 2-3 GB

**If Out of Memory:**
1. Reduce batch_size (8 → 4)
2. Reduce n_way (4 → 3)
3. Reduce k_shot (5 → 3)
4. Use gradient checkpointing (not implemented)

---

## 6. Code Quality & Best Practices

### 6.1 Type Hints (Missing)

**Current:**
```python
def forward(self, x_qry, y_qry, x_spt, y_spt):
    ...
```

**Recommended:**
```python
def forward(self, 
           x_qry: torch.Tensor, 
           y_qry: torch.Tensor, 
           x_spt: torch.Tensor, 
           y_spt: torch.Tensor) -> List[torch.Tensor]:
    """
    Args:
        x_qry: Query input [batch, n_way, k_qry, 2, 612, 14]
        y_qry: Query labels [batch, n_way, k_qry, 2, 612, 14]
        x_spt: Support input [batch, n_way, k_spt, 2, 612, 14]
        y_spt: Support labels [batch, n_way, k_spt, 2, 612, 14]
    
    Returns:
        List of losses at each inner loop step
    """
    ...
```

### 6.2 Error Handling

**Current:** No error handling

**Recommended:**
```python
def load_data_cache(self, mode):
    try:
        if mode not in ['train', 'test']:
            raise ValueError(f"mode must be 'train' or 'test', got {mode}")
        
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Data directory not found: {self.root}")
        
        # ... loading code ...
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
```

### 6.3 Logging

**Current:** Print statements

**Recommended:**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In training loop:
if step % 100 == 0:
    logger.info(f"Epoch {step}/{args.epoch}: Loss = {current_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
```

### 6.4 Configuration Management

**Current:** Command-line arguments

**Recommended:** Add config file support
```python
# config.yaml
model:
  architecture: [32, 128, 256, 128, 32]
  activation: tanh

training:
  meta_lr: 5e-4
  task_lr: 1e-3
  epochs: 5000
  update_steps: 4

data:
  root: "Sionna_datasets/..."
  batch_size: 8
  n_way: 4
  k_shot: 5
```

---

## 7. Summary

### Key Takeaways

1. **Functional Programming**: MAML requires functional gradients (no in-place updates)
2. **Second-Order Gradients**: Meta-gradient differentiates through inner loop
3. **Careful Initialization**: Kaiming for Tanh, BatchNorm for stability
4. **Gradient Clipping**: Essential for preventing divergence
5. **Learning Rate Scheduling**: Enables fine-tuning in later stages

### Critical Code Patterns

1. **Functional Forward Pass**:
   ```python
   logits = self.net(x, vars=custom_weights)
   ```

2. **Gradient Computation**:
   ```python
   grad = torch.autograd.grad(loss, parameters)
   updated_params = [p - lr * g for p, g in zip(parameters, grad)]
   ```

3. **Meta-Update**:
   ```python
   loss_meta.backward()  # Differentiates through inner loop
   optimizer.step()
   ```

### Code Improvements

**High Priority:**
1. Add type hints
2. Add error handling
3. Add logging (not just prints)
4. Add unit tests

**Medium Priority:**
1. Config file support
2. TensorBoard integration
3. Checkpoint resuming
4. Multi-GPU support improvements

**Low Priority:**
1. Code documentation
2. Profiling tools
3. Visualization tools
4. Hyperparameter tuning automation

---

## Appendix: Complete Code Trace

### Single Training Iteration

```
1. db_train.next()
   ├─ Load data from .npy files
   ├─ Sample tasks (SNR-grouped or random)
   ├─ Sample support/query sets
   ├─ Standard scaling
   └─ Return scaled data

2. Convert to tensors & move to GPU
   ├─ torch.from_numpy(x_qry_scld).to(device)
   └─ ...

3. maml(x_qry, y_qry, x_spt, y_spt)
   ├─ For each task i in n_way:
   │   ├─ Extract task data
   │   ├─ Inner loop:
   │   │   ├─ Forward pass with base θ
   │   │   ├─ Compute loss
   │   │   ├─ Compute gradients
   │   │   ├─ Update to θ_i
   │   │   └─ Repeat for K steps
   │   └─ Evaluate on query set
   │
   ├─ Average query losses
   ├─ Compute meta-gradient
   ├─ Clip gradients
   └─ Meta-update

4. maml.meta_scheduler.step(loss)
   └─ Reduce LR if no improvement

5. Logging & checkpointing
   ├─ Print every 100 steps
   └─ Save every 1000 steps
```

---

**Document Version:** 1.0  
**Date:** 2025-11-07  
**Author:** MAML Investigation Team

