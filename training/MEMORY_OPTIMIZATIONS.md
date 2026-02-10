# Memory Optimization Applied ✓

## Primary Fix: Flash Attention

**Updated:** `utills/embedding.py` - RoPEEmbedding class

**Change:** Replaced manual attention computation with `F.scaled_dot_product_attention`

```python
# OLD (Memory intensive):
attn = (q @ k.transpose(-2, -1)) * self.scale
attn = attn.softmax(dim=-1)
attn = self.dropout(attn)
out = (attn @ v)

# NEW (Flash Attention - much more efficient):
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attn_mask,
    dropout_p=self.dropout.p if self.training else 0.0,
    scale=self.scale,
)
```

**Memory Savings:** 
- Flash Attention reduces memory from O(N²) to O(N) for sequence length N
- For your 2794 token sequences: ~3.72 GiB → ~1.3 GiB (estimated 65% reduction)
- Automatically uses optimized CUDA kernels when available

## Additional Optimizations (if still needed)

### 1. Gradient Checkpointing (Most Effective)
Add to your DiT model initialization in the notebook:

```python
# Cell 4 - After loading model
from torch.utils.checkpoint import checkpoint_sequential

# Enable gradient checkpointing for DiT blocks
model.dit.blocks = checkpoint_sequential(
    model.dit.blocks, 
    segments=len(model.dit.blocks) // 2  # Trade compute for memory
)
```

**Memory Savings:** 40-60% reduction, 20-30% slower training

### 2. Reduce Batch Size (Easiest)
In Cell 3:

```python
BATCH_SIZE = 8  # Down from 16
# or
BATCH_SIZE = 4  # For very tight memory
```

**Memory Savings:** 50% reduction (8 vs 16), linear scaling

### 3. Increase Gradient Accumulation (Maintain effective batch size)
In Cell 7:

```python
accumulation_steps=8,  # Up from 4
# Effective batch = BATCH_SIZE * accumulation_steps
# 8 * 8 = 64 (same as 16 * 4)
```

**Memory Savings:** Same effective batch size with smaller actual batches

### 4. Mixed Precision (Already enabled if use_amp=True)
Verify in Cell 6:

```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device=device,
    use_amp=True,  # ← Should be True for CUDA
    ...
)
```

**Memory Savings:** ~30-40% reduction from FP32 → FP16

### 5. PyTorch Memory Allocator (Environment Variable)
Set before running:

```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

Or in terminal:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Effect:** Reduces memory fragmentation

### 6. Clear Cache Between Epochs
Add to Cell 7 training loop:

```python
# After results = trainer.train(...)
import torch
torch.cuda.empty_cache()
```

## Recommended Combination

**For 23 GB GPU with current settings:**

1. ✅ Flash Attention (DONE)
2. Reduce batch size: `BATCH_SIZE = 8`
3. Increase accumulation: `accumulation_steps = 8`
4. Enable gradient checkpointing (if still OOM)

This should reduce peak memory from ~23 GB to ~12-15 GB while maintaining training quality.

## Current Training Config (from error)

```python
BATCH_SIZE = 16  # Consider reducing to 8
accumulation_steps = 4  # Consider increasing to 8
```

**Effective batch size:** 16 × 4 = 64

**Alternative (same effective batch, less memory):**
```python
BATCH_SIZE = 8
accumulation_steps = 8
# Still 8 × 8 = 64 effective batch size
```

## Monitor Memory Usage

```python
import torch

# Check current memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Reset peak stats
torch.cuda.reset_peak_memory_stats()
```
