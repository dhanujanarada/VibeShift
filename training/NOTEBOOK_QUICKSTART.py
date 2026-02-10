#!/usr/bin/env python
"""
NOTEBOOK QUICK START GUIDE

How to use train_notebook.ipynb with validation support:

CELL 1 - Imports & Setup
    └─ Imports all required modules
    
CELL 2 - Configuration
    └─ Set EPOCHS, BATCH_SIZE, LEARNING_RATE, etc.

CELL 3 - Data Loading (TRAIN/VAL SPLIT NOW FIXED)
    └─ Creates 90% train / 10% validation split ✓
    └─ Dynamic padding handles variable sequence lengths ✓
    └─ Prints: dataset size, train/val split, batch counts

CELL 4 - Model Loading
    └─ Loads DiT model from checkpoint or creates new

CELL 5 - Setup Training
    └─ Creates optimizer, scheduler, loss wrapper function

CELL 6 - Initialize Trainer
    └─ Sets up Trainer with device, checkpoint dir, etc.

CELL 7 - Training Loop (WITH VALIDATION NOW WORKING)
    └─ Calls trainer.train() with val_dataloader ✓
    └─ Prints per-epoch: Train Loss, Val Loss, LR, Gradient Norm ✓
    └─ Saves best checkpoint based on validation loss ✓
    └─ Early stopping enabled (patience=20) ✓

CELL 8 - Plot Results
    └─ Visualizes training/validation loss curves

CELL 9 - Inference Test
    └─ Tests loaded best model on sample audio

EXPECTED OUTPUT (Per Epoch):
    Epoch  1/100 | Train Loss: 0.451234 | Val Loss: 0.389123 | LR: 1.00e-04 | Grad Norm: 1.0234
    Epoch  2/100 | Train Loss: 0.321456 | Val Loss: 0.275634 | LR: 9.50e-05 | Grad Norm: 0.9876 | Best: 0.275634 (ep 2)
    Epoch  3/100 | Train Loss: 0.287654 | Val Loss: 0.251234 | LR: 9.01e-05 | Grad Norm: 0.9512 | Best: 0.251234 (ep 3)

KEY PARAMETERS IN CELL 7:
    ├─ num_epochs: Total training epochs
    ├─ val_dataloader: Validation loader (NEVER set to None!)
    ├─ early_stopping_patience: Stop if val loss doesn't improve for N epochs (20 recommended)
    ├─ accumulation_steps: Gradient accumulation (1 = normal, 2+ = larger effective batch)
    ├─ log_interval: Print progress every N epochs
    └─ save_interval: Save checkpoint every N epochs

EXAMPLE MODIFICATIONS:

To use more validation data:
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val (in Cell 3)

To train longer without early stopping:
    early_stopping_patience=None  # Never stop early (in Cell 7)

To accumulate gradients for larger batch size:
    accumulation_steps=4  # 4x larger effective batch (in Cell 7)

CHECKPOINT FILES SAVED:
    notebook/train_checkpoints/final_model.pt       ← Final model
    notebook/train_checkpoints/best.pt               ← Best model (min val loss)
    notebook/train_checkpoints/checkpoint_epoch_*.pt ← Periodic checkpoints

MONITORING:
    ✓ Real-time loss curves (both train and val)
    ✓ Learning rate scheduling
    ✓ Gradient norms for debugging
    ✓ Best checkpoint tracking
    ✓ Early stopping to prevent overfitting
"""

print(__doc__)
