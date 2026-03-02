#!/usr/bin/env python
"""Quick test of training loop with logging"""

import sys, torch, torch.nn as nn
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from training import Trainer
from dataloader import create_dataloader

print("\n" + "="*70)
print("TRAINING SYSTEM TEST")
print("="*70)

# Test 1: Dataloader
print("\n[TEST 1] Loading dataloader...")
try:
    train_loader, train_dataset = create_dataloader(
        source_dir="../data/latent_data/latent_classical",
        target_dir="../data/latent_data/latent_synth1",
        batch_size=4, shuffle=False, num_workers=0,
        max_samples=10, drop_last=False, pin_memory=False,
    )
    info = train_dataset.get_info()
    print(f"  [OK] Dataset: {len(train_dataset)} samples")
    print(f"  [OK] Embedding dim: {info['embedding_dim']}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Test 2: Model
print("\n[TEST 2] Creating model...")
try:
    class SimpleModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x0, x1, genre_ids, mask=None):
            # x0/x1: (batch, seq_len, input_dim)
            x0_mean = x0.mean(dim=1)  # (batch, input_dim)
            x1_mean = x1.mean(dim=1)  # (batch, input_dim)
            # Use the linear layer to create a loss that depends on it
            pred0 = self.linear(x0_mean)  # (batch, 1)
            pred1 = self.linear(x1_mean)  # (batch, 1)
            return torch.nn.functional.mse_loss(pred0, pred1)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel(info['embedding_dim']).to(DEVICE)
    print(f"  [OK] Model created on {DEVICE}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Test 3: Trainer
print("\n[TEST 3] Initializing trainer...")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    
    def loss_fn(x0, x1, genre_ids, mask=None):
        return model(x0, x1, genre_ids, mask)
    
    trainer = Trainer(
        model=model, optimizer=optimizer, device=DEVICE,
        checkpoint_dir="test_checkpoints", name="test_flow",
        use_amp=False,
    )
    print(f"  [OK] Trainer initialized")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Test 4: Training (3 epochs)
print("\n[TEST 4] Running 3-epoch training loop...")
print("         (Observe loss/val loss printing below)\n")

try:
    results = trainer.train(
        train_dataloader=train_loader, num_epochs=3, loss_fn=loss_fn,
        val_dataloader=None, scheduler=scheduler, gradient_clip=1.0,
        accumulation_steps=1, early_stopping_patience=None,
        log_interval=1, monitor_gradients=True, save_interval=2,
    )
    
    print(f"\n  [OK] Training completed!")
    print(f"  [OK] Epochs: {len(results['epoch_losses'])}")
    print(f"  [OK] Best loss: {results['best_loss']:.6f}")
    print(f"  [OK] Learning rates: {len(results['learning_rates'])} tracked")
    print(f"  [OK] Gradient norms: {len(results['gradient_norms'])} tracked")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Checkpoints
print("\n[TEST 5] Validating checkpoints...")
try:
    ckpts = list(Path(trainer.checkpoint_dir).glob("*.pt"))
    print(f"  [OK] Checkpoints created: {len(ckpts)}")
    for c in sorted(ckpts):
        print(f"    - {c.name}")
except Exception as e:
    print(f"  [ERROR] {e}")
    sys.exit(1)

# Cleanup
print("\n[CLEANUP] Removing test_checkpoints...")
shutil.rmtree("test_checkpoints", ignore_errors=True)
print("  [OK] Done")

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nVerified:")
print("  [x] Dataloader working")
print("  [x] Training loop working")
print("  [x] Loss printed each epoch (check above)")
print("  [x] Gradient norms tracked")
print("  [x] Learning rate tracked")
print("  [x] Checkpoint saving working")
print("\n")
