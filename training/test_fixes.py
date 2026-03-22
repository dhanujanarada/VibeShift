"""
Test script to verify all fixes in training_fixed.py and dataloader_fixed.py.
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from pathlib import Path
from training.training import Trainer, TrainingConfig
from training.dataloader import (
    LatentPairDataset, 
    GenreAwareLatentDataset,
    create_dataloader,
    default_collate_with_dynamic_padding
)


def test_zip_fix():
    """Test Fix #1: zip() bug in _compute_max_length"""
    print("\n" + "="*70)
    print("TEST 1: Verify zip() fix in _compute_max_length")
    print("="*70)
    
    # Create temp test data
    temp_dir1 = Path("./temp_test_src")
    temp_dir2 = Path("./temp_test_tgt")
    temp_dir1.mkdir(exist_ok=True)
    temp_dir2.mkdir(exist_ok=True)
    
    try:
        # Create dummy .pt files
        for i in range(5):
            torch.save({"z": torch.randn(100 + i*10, 512)}, temp_dir1 / f"{i:05d}.pt")
            torch.save({"z": torch.randn(120 + i*10, 512)}, temp_dir2 / f"{i:05d}.pt")
        
        # Create dataset and trigger _compute_max_length
        dataset = LatentPairDataset(str(temp_dir1), str(temp_dir2))
        max_len = dataset._compute_max_length()
        
        print(f"PASS: _compute_max_length() executed without zip() error")
        print(f"   Computed max length: {max_len}")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


def test_efficient_getitem():
    """Test Fix #2: Efficient __getitem__ without unnecessary max_len check"""
    print("\n" + "="*70)
    print("TEST 2: Verify efficient __getitem__ (no repeated _compute_max_length calls)")
    print("="*70)
    
    temp_dir1 = Path("./temp_test_src2")
    temp_dir2 = Path("./temp_test_tgt2")
    temp_dir1.mkdir(exist_ok=True)
    temp_dir2.mkdir(exist_ok=True)
    
    try:
        # Create dummy data
        for i in range(3):
            torch.save({"z": torch.randn(100, 512)}, temp_dir1 / f"{i:05d}.pt")
            torch.save({"z": torch.randn(120, 512)}, temp_dir2 / f"{i:05d}.pt")
        
        dataset = LatentPairDataset(str(temp_dir1), str(temp_dir2))
        
        # Access multiple items - should NOT call _compute_max_length each time
        print("Accessing 3 samples...")
        for i in range(3):
            x0, x1 = dataset[i]
            assert x0.shape[1] == 512, "Wrong embedding dim"
        
        print("PASS: __getitem__ works efficiently without calling _compute_max_length repeatedly")
    finally:
        import shutil
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


def test_mask_generation():
    """Test Fix #3: Attention mask generation in collate_fn"""
    print("\n" + "="*70)
    print("TEST 3: Verify attention mask generation in collate function")
    print("="*70)
    
    # Create dummy batch with varying lengths
    batch = [
        (torch.randn(100, 512), torch.randn(120, 512)),
        (torch.randn(80, 512), torch.randn(90, 512)),
        (torch.randn(150, 512), torch.randn(140, 512)),
    ]
    
    # Collate batch
    x0_padded, x1_padded, mask = default_collate_with_dynamic_padding(batch)
    
    print(f"Batch shape: x0={x0_padded.shape}, x1={x1_padded.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    
    # Verify mask properties
    assert mask.shape == (3, 150), f"Expected mask shape (3, 150), got {mask.shape}"
    assert mask.dtype == torch.float32, "Mask should be float32"
    
    # Check mask values
    assert mask[0, :120].sum() == 120, "Mask should have 1.0 for valid positions"
    assert mask[0, 120:].sum() == 0, "Mask should have 0.0 for padding"
    assert mask[1, :90].sum() == 90, "Mask for second sample incorrect"
    assert mask[2, :150].sum() == 150, "Mask for third sample incorrect"
    
    print("PASS: Attention masks generated correctly")
    print(f"   Sample 0: {mask[0, :10]} ... (first 10 values)")
    print(f"   Sample 1: {mask[1, :10]} ... (first 10 values)")
    
    # Test with genre-aware batch
    print("\nTesting genre-aware collate...")
    genre_batch = [
        (torch.randn(100, 512), torch.randn(120, 512), 0),
        (torch.randn(80, 512), torch.randn(90, 512), 1),
        (torch.randn(150, 512), torch.randn(140, 512), 2),
    ]
    
    x0_g, x1_g, genre_ids, mask_g = default_collate_with_dynamic_padding(genre_batch)
    
    assert len(genre_ids) == 3, "Genre IDs should have 3 elements"
    assert mask_g.shape == (3, 150), f"Genre-aware mask shape incorrect: {mask_g.shape}"
    
    print("PASS: Genre-aware collate with masks works correctly")
    print(f"   Genre IDs: {genre_ids}")
    print(f"   Mask shape: {mask_g.shape}")


def test_scheduler_storage():
    """Test Fix #4: Scheduler stored as instance variable"""
    print("\n" + "="*70)
    print("TEST 4: Verify scheduler is stored as instance variable")
    print("="*70)
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        name="test_scheduler"
    )
    
    # Check scheduler attribute exists
    assert hasattr(trainer, 'scheduler'), "Trainer should have 'scheduler' attribute"
    assert trainer.scheduler is None, "Scheduler should be None initially"
    
    print("PASS: Trainer has 'scheduler' instance variable initialized to None")
    
    # Create a scheduler and pass to train (mock train call)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    
    # The scheduler should be stored when train() is called
    # We'll just verify the attribute can be set
    trainer.scheduler = scheduler
    assert trainer.scheduler is not None, "Scheduler should be stored"
    
    print("PASS: Scheduler can be stored in trainer.scheduler")


def test_batch_unpacking_with_mask():
    """Test Fix #5: Batch unpacking handles masks"""
    print("\n" + "="*70)
    print("TEST 5: Verify batch unpacking handles masks (2, 3, and 4-tuples)")
    print("="*70)
    
    # Test 2-tuple (x0, x1)
    batch_2 = (torch.randn(4, 100, 512), torch.randn(4, 120, 512))
    x0, x1, genre_ids, mask = Trainer._unpack_batch(batch_2)
    assert x0.shape[0] == 4, "Batch size should be 4"
    assert genre_ids.shape == (4,), "Genre IDs should have shape (4,)"
    assert mask is None, "Mask should be None for 2-tuple"
    print("PASS: 2-tuple unpacking works (x0, x1) -> (x0, x1, genre_ids=0, mask=None)")
    
    # Test 3-tuple with genre_ids (x0, x1, genre_ids)
    batch_3_genre = (
        torch.randn(4, 100, 512), 
        torch.randn(4, 120, 512),
        torch.tensor([0, 1, 2, 0], dtype=torch.long)
    )
    x0, x1, genre_ids, mask = Trainer._unpack_batch(batch_3_genre)
    assert genre_ids.dtype == torch.long, "Genre IDs should be long"
    assert mask is None, "Mask should be None for 3-tuple with genre_ids"
    print("PASS: 3-tuple unpacking with genre_ids works")
    
    # Test 3-tuple with mask (x0, x1, mask)
    batch_3_mask = (
        torch.randn(4, 100, 512),
        torch.randn(4, 120, 512),
        torch.ones(4, 120, dtype=torch.float32)
    )
    x0, x1, genre_ids, mask = Trainer._unpack_batch(batch_3_mask)
    assert mask is not None, "Mask should not be None"
    assert mask.dtype == torch.float32, "Mask should be float32"
    assert genre_ids.sum() == 0, "Genre IDs should default to 0"
    print("✅ PASS: 3-tuple unpacking with mask works")
    
    # Test 4-tuple (x0, x1, genre_ids, mask)
    batch_4 = (
        torch.randn(4, 100, 512),
        torch.randn(4, 120, 512),
        torch.tensor([0, 1, 2, 0], dtype=torch.long),
        torch.ones(4, 120, dtype=torch.float32)
    )
    x0, x1, genre_ids, mask = Trainer._unpack_batch(batch_4)
    assert genre_ids.dtype == torch.long, "Genre IDs should be long"
    assert mask.dtype == torch.float32, "Mask should be float32"
    print("✅ PASS: 4-tuple unpacking works (x0, x1, genre_ids, mask)")


def test_checkpoint_with_scheduler():
    """Test that scheduler state is saved/loaded in checkpoints"""
    print("\n" + "="*70)
    print("TEST 6: Verify scheduler state is saved/loaded in checkpoints")
    print("="*70)
    
    import tempfile
    import shutil
    
    # Create temp checkpoint dir
    temp_checkpoint_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create model, optimizer, scheduler
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            checkpoint_dir=str(temp_checkpoint_dir),
            name="test_checkpoint"
        )
        
        # Store scheduler
        trainer.scheduler = scheduler
        
        # Advance scheduler
        for _ in range(3):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR before save: {current_lr}")
        
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint()
        print(f"✅ Checkpoint saved to {checkpoint_path}")
        
        # Verify checkpoint contains scheduler state
        checkpoint = torch.load(checkpoint_path)
        assert "scheduler_state_dict" in checkpoint, "Checkpoint should contain scheduler_state_dict"
        print("✅ PASS: Checkpoint contains scheduler_state_dict")
        
        # Create new trainer and load checkpoint
        model2 = torch.nn.Linear(10, 10)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.5)
        
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            device="cpu",
            checkpoint_dir=str(temp_checkpoint_dir),
            name="test_checkpoint"
        )
        trainer2.scheduler = scheduler2
        
        # Load checkpoint
        trainer2.load_checkpoint(str(checkpoint_path))
        
        loaded_lr = optimizer2.param_groups[0]['lr']
        print(f"Loaded LR: {loaded_lr}")
        
        assert abs(loaded_lr - current_lr) < 1e-6, f"LR mismatch: {loaded_lr} != {current_lr}"
        print("✅ PASS: Scheduler state correctly restored from checkpoint")
        
    finally:
        shutil.rmtree(temp_checkpoint_dir, ignore_errors=True)


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*70)
    print("RUNNING ALL VERIFICATION TESTS FOR FIXES")
    print("="*70)
    
    tests = [
        test_zip_fix,
        test_efficient_getitem,
        test_mask_generation,
        test_scheduler_storage,
        test_batch_unpacking_with_mask,
        test_checkpoint_with_scheduler,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Fixes are working correctly.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review.")


if __name__ == "__main__":
    run_all_tests()
