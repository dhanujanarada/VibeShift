"""
Training module for VibeShift (FIXED VERSION).
Provides training loop, checkpoint management, AMP support, and gradient accumulation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
import json
import warnings
import numpy as np


class Trainer:
    """
    FIXED Trainer class for flow matching model training.
    
    Improvements:
    - Proper batch data unpacking for different formats
    - Flexible scheduler handling (works with all scheduler types)
    - Gradient accumulation support
    - Mixed precision (AMP) support
    - Complete checkpoint save/load with all training state
    - Proper model train/eval state restoration
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
        name: str = "experiment",
        use_amp: bool = False,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The flow matching model to train
            optimizer: Optimizer for the model
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            name: Experiment name for checkpoint naming
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.name = name
        self.use_amp = use_amp
        self.scheduler = None  # FIXED: Store scheduler for checkpointing
        
        # Setup AMP if requested
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("AMP enabled")
        else:
            self.scaler = None
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints")
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        self.checkpoint_dir = checkpoint_dir / name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking metrics
        self.epoch_losses = []
        self.batch_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.gradient_norms = []
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        
        print(f"Trainer initialized for '{name}'")
        print(f"  Device: {device}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    @staticmethod
    def _unpack_batch(batch_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        FIXED: Properly unpack batch data in various formats with optional mask.
        
        Handles:
        - 2-tuple: (x0, x1) -> unpacks with default genre_ids=0, mask=None
        - 3-tuple: (x0, x1, genre_ids) or (x0, x1, mask) -> unpacks appropriately
        - 4-tuple: (x0, x1, genre_ids, mask) -> unpacks all
        
        Returns:
            (x0, x1, genre_ids, mask)
        """
        if not isinstance(batch_data, (tuple, list)):
            raise TypeError(f"Expected tuple/list batch, got {type(batch_data)}")
        
        if len(batch_data) == 2:
            x0, x1 = batch_data
            genre_ids = torch.zeros(x0.size(0), dtype=torch.long, device=x0.device)
            mask = None
        elif len(batch_data) == 3:
            x0, x1, third = batch_data
            # Determine if third element is genre_ids or mask
            if third.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                genre_ids = third
                mask = None
            elif third.dtype == torch.float32:
                # It's a mask
                genre_ids = torch.zeros(x0.size(0), dtype=torch.long, device=x0.device)
                mask = third
            else:
                genre_ids = third
                mask = None
        elif len(batch_data) == 4:
            x0, x1, genre_ids, mask = batch_data
        else:
            raise ValueError(f"Unexpected batch length: {len(batch_data)}")
        
        return x0, x1, genre_ids, mask
    
    def train_epoch(
        self,
        train_dataloader: DataLoader,
        loss_fn: Callable,
        gradient_clip: Optional[float] = None,
        accumulation_steps: int = 1,
        monitor_gradients: bool = False,
    ) -> float:
        """
        Train for one epoch with gradient accumulation support.
        
        Args:
            train_dataloader: DataLoader for training data
            loss_fn: Loss function (receives unpacked x0, x1, genre_ids)
            gradient_clip: Max gradient norm for clipping (None = no clipping)
            accumulation_steps: Gradient accumulation steps (>1 for larger effective batch)
            monitor_gradients: Whether to monitor gradient norms
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            # FIXED: Proper batch unpacking with mask support
            x0, x1, genre_ids, mask = self._unpack_batch(batch_data)
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            genre_ids = genre_ids.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass with AMP if enabled
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = loss_fn(x0, x1, genre_ids, mask)
            else:
                loss = loss_fn(x0, x1, genre_ids, mask)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if gradient_clip is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip)
                
                # Monitor gradients if requested
                if monitor_gradients:
                    grad_norm = self._compute_gradient_norm()
                    self.gradient_norms.append(grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track loss
            batch_loss = loss.item() * accumulation_steps  # Unscale for reporting
            epoch_loss += batch_loss
            self.batch_losses.append(batch_loss)
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        self.epoch_losses.append(avg_epoch_loss)
        self.current_epoch += 1
        
        return avg_epoch_loss
    
    def validate(
        self,
        val_dataloader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """
        Validate the model.
        
        Args:
            val_dataloader: DataLoader for validation data
            loss_fn: Loss function (receives unpacked x0, x1, genre_ids)
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_dataloader:
                # FIXED: Proper batch unpacking with mask support
                x0, x1, genre_ids, mask = self._unpack_batch(batch_data)
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                genre_ids = genre_ids.to(self.device)
                if mask is not None:
                    mask = mask.to(self.device)
                
                # Forward pass with AMP if enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = loss_fn(x0, x1, genre_ids, mask)
                else:
                    loss = loss_fn(x0, x1, genre_ids, mask)
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        loss_fn: Callable,
        val_dataloader: Optional[DataLoader] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        accumulation_steps: int = 1,
        early_stopping_patience: Optional[int] = None,
        log_interval: int = 10,
        monitor_gradients: bool = False,
        save_interval: int = 10,
    ) -> Dict:
        """
        Complete training loop with gradient accumulation and proper scheduling.
        
        Args:
            train_dataloader: Training data
            num_epochs: Number of epochs to train
            loss_fn: Loss function (receives unpacked x0, x1, genre_ids, mask)
            val_dataloader: Validation data (optional)
            scheduler: Learning rate scheduler (optional)
            gradient_clip: Gradient clipping threshold
            accumulation_steps: Gradient accumulation steps
            early_stopping_patience: Stop if val loss doesn't improve for N epochs
            log_interval: Log progress every N epochs
            monitor_gradients: Whether to monitor gradient norms
            save_interval: Save checkpoint every N epochs
        
        Returns:
            Training results dict with metrics
        """
        # FIXED: Store scheduler for checkpointing
        self.scheduler = scheduler
        
        print(f"\n{'='*70}")
        print(f"Training '{self.name}' for {num_epochs} epochs")
        print(f"  Accumulation steps: {accumulation_steps}")
        if self.use_amp:
            print(f"  AMP: Enabled")
        print(f"{'='*70}")
        
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(
                train_dataloader,
                loss_fn,
                gradient_clip=gradient_clip,
                accumulation_steps=accumulation_steps,
                monitor_gradients=monitor_gradients,
            )
            
            # Validate
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader, loss_fn)
            
            # Learning rate scheduling - FIXED: Flexible scheduler handling
            if scheduler is not None:
                # Check if scheduler expects a metric (ReduceLROnPlateau)
                if hasattr(scheduler, 'step'):
                    try:
                        # Try passing val_loss for metric-based schedulers
                        if val_loss is not None and hasattr(scheduler, 'best'):
                            # This is likely ReduceLROnPlateau
                            scheduler.step(val_loss)
                        else:
                            # Epoch-based scheduler
                            scheduler.step()
                    except TypeError:
                        # Fallback to epoch-based if metric not expected
                        scheduler.step()
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)
            
            # Logging - always log epoch completion with all details
            log_str = f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.6f}"
            if val_loss is not None:
                log_str += f" | Val Loss: {val_loss:.6f}"
            log_str += f" | LR: {current_lr:.2e}"
            
            # Add gradient norm info if monitoring
            if monitor_gradients and len(self.gradient_norms) > 0:
                latest_grad_norm = self.gradient_norms[-1]
                log_str += f" | Grad Norm: {latest_grad_norm:.4f}"
            
            # Add best loss info
            if epoch > 0:
                log_str += f" | Best: {self.best_loss:.6f} (ep {self.best_epoch+1})"
            
            print(log_str)

            # Save checkpoint periodically
            if (epoch + 1) % save_interval == 0:
                ckpt_path = self.save_checkpoint(suffix=f"_epoch_{epoch+1}")
                print(f"  [Checkpoint] Saved → {ckpt_path.name}")

            # Early stopping
            if val_loss is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    early_stop_counter = 0
                    # Save best checkpoint
                    ckpt_path = self.save_checkpoint(is_best=True)
                    print(f"  [Best] New best val loss {self.best_loss:.6f} → {ckpt_path.name}")
                else:
                    early_stop_counter += 1
                    
                    if early_stopping_patience is not None:
                        if early_stop_counter >= early_stopping_patience:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                            print(f"Best validation loss: {self.best_loss:.6f} (epoch {self.best_epoch+1})")
                            break
            else:
                # Save best checkpoint based on training loss
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self.best_epoch = epoch
                    self.save_checkpoint(is_best=True)
        
        # Print summary
        self._print_summary()
        
        return {
            "epoch_losses": self.epoch_losses,
            "batch_losses": self.batch_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "gradient_norms": self.gradient_norms,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "final_epoch": self.current_epoch,
        }
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = "") -> Path:
        """
        Save model checkpoint with ALL training state.
        
        Args:
            is_best: Whether this is the best checkpoint so far
            suffix: Additional suffix for checkpoint name
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch_losses": self.epoch_losses,
            "batch_losses": self.batch_losses,
            "val_losses": self.val_losses,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "model_training_state": self.model.training,  # FIXED: Save train/eval state
        }
        
        # FIXED: Save scheduler and scaler state
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"best{suffix}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}{suffix}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """
        Load model checkpoint with proper state restoration.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Epoch at which checkpoint was saved
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # FIXED: Restore train/eval state
        if "model_training_state" in checkpoint:
            if checkpoint["model_training_state"]:
                self.model.train()
            else:
                self.model.eval()
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # FIXED: Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state if available
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.epoch_losses = checkpoint.get("epoch_losses", [])
        self.batch_losses = checkpoint.get("batch_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best loss so far: {self.best_loss:.6f}")
        
        return self.current_epoch
    
    def _compute_gradient_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _print_summary(self):
        """Print training summary."""
        print(f"\n{'='*70}")
        print(f"Training Complete")
        print(f"{'='*70}")
        print(f"\nLoss Statistics:")
        print(f"  Initial loss: {self.epoch_losses[0]:.8f}")
        print(f"  Final loss:   {self.epoch_losses[-1]:.8f}")
        print(f"  Best loss:    {self.best_loss:.8f} (epoch {self.best_epoch+1})")
        
        if self.epoch_losses[0] > 0:
            improvement = (
                (self.epoch_losses[0] - self.epoch_losses[-1]) / self.epoch_losses[0]
            ) * 100
            print(f"  Improvement:  {improvement:.2f}%")
        
        if self.val_losses:
            print(f"\nValidation Statistics:")
            print(f"  Best val loss: {min(self.val_losses):.8f}")
        
        if self.gradient_norms:
            print(f"\nGradient Statistics:")
            print(f"  Average gradient norm: {np.mean(self.gradient_norms):.6f}")
            print(f"  Min gradient norm:     {np.min(self.gradient_norms):.2e}")
            print(f"  Max gradient norm:     {np.max(self.gradient_norms):.2e}")
        
        print(f"{'='*70}\n")
    
    def get_metrics(self) -> Dict:
        """Get all tracked metrics."""
        return {
            "epoch_losses": self.epoch_losses,
            "batch_losses": self.batch_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "gradient_norms": self.gradient_norms,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "current_epoch": self.current_epoch,
        }


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(
        self,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip: Optional[float] = 1.0,
        accumulation_steps: int = 1,
        early_stopping_patience: Optional[int] = None,
        log_interval: int = 10,
        monitor_gradients: bool = False,
        use_amp: bool = torch.cuda.is_available(),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize training configuration.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training (default 32 for RTX 4090; reduce to 16 or 8 if OOM)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization coefficient
            gradient_clip: Max gradient norm for clipping (None to disable)
            accumulation_steps: Gradient accumulation steps
            early_stopping_patience: Stop if val loss doesn't improve for N epochs
            log_interval: Log progress every N epochs
            monitor_gradients: Whether to monitor gradient norms
            use_amp: Whether to use automatic mixed precision (auto-enabled on CUDA)
            device: Device to train on
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.log_interval = log_interval
        self.monitor_gradients = monitor_gradients
        self.use_amp = use_amp
        self.device = device
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "accumulation_steps": self.accumulation_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "log_interval": self.log_interval,
            "monitor_gradients": self.monitor_gradients,
            "use_amp": self.use_amp,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """String representation of config."""
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"TrainingConfig({', '.join(items)})"
