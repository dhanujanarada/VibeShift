

import os
import re
import logging
import torch
from pathlib import Path
import glob
from datetime import datetime

from training.dataloader import DACDataset
from torch.utils.data import DataLoader
from training.training import TrainingPipeline, TrainingConfig




LOG_DIR = Path("logs")
CHECKPOINT_DIR = Path("checkpoints")
ROOT = Path(r"c:\Users\Dhanuja\Desktop\Vibeshift\VibeShift")
DATA_DIR = ROOT / "data" / "output"

NON_ROCK_DIR = DATA_DIR / "non_rock_dac"
ROCK_DIR = DATA_DIR / "rock_dac"




def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logger = logging.getLogger("vibeshift_training")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_data(logger: logging.Logger) -> tuple:
    """Load and validate training data."""
    logger.info("Setting up training data...")
    
    if not NON_ROCK_DIR.exists():
        logger.error(f"Non-rock data directory not found: {NON_ROCK_DIR}")
        raise FileNotFoundError(f"Non-rock data directory not found: {NON_ROCK_DIR}")
    
    if not ROCK_DIR.exists():
        logger.error(f"Rock data directory not found: {ROCK_DIR}")
        raise FileNotFoundError(f"Rock data directory not found: {ROCK_DIR}")
    
    source_files = sorted(glob.glob(str(NON_ROCK_DIR / "*.pt")))
    target_files = sorted(glob.glob(str(ROCK_DIR / "*.pt")))
    
    logger.info(f"Found {len(source_files)} source (non-rock) files")
    logger.info(f"Found {len(target_files)} target (rock) files")
    
    if not source_files or not target_files:
        logger.error("No data files found in the specified directories")
        raise ValueError("No data files found in the specified directories")
    
    return source_files, target_files



def validate_dataset(dataset: DACDataset, logger: logging.Logger) -> None:
    """Validate dataset and log sample information."""
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    try:
        x0, x1 = dataset[0]
        logger.info(f"Sample shapes - x0: {x0.shape}, x1: {x1.shape}")
        logger.info(f"  DAC latent_dim: {x0.shape[-1]}")
        logger.debug(f"  x0 dtype: {x0.dtype}, range: [{x0.min():.4f}, {x0.max():.4f}]")
        logger.debug(f"  x1 dtype: {x1.dtype}, range: [{x1.min():.4f}, {x1.max():.4f}]")
    except Exception as e:
        logger.error(f"Failed to load sample from dataset: {e}", exc_info=True)
        raise




def setup_model(logger: logging.Logger) -> tuple:
    """Initialize training configuration and pipeline."""
    logger.info("Initializing training configuration...")
    config = TrainingConfig()
    logger.debug(f"Config: {config.__dict__}")
    
    logger.info("Creating training pipeline...")
    pipeline = TrainingPipeline(config)
    logger.info("Training pipeline initialized successfully")
    
    return config, pipeline




def run_training(
    pipeline: TrainingPipeline,
    loader: DataLoader,
    logger: logging.Logger,
    start_epoch: int = 1,
    best_loss: float | None = None,
) -> list:
    """Execute training loop."""
    logger.info("=" * 60)
    logger.info("Starting training loop")
    logger.info("=" * 60)
    
    try:
        losses = pipeline.train(loader, start_epoch=start_epoch, best_loss=best_loss)
        logger.info("Training completed successfully")
        logger.info(f"Training losses: {losses}")
        logger.info(f"Final loss: {losses[-1]:.4f}")
        
        if len(losses) > 1:
            improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
            logger.info(f"Loss improvement: {improvement:.2f}%")
        
        return losses
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def get_latest_checkpoint(logger: logging.Logger) -> Path | None:
    """Get latest checkpoint file based on epoch in filename."""
    if not CHECKPOINT_DIR.exists():
        logger.info(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return None

    candidates: list[tuple[int, Path]] = []
    for ckpt in CHECKPOINT_DIR.glob("checkpoint_epoch_*.pt"):
        match = re.search(r"checkpoint_epoch_(\d+)$", ckpt.stem)
        if match:
            candidates.append((int(match.group(1)), ckpt))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    latest_epoch, latest_ckpt = candidates[-1]
    logger.info(f"Resuming from checkpoint: {latest_ckpt.name} (epoch {latest_epoch})")
    return latest_ckpt




def verify_checkpoints(logger: logging.Logger) -> None:
    """Verify and log saved checkpoints."""
    logger.info("=" * 60)
    logger.info("Verifying saved checkpoints")
    logger.info("=" * 60)
    
    if not CHECKPOINT_DIR.exists():
        logger.warning(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return
    
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
    
    if not checkpoints:
        logger.warning("No checkpoints found")
        return
    
    logger.info(f"Found {len(checkpoints)} checkpoint(s)")
    
    for ckpt in checkpoints:
        try:
            size_mb = os.path.getsize(ckpt) / (1024 * 1024)
            logger.info(f"  {ckpt.name} ({size_mb:.1f} MB)")
            
            state = torch.load(ckpt, map_location='cpu')
            epoch = state.get('epoch', 'unknown')
            avg_loss = state.get('avg_loss', 'unknown')
            logger.debug(f"    Epoch: {epoch}, Average Loss: {avg_loss}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt.name}: {e}", exc_info=True)



def main():
    """Main training script entry point."""
    logger = setup_logging(LOG_DIR)
    
    logger.info("=" * 60)
    logger.info("VibeShift Training Script Started")
    logger.info("=" * 60)
    
    try:
        # Data setup
        source_files, target_files = setup_data(logger)
        
        # Dataset initialization
        logger.info("Creating DAC dataset...")
        dataset = DACDataset(source_files, target_files)
        validate_dataset(dataset, logger)
        
        # Model setup
        config, pipeline = setup_model(logger)

        # Resume from latest checkpoint if available
        start_epoch = 1
        best_loss = None
        latest_ckpt = get_latest_checkpoint(logger)
        if latest_ckpt is not None:
            state = pipeline.load_checkpoint(str(latest_ckpt))
            last_epoch = int(state.get("epoch", 0))
            best_loss = state.get("best_loss", state.get("avg_loss", None))
            start_epoch = last_epoch + 1
            if start_epoch <= config.num_epochs:
                logger.info(f"Resuming training at epoch {start_epoch}")
            else:
                logger.info(
                    f"Latest checkpoint is at epoch {last_epoch}; "
                    f"num_epochs is {config.num_epochs}. Skipping training."
                )
        
        # Data loading
        logger.info("Setting up data loader...")
        loader = pipeline.setup_data(source_files, target_files)
        
        # Validate batch loading
        logger.info("Testing batch loading...")
        batch = next(iter(loader))
        x0_batch, x1_batch, mask_batch = batch
        logger.debug(f"Batch shapes - x0: {x0_batch.shape}, x1: {x1_batch.shape}, mask: {mask_batch.shape}")
        
        # Run training
        if start_epoch <= config.num_epochs:
            losses = run_training(
                pipeline,
                loader,
                logger,
                start_epoch=start_epoch,
                best_loss=best_loss,
            )
        else:
            losses = []
        
        # Verify checkpoints
        verify_checkpoints(logger)
        
        logger.info("=" * 60)
        logger.info("Training script completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical(f"Training script failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()