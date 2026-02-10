"""
Training module for VibeShift.
Provides training utilities, dataloaders, and model training classes.
"""

from .training import Trainer, TrainingConfig
from .dataloader import (
    LatentPairDataset,
    GenreAwareLatentDataset,
    create_dataloader,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "LatentPairDataset",
    "GenreAwareLatentDataset",
    "create_dataloader",
]
