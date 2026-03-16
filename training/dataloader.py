"""
Dataloader classes for VibeShift training (FIXED VERSION).
Implements lazy loading, dynamic padding, and proper device handling.
"""

import torch
import numpy as np
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
from torch.utils.data import Dataset, DataLoader


def default_collate_with_dynamic_padding(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function with dynamic padding and attention mask generation.
    Pads sequences to the maximum length in the current batch, not the dataset max.
    
    Args:
        batch: List of (x0, x1) or (x0, x1, genre_id) tuples
    
    Returns:
        Padded batch tensors with attention mask:
        - For genre-aware: (x0_padded, x1_padded, genre_ids, mask)
        - For standard: (x0_padded, x1_padded, mask)
    """
    if isinstance(batch[0], tuple) and len(batch[0]) == 3:
        # Genre-aware dataset
        x0_list, x1_list, genre_ids = zip(*batch)
        x0_list = list(x0_list)
        x1_list = list(x1_list)
        genre_ids = torch.tensor(genre_ids, dtype=torch.long)
        
        # Compute original lengths BEFORE padding
        x0_lengths = [x.size(0) for x in x0_list]
        x1_lengths = [x.size(0) for x in x1_list]
        max_len = max(max(x0_lengths), max(x1_lengths))
        
        # Pad to batch max
        x0_padded = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, max_len - x.size(0))) 
            for x in x0_list
        ])
        x1_padded = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, max_len - x.size(0))) 
            for x in x1_list
        ])
        
        # Mask based on source (x0) length only — frames where source is
        # zero-padded produce v_true = x1 - 0 = x1, a nonsensical training signal.
        mask = torch.zeros(len(batch), max_len)
        for i, len0 in enumerate(x0_lengths):
            mask[i, :len0] = 1.0
        
        return x0_padded, x1_padded, genre_ids, mask
    else:
        # Standard dataset
        x0_list, x1_list = zip(*batch)
        x0_list = list(x0_list)
        x1_list = list(x1_list)
        
        # Compute original lengths BEFORE padding
        x0_lengths = [x.size(0) for x in x0_list]
        x1_lengths = [x.size(0) for x in x1_list]
        max_len = max(max(x0_lengths), max(x1_lengths))
        
        # Pad to batch max
        x0_padded = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, max_len - x.size(0))) 
            for x in x0_list
        ])
        x1_padded = torch.stack([
            torch.nn.functional.pad(x, (0, 0, 0, max_len - x.size(0))) 
            for x in x1_list
        ])
        
        # Mask based on source (x0) length only — same reasoning as above.
        mask = torch.zeros(len(batch), max_len)
        for i, len0 in enumerate(x0_lengths):
            mask[i, :len0] = 1.0
        
        return x0_padded, x1_padded, mask


class LatentPairDataset(Dataset):
   
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the latent pair dataset with LAZY LOADING.
        
        Args:
            source_dir: Directory containing source latent embeddings (.pt files)
            target_dir: Directory containing target latent embeddings (.pt files)
            max_samples: Maximum number of samples to load (None = all)
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Find all source files (lazy loading - just store paths)
        self.source_files = sorted(self.source_dir.glob("*.pt"))
        self.target_files = sorted(self.target_dir.glob("*.pt"))
        
        if not self.source_files:
            raise FileNotFoundError(f"No .pt files found in {source_dir}")
        if not self.target_files:
            raise FileNotFoundError(f"No .pt files found in {target_dir}")
        
        if len(self.source_files) != len(self.target_files):
            raise ValueError(
                f"Mismatch: {len(self.source_files)} source files, "
                f"{len(self.target_files)} target files"
            )
        
        # Limit to max_samples if specified
        if max_samples is not None:
            self.source_files = self.source_files[:max_samples]
            self.target_files = self.target_files[:max_samples]
        
        # Cache for max sequence length (computed lazily)
        self._max_length = None
        self._embedding_dim = None
        
        print(f"Dataset initialized with {len(self)} samples (lazy loading)")
        print(f"  Source dir: {source_dir}")
        print(f"  Target dir: {target_dir}")
    
    def _compute_max_length(self) -> int:
        """Compute maximum sequence length (cached on first call)."""
        if self._max_length is not None:
            return self._max_length
        
        print("Computing dataset max sequence length...")
        max_len = 0
        
        # FIXED: Provide both iterables to zip()
        sample_size = min(10, len(self.source_files))
        for src_file, tgt_file in zip(self.source_files[:sample_size], 
                                       self.target_files[:sample_size]):
            try:
                src_data = torch.load(src_file, map_location="cpu", weights_only=True)
                src_emb = self._extract_embeddings(src_data)
                if src_emb.dim() == 3 and src_emb.size(0) == 1:
                    src_emb = src_emb.squeeze(0)
                max_len = max(max_len, src_emb.size(0))
                
                tgt_data = torch.load(tgt_file, map_location="cpu", weights_only=True)
                tgt_emb = self._extract_embeddings(tgt_data)
                if tgt_emb.dim() == 3 and tgt_emb.size(0) == 1:
                    tgt_emb = tgt_emb.squeeze(0)
                max_len = max(max_len, tgt_emb.size(0))
            except Exception as e:
                warnings.warn(f"Could not compute length for {src_file}: {e}")
        
        self._max_length = max_len
        return max_len
    
    @staticmethod
    def _extract_embeddings(data) -> torch.Tensor:
        """Extract embeddings from various data formats."""
        if isinstance(data, dict):
            for key in ["z", "embeddings", "latents"]:
                if key in data:
                    return data[key]
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"No tensor found in dict with keys: {data.keys()}")
        elif isinstance(data, torch.Tensor):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.source_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample pair (LAZY LOADING).
        Tensors are kept on CPU; DataLoader moves them to device.
        
        Returns:
            (x0, x1): Source and target embeddings on CPU
        """
        src_file = self.source_files[idx]
        tgt_file = self.target_files[idx]
        
        try:
            # Load from disk
            src_data = torch.load(src_file, map_location="cpu", weights_only=True)
            x0 = self._extract_embeddings(src_data)
            
            tgt_data = torch.load(tgt_file, map_location="cpu", weights_only=True)
            x1 = self._extract_embeddings(tgt_data)
            
            # Remove batch dimension if present
            if x0.dim() == 3 and x0.size(0) == 1:
                x0 = x0.squeeze(0)
            if x1.dim() == 3 and x1.size(0) == 1:
                x1 = x1.squeeze(0)
            
            # FIXED: Only warn about extremely long sequences (avoid calling _compute_max_length every time)
            if x0.size(0) > 10000 or x1.size(0) > 10000:
                warnings.warn(
                    f"Sample {idx}: Very long sequence detected. "
                    f"x0: {x0.size(0)}, x1: {x1.size(0)}"
                )
            
            return x0, x1
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} from {src_file}: {e}")
    
    def get_info(self) -> Dict:
        """Return dataset information."""
        # Compute embedding dim from first sample
        if self._embedding_dim is None:
            x0, _ = self[0]
            self._embedding_dim = x0.size(-1)
        
        return {
            "num_samples": len(self),
            "max_sequence_length": self._compute_max_length(),
            "embedding_dim": self._embedding_dim,
            "source_dir": str(self.source_dir),
            "target_dir": str(self.target_dir),
            "memory_efficient": "lazy loading",
        }


class GenreAwareLatentDataset(LatentPairDataset):
    """
    Extended lazy-loading dataset with genre labels.
    Useful for training genre-conditional models.
    """
    
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        genre_labels: Optional[Dict[int, int]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize genre-aware dataset with lazy loading.
        
        Args:
            source_dir: Directory containing source latent embeddings
            target_dir: Directory containing target latent embeddings
            genre_labels: Mapping of sample index to genre id
            max_samples: Maximum number of samples to load
        """
        super().__init__(
            source_dir=source_dir,
            target_dir=target_dir,
            max_samples=max_samples,
        )
        
        # Set default genre labels if not provided.
        # Default to genre 1 (punk/target) — all target files in VibeShift are punk.
        # Genre convention: 0=synth (source), 1=punk (target), num_genres=null (CFG)
        if genre_labels is None:
            genre_labels = {i: 1 for i in range(len(self))}
        
        self.genre_labels = genre_labels
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample with genre label (lazy loading).
        
        Returns:
            (x0, x1, genre_id): Source and target embeddings with genre
        """
        x0, x1 = super().__getitem__(idx)
        genre_id = self.genre_labels.get(idx, 0)
        return x0, x1, genre_id


def create_dataloader(
    source_dir: str,
    target_dir: str,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    genre_aware: bool = False,
    genre_labels: Optional[Dict[int, int]] = None,
) -> Tuple[DataLoader, Dataset]:
    """
    Create a dataloader with proper device handling and dynamic padding.
    
    FIXED: Keeps tensors on CPU, uses pin_memory for efficient GPU transfer.
    
    Args:
        source_dir: Directory containing source latent embeddings
        target_dir: Directory containing target latent embeddings
        batch_size: Batch size for dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes (0 for single-threaded)
        max_samples: Maximum number of samples to load
        drop_last: Drop last incomplete batch
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep worker processes alive between epochs (recommended
            when num_workers > 0 in Jupyter to prevent workers dying between epochs)
        genre_aware: If True, use GenreAwareLatentDataset; if False, use LatentPairDataset
        genre_labels: Mapping of sample index to genre ID. Only used when genre_aware=True.
            If None, defaults to all samples being genre 1 (punk/target).
    
    Returns:
        (dataloader, dataset): PyTorch DataLoader and underlying Dataset
    """
    # Warn if using multiprocessing with CUDA
    if num_workers > 0:
        print(f"Warning: Using {num_workers} workers. "
              "Ensure CUDA is not initialized in worker processes.")
    
    if genre_aware:
        # Default all samples to genre 1 (punk = target genre) if no labels provided.
        # In VibeShift, all target files are punk so genre 1 is the correct default.
        if genre_labels is None:
            # Build default labels: all samples map to genre 1
            _base = LatentPairDataset(
                source_dir=source_dir,
                target_dir=target_dir,
                max_samples=max_samples,
            )
            genre_labels = {i: 1 for i in range(len(_base))}
            del _base  # Clean up temporary dataset
        
        dataset = GenreAwareLatentDataset(
            source_dir=source_dir,
            target_dir=target_dir,
            genre_labels=genre_labels,
            max_samples=max_samples,
        )
    else:
        dataset = LatentPairDataset(
            source_dir=source_dir,
            target_dir=target_dir,
            max_samples=max_samples,
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=default_collate_with_dynamic_padding,
    )
    
    return dataloader, dataset
