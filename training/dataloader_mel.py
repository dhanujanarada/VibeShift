"""
Mel-spectrogram dataloader for VibeShift models2 training.

Mirrors training/dataloader.py but handles:
  • .pt files saved by utills/mel.save_mel_with_metadata()
    (dict with keys: 'mel', 'genre_id', 'sample_rate', ...)
  • raw .pt tensors of shape (1, n_mels, T) or (n_mels, T)
  • fixed-length chunking so every sample in a batch has the same T
  • mel normalisation (log1p, optional per-dataset mean/std z-score)

Output tensors are always (1, n_mels, chunk_T)  — channel-first, ready for DiTMel.
"""

import warnings
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────────

def mel_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of (x0, x1, genre_id) mel pairs.

    All tensors are assumed to be the same size (chunk_T) — the dataset
    handles chunking.  A time mask of all-ones is returned for API
    compatibility with FlowMatchingMel.compute_loss().

    Returns:
        x0:        (B, 1, n_mels, T)
        x1:        (B, 1, n_mels, T)
        genre_ids: (B,)
        mask:      (B, T)  — all ones (no padding needed)
    """
    x0_list, x1_list, genre_ids = zip(*batch)

    x0 = torch.stack(x0_list)          # (B, 1, n_mels, T)
    x1 = torch.stack(x1_list)
    genre_ids = torch.tensor(genre_ids, dtype=torch.long)

    T = x0.shape[-1]
    mask = torch.ones(len(batch), T)

    return x0, x1, genre_ids, mask


def mel_collate_pad_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate with dynamic padding — use this when NOT chunking (variable-T files).

    Pads the time axis to the longest sample in the batch and returns
    a boolean mask (1 = valid, 0 = pad).
    """
    x0_list, x1_list, genre_ids = zip(*batch)
    genre_ids = torch.tensor(genre_ids, dtype=torch.long)

    max_T = max(x.shape[-1] for x in x0_list)

    def pad(tensors):
        return torch.stack([
            F.pad(t, (0, max_T - t.shape[-1]))
            for t in tensors
        ])

    x0 = pad(x0_list)
    x1 = pad(x1_list)

    lengths = torch.tensor([x.shape[-1] for x in x0_list])
    mask = torch.arange(max_T).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)

    return x0, x1, genre_ids, mask.float()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load a single .pt mel file
# ──────────────────────────────────────────────────────────────────────────────

def _load_mel_tensor(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Load a mel .pt file and return (mel, genre_id).

    Handles:
      • dict from utills/mel.save_mel_with_metadata()
        {'mel': (1, n_mels, T), 'genre_id': int, ...}
      • raw tensor (1, n_mels, T) or (n_mels, T)

    Returns:
        mel:      (1, n_mels, T)  float32
        genre_id: int (0 if not found)
    """
    raw = torch.load(str(path), map_location="cpu", weights_only=True)

    if isinstance(raw, dict):
        mel = raw.get("mel") or raw.get("mel_spec") or raw.get("spectrogram")
        if mel is None:
            # Fall back: take the first tensor value
            for v in raw.values():
                if isinstance(v, torch.Tensor):
                    mel = v
                    break
        if mel is None:
            raise ValueError(f"No mel tensor found in {path.name}. Keys: {list(raw.keys())}")
        genre_id = int(raw.get("genre_id", 0))
    elif isinstance(raw, torch.Tensor):
        mel = raw
        genre_id = 0
    else:
        raise TypeError(f"Unexpected type {type(raw)} in {path.name}")

    mel = mel.float()
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)   # (n_mels, T) → (1, n_mels, T)
    if mel.dim() != 3:
        raise ValueError(f"{path.name}: expected 3-D mel, got {mel.shape}")

    return mel, genre_id


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MelPairDataset(Dataset):
    """
    Paired mel-spectrogram dataset for genre transfer training.

    Expects two directories:
        source_dir/  *.pt   — source genre mels  (e.g. non-rock / classical)
        target_dir/  *.pt   — target genre mels  (e.g. rock / punk)

    Files are paired by sorted filename order (same convention as
    training/dataloader.py LatentPairDataset).

    Args:
        source_dir:    path to source .pt files
        target_dir:    path to target .pt files
        target_genre:  genre id for the target class (default 1 = rock/punk)
        source_genre:  genre id for the source class (default 0 = non-rock)
        chunk_T:       fixed time length per sample (frames).  Longer mels are
                       randomly cropped; shorter are either padded or skipped.
                       Set to None to disable chunking (use mel_collate_pad_fn).
        chunk_pad:     if True, pad short mels instead of skipping them.
        log_scale:     apply log1p to linearise power mel (recommended).
        normalise:     if True, z-score normalise each mel independently.
        max_samples:   cap the dataset size (useful for quick tests).
        seed:          random seed for reproducible chunking.
    """

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        target_genre: int = 1,
        source_genre: int = 0,
        chunk_T: Optional[int] = 256,
        chunk_pad: bool = True,
        log_scale: bool = True,
        normalise: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.source_dir   = Path(source_dir)
        self.target_dir   = Path(target_dir)
        self.target_genre = target_genre
        self.source_genre = source_genre
        self.chunk_T      = chunk_T
        self.chunk_pad    = chunk_pad
        self.log_scale    = log_scale
        self.normalise    = normalise
        self.seed         = seed

        src_files = sorted(self.source_dir.glob("*.pt"))
        tgt_files = sorted(self.target_dir.glob("*.pt"))

        if not src_files:
            raise FileNotFoundError(f"No .pt files found in {source_dir}")
        if not tgt_files:
            raise FileNotFoundError(f"No .pt files found in {target_dir}")

        n = min(len(src_files), len(tgt_files))
        if len(src_files) != len(tgt_files):
            warnings.warn(
                f"Source/target counts differ ({len(src_files)} vs {len(tgt_files)}). "
                f"Using first {n} of each."
            )

        self.source_files = src_files[:n]
        self.target_files = tgt_files[:n]

        if max_samples is not None:
            self.source_files = self.source_files[:max_samples]
            self.target_files = self.target_files[:max_samples]

        print(f"MelPairDataset: {len(self)} pairs")
        print(f"  source : {source_dir}  (genre {source_genre})")
        print(f"  target : {target_dir}  (genre {target_genre})")
        print(f"  chunk_T: {chunk_T}  |  log_scale: {log_scale}  |  normalise: {normalise}")

    # ------------------------------------------------------------------
    def _preprocess(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply log-scaling and optional normalisation. Returns (1, n_mels, T)."""
        if self.log_scale:
            mel = torch.log1p(mel.clamp(min=0))
        if self.normalise:
            mu  = mel.mean()
            std = mel.std().clamp(min=1e-6)
            mel = (mel - mu) / std
        return mel

    def _chunk(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Crop or pad the time axis to self.chunk_T.
        Returns (1, n_mels, chunk_T).
        """
        T = mel.shape[-1]
        if T >= self.chunk_T:
            # Random crop
            start = random.randint(0, T - self.chunk_T)
            return mel[:, :, start : start + self.chunk_T]
        elif self.chunk_pad:
            return F.pad(mel, (0, self.chunk_T - T))
        else:
            return None   # caller will skip

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.source_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            x0:       (1, n_mels, chunk_T)  source mel
            x1:       (1, n_mels, chunk_T)  target mel
            genre_id: int (= self.target_genre)
        """
        x0, _ = _load_mel_tensor(self.source_files[idx])
        x1, _ = _load_mel_tensor(self.target_files[idx])

        x0 = self._preprocess(x0)
        x1 = self._preprocess(x1)

        if self.chunk_T is not None:
            x0 = self._chunk(x0)
            x1 = self._chunk(x1)
            if x0 is None or x1 is None:
                # File too short and chunk_pad=False — return next sample
                return self[(idx + 1) % len(self)]

        return x0, x1, self.target_genre

    # ------------------------------------------------------------------
    def get_info(self) -> Dict:
        x0, x1, g = self[0]
        return {
            "num_samples"  : len(self),
            "mel_shape"    : tuple(x0.shape),
            "source_genre" : self.source_genre,
            "target_genre" : self.target_genre,
            "chunk_T"      : self.chunk_T,
            "log_scale"    : self.log_scale,
            "normalise"    : self.normalise,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Single-folder dataset (each file carries its own genre_id from metadata)
# ──────────────────────────────────────────────────────────────────────────────

class MelSingleFolderDataset(Dataset):
    """
    Dataset that pairs files from a single mixed folder.

    Useful when you have one folder of source mels and want to pair them
    with random target mels (self-supervised / unpaired training).
    Each .pt file must be saved with utills/mel.save_mel_with_metadata()
    so the genre_id is stored in the dict.

    Paired by sorted index (file[i] → source,  file[(i + offset) % N] → target).
    """

    def __init__(
        self,
        mel_dir: str,
        target_genre: int = 1,
        chunk_T: Optional[int] = 256,
        chunk_pad: bool = True,
        log_scale: bool = True,
        normalise: bool = False,
        pair_offset: int = 1,
        max_samples: Optional[int] = None,
    ):
        self.mel_dir      = Path(mel_dir)
        self.target_genre = target_genre
        self.chunk_T      = chunk_T
        self.chunk_pad    = chunk_pad
        self.log_scale    = log_scale
        self.normalise    = normalise
        self.pair_offset  = pair_offset

        files = sorted(self.mel_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {mel_dir}")
        if max_samples:
            files = files[:max_samples]
        self.files = files

        print(f"MelSingleFolderDataset: {len(self)} files in {mel_dir}")

    def _preprocess(self, mel):
        if self.log_scale:
            mel = torch.log1p(mel.clamp(min=0))
        if self.normalise:
            mu, std = mel.mean(), mel.std().clamp(min=1e-6)
            mel = (mel - mu) / std
        return mel

    def _chunk(self, mel):
        T = mel.shape[-1]
        if T >= self.chunk_T:
            start = random.randint(0, T - self.chunk_T)
            return mel[:, :, start : start + self.chunk_T]
        return F.pad(mel, (0, self.chunk_T - T)) if self.chunk_pad else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tgt_idx = (idx + self.pair_offset) % len(self)
        x0, g0 = _load_mel_tensor(self.files[idx])
        x1, _  = _load_mel_tensor(self.files[tgt_idx])

        x0, x1 = self._preprocess(x0), self._preprocess(x1)
        if self.chunk_T is not None:
            x0, x1 = self._chunk(x0), self._chunk(x1)
            if x0 is None or x1 is None:
                return self[(idx + 1) % len(self)]

        genre_id = self.target_genre if self.target_genre is not None else g0
        return x0, x1, genre_id


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_mel_dataloader(
    source_dir: str,
    target_dir: str,
    target_genre: int = 1,
    source_genre: int = 0,
    batch_size: int = 8,
    chunk_T: int = 256,
    chunk_pad: bool = True,
    log_scale: bool = True,
    normalise: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
    drop_last: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, MelPairDataset]:
    """
    Create a DataLoader for paired mel spectrograms.

    Args:
        source_dir:   folder with source .pt mels
        target_dir:   folder with target .pt mels
        target_genre: genre id of the target class
        source_genre: genre id of the source class
        batch_size:   samples per batch
        chunk_T:      fixed time length (frames) per sample
        chunk_pad:    pad short files instead of skipping
        log_scale:    apply log1p to mel power
        normalise:    z-score each mel independently
        shuffle:      shuffle dataset each epoch
        num_workers:  DataLoader workers (0 = main process)
        max_samples:  cap dataset size
        drop_last:    drop final incomplete batch
        pin_memory:   pin CPU tensors for fast GPU transfer
        seed:         random seed

    Returns:
        (dataloader, dataset)
    """
    dataset = MelPairDataset(
        source_dir   = source_dir,
        target_dir   = target_dir,
        target_genre = target_genre,
        source_genre = source_genre,
        chunk_T      = chunk_T,
        chunk_pad    = chunk_pad,
        log_scale    = log_scale,
        normalise    = normalise,
        max_samples  = max_samples,
        seed         = seed,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        drop_last   = drop_last,
        pin_memory  = pin_memory,
        persistent_workers = (num_workers > 0),
        collate_fn  = mel_collate_fn,
    )

    return loader, dataset
