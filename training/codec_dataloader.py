import torch
from torch.utils.data import Dataset
import random


class CodecDataset(Dataset):
    """Load paired codec representations (codes or embeddings)"""
    def __init__(self, source_files, target_files, use_embeddings=True):
        """
        Args:
            source_files: List of source codec files
            target_files: List of target codec files
            use_embeddings: If True, use continuous embeddings. If False, use discrete codes.
        """
        self.source_files = source_files
        self.target_files = target_files
        self.use_embeddings = use_embeddings
    
    def __len__(self):
        return len(self.source_files)
    
    def _extract_codec(self, obj, use_embeddings=True):
        """Extract embeddings or codes from loaded data"""
        if isinstance(obj, dict):
            key = 'embeddings' if use_embeddings else 'codes'
            if key in obj:
                return obj[key]
            # Fallback to other keys
            if use_embeddings and 'embeddings' in obj:
                return obj['embeddings']
            if not use_embeddings and 'codes' in obj:
                return obj['codes']
            raise KeyError(f"'{key}' not found in codec data. Keys: {list(obj.keys())}")
        return obj
    
    def __getitem__(self, idx):
        source = torch.load(self.source_files[idx])
        target = torch.load(self.target_files[idx])
        
        source = self._extract_codec(source, self.use_embeddings)
        target = self._extract_codec(target, self.use_embeddings)
        
        return source, target


class RandomPairCodecDataset(Dataset):
    """Load codec with random target pairing"""
    def __init__(self, source_files, target_files, use_embeddings=True):
        self.source_files = source_files
        self.target_files = target_files
        self.use_embeddings = use_embeddings
    
    def __len__(self):
        return len(self.source_files)
    
    def _extract_codec(self, obj, use_embeddings=True):
        if isinstance(obj, dict):
            key = 'embeddings' if use_embeddings else 'codes'
            if key in obj:
                return obj[key]
            raise KeyError(f"'{key}' not found in codec data")
        return obj
    
    def __getitem__(self, idx):
        source = torch.load(self.source_files[idx])
        target_idx = random.randint(0, len(self.target_files) - 1)
        target = torch.load(self.target_files[target_idx])
        
        source = self._extract_codec(source, self.use_embeddings)
        target = self._extract_codec(target, self.use_embeddings)
        
        return source, target


class PairedCodecDataset(Dataset):
    """Load paired codecs with deterministic pairing (for overfitting tests)"""
    def __init__(self, source_files, target_files, use_embeddings=True, repeat=1):
        if len(source_files) != len(target_files):
            raise ValueError(f"Source and target counts must match: {len(source_files)} != {len(target_files)}")
        
        self.source_files = source_files
        self.target_files = target_files
        self.use_embeddings = use_embeddings
        self.repeat = repeat
    
    def __len__(self):
        return len(self.source_files) * self.repeat
    
    def _extract_codec(self, obj, use_embeddings=True):
        if isinstance(obj, dict):
            key = 'embeddings' if use_embeddings else 'codes'
            if key in obj:
                return obj[key]
            raise KeyError(f"'{key}' not found in codec data")
        return obj
    
    def __getitem__(self, idx):
        actual_idx = idx % len(self.source_files)
        source = torch.load(self.source_files[actual_idx])
        target = torch.load(self.target_files[actual_idx])
        
        source = self._extract_codec(source, self.use_embeddings)
        target = self._extract_codec(target, self.use_embeddings)
        
        return source, target


def collate_variable_length_codec(batch):
    """Pad variable-length codec sequences to max length in batch"""
    sources, targets = zip(*batch)
    
    # Get max lengths
    max_frames_src = max(s.shape[0] for s in sources)
    max_frames_tgt = max(t.shape[0] for t in targets)
    max_frames = max(max_frames_src, max_frames_tgt)
    
    # Pad sequences
    source_padded = []
    target_padded = []
    masks = []
    
    for source, target in zip(sources, targets):
        # Store original lengths
        original_src = source.shape[0]
        original_tgt = target.shape[0]
        
        # Pad source
        pad_src = max_frames - source.shape[0]
        if pad_src > 0:
            source = torch.nn.functional.pad(source, (0, 0, 0, pad_src), value=0)
        source_padded.append(source)
        
        # Pad target
        pad_tgt = max_frames - target.shape[0]
        if pad_tgt > 0:
            target = torch.nn.functional.pad(target, (0, 0, 0, pad_tgt), value=0)
        target_padded.append(target)
        
        # Create mask (1 for valid frames, 0 for padding)
        mask = torch.ones(max_frames)
        valid_frames = min(original_src, original_tgt)
        mask[valid_frames:] = 0
        masks.append(mask)
    
    # Stack into batches (B, T, D)
    source_batch = torch.stack(source_padded, dim=0)
    target_batch = torch.stack(target_padded, dim=0)
    mask_batch = torch.stack(masks, dim=0)
    
    return source_batch, target_batch, mask_batch
