import torch
from torch.utils.data import Dataset


def collate_variable_length_dac(batch):
    """
    Collate function for batches with variable-length DAC embeddings.
    Pads all sequences to the maximum length in the batch.
    
    Args:
        batch: List of (x0, x1) tuples where x0, x1 are (T, latent_dim) tensors
    
    Returns:
        Tuple of (x0_batch, x1_batch, masks) where:
        - x0_batch: (B, max_T, latent_dim) padded source embeddings
        - x1_batch: (B, max_T, latent_dim) padded target embeddings
        - masks: (B, max_T) boolean mask (True where data exists, False where padded)
    """
    x0_list, x1_list = zip(*batch)
    
    # Find max time length in this batch
    max_time = max(max(x0.shape[0], x1.shape[0]) for x0, x1 in zip(x0_list, x1_list))
    latent_dim = x0_list[0].shape[-1]
    
    x0_batch = []
    x1_batch = []
    masks = []
    
    for x0, x1 in zip(x0_list, x1_list):
        # Store original time before padding
        original_time = x0.shape[0]
        
        # Pad x0 to max_time: (T, D) -> (max_T, D)
        if x0.shape[0] < max_time:
            pad_amount = max_time - x0.shape[0]
            x0 = torch.nn.functional.pad(x0, (0, 0, 0, pad_amount), mode='constant', value=0)
        
        # Pad x1 to max_time
        if x1.shape[0] < max_time:
            pad_amount = max_time - x1.shape[0]
            x1 = torch.nn.functional.pad(x1, (0, 0, 0, pad_amount), mode='constant', value=0)
        
        # Create mask (True for original data, False for padding)
        mask = torch.ones(max_time, dtype=torch.bool)
        mask[original_time:] = False
        
        x0_batch.append(x0)
        x1_batch.append(x1)
        masks.append(mask)
    
    # Stack into batch (B, max_T, latent_dim)
    x0_batch = torch.stack(x0_batch, dim=0)
    x1_batch = torch.stack(x1_batch, dim=0)
    masks = torch.stack(masks, dim=0)  # (B, max_T)
    
    return x0_batch, x1_batch, masks


class DACDataset(Dataset):
    """Load paired DAC embeddings (deterministic 1-to-1 pairing)"""
    def __init__(self, source_files, target_files):
        self.source_files = source_files
        self.target_files = target_files

    def __len__(self):
        return len(self.source_files)

    def _extract_embeddings(self, obj):
        """Extract embeddings from saved DAC data"""
        if isinstance(obj, dict):
            # DAC saves with 'embeddings' key
            if 'embeddings' in obj:
                return obj['embeddings']
            # Fallback keys
            for k in ("embeddings", "latent", "z", "x"):
                if k in obj:
                    return obj[k]
            raise KeyError(f"Embeddings not found in dict keys: {list(obj.keys())}")
        return obj

    def _load_embeddings(self, path):
        """Load DAC embeddings and ensure 2D: (T, latent_dim)"""
        data = torch.load(path)
        emb = self._extract_embeddings(data)

        # Ensure 2D: (T, latent_dim)
        while emb.dim() > 2:
            emb = emb.squeeze(0)
        
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)  # (1, latent_dim) - single frame

        return emb

    def __getitem__(self, idx):
        x0 = self._load_embeddings(self.source_files[idx])
        x1 = self._load_embeddings(self.target_files[idx])
        return x0, x1


class PairedDACDataset(Dataset):
    """Load DAC embeddings with deterministic paired pairing for overfitting tests"""
    def __init__(self, source_files, target_files, repeat=1):
        """
        Args:
            source_files: List of source DAC embedding file paths
            target_files: List of target DAC embedding file paths (must align with source_files by index)
            repeat: Number of times to repeat the dataset (for overfitting on small datasets)
        """
        if len(source_files) != len(target_files):
            raise ValueError(f"Source and target file counts must match: {len(source_files)} != {len(target_files)}")
        
        self.source_files = source_files
        self.target_files = target_files
        self.repeat = repeat

    def __len__(self):
        return len(self.source_files) * self.repeat

    def _extract_embeddings(self, obj):
        """Extract embeddings from saved DAC data"""
        if isinstance(obj, dict):
            if 'embeddings' in obj:
                return obj['embeddings']
            for k in ("embeddings", "latent", "z", "x"):
                if k in obj:
                    return obj[k]
            raise KeyError(f"Embeddings not found in dict keys: {list(obj.keys())}")
        return obj

    def _load_embeddings(self, path):
        """Load DAC embeddings and ensure 2D: (T, latent_dim)"""
        data = torch.load(path)
        emb = self._extract_embeddings(data)

        while emb.dim() > 2:
            emb = emb.squeeze(0)
        
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        return emb

    def __getitem__(self, idx):
        # Map index to actual file pair (handling repeat)
        actual_idx = idx % len(self.source_files)
        x0 = self._load_embeddings(self.source_files[actual_idx])
        x1 = self._load_embeddings(self.target_files[actual_idx])
        return x0, x1
