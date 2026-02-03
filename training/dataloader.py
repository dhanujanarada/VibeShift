import torch
from torch.utils.data import Dataset
import random

class MelDataset(Dataset):
    """Load paired mel spectrograms (deterministic 1-to-1 pairing)"""
    def __init__(self, source_files, target_files):
        self.source_files = source_files
        self.target_files = target_files

    def __len__(self):
        return len(self.source_files)

    def _extract_mel(self, obj):
        if isinstance(obj, dict):
            for k in ("mel", "spec", "melspec", "x"):
                if k in obj:
                    return obj[k]
            raise KeyError(f"Mel tensor not found in dict keys: {list(obj.keys())}")
        return obj

    def _load_mel(self, path):
        """Load mel and ensure 3D: (1, n_mels, time)"""
        mel = torch.load(path)
        mel = self._extract_mel(mel)

        # Squeeze to 2D first, then unsqueeze to (1, n_mels, time)
        while mel.dim() > 2:
            mel = mel.squeeze(0)
        
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # (1, n_mels, time)

        return mel

    def __getitem__(self, idx):
        x0 = self._load_mel(self.source_files[idx])
        x1 = self._load_mel(self.target_files[idx])
        return x0, x1


class RandomPairMelDataset(Dataset):
    """Load mel spectrograms with random target pairing"""
    def __init__(self, source_files, target_files):
        self.source_files = source_files
        self.target_files = target_files

    def __len__(self):
        return len(self.source_files)

    def _extract_mel(self, obj):
        if isinstance(obj, dict):
            for k in ("mel", "spec", "melspec", "x"):
                if k in obj:
                    return obj[k]
            raise KeyError(f"Mel tensor not found in dict keys: {list(obj.keys())}")
        return obj

    def _load_mel(self, path):
        """Load mel and ensure 3D: (1, n_mels, time)"""
        mel = torch.load(path)
        mel = self._extract_mel(mel)

        # Squeeze to 2D first, then unsqueeze to (1, n_mels, time)
        while mel.dim() > 2:
            mel = mel.squeeze(0)
        
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # (1, n_mels, time)

        return mel

    def __getitem__(self, idx):
        x0 = self._load_mel(self.source_files[idx])
        x1 = self._load_mel(random.choice(self.target_files))
        return x0, x1
