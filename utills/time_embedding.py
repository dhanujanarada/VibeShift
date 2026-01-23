import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    """
    Time embedding layer for diffusion models.
    Converts continuous or discrete time values to sinusoidal embeddings.
    """
    
    def __init__(self, dim):
        """
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (B,) continuous or discrete time values
        
        Returns:
            (B, dim) time embeddings
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        
        return emb
