import math
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for vision transformers.
    Converts image patches to embeddings.
    """
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        """
        Args:
            img_size: Size of input image (assumes square)
            patch_size: Size of each patch (assumes square)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) image tensor
        
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PositionalEmbedding(nn.Module):
    """
    Positional embedding layer for transformers.
    Adds learnable position embeddings to token embeddings.
    """
    
    def __init__(self, num_positions, embed_dim, sinusoidal=False):
        """
        Args:
            num_positions: Maximum number of positions
            embed_dim: Embedding dimension
            sinusoidal: If True, use sinusoidal positional encoding; else learnable
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.sinusoidal = sinusoidal
        
        if sinusoidal:
            self.register_buffer('pos_emb', self._create_sinusoidal_embeddings(num_positions, embed_dim))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
            nn.init.normal_(self.pos_emb, std=0.02)
    
    def _create_sinusoidal_embeddings(self, num_positions, embed_dim):
        """Create sinusoidal positional embeddings."""
        pos = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe = torch.zeros(num_positions, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, embed_dim) token embeddings
        
        Returns:
            (B, seq_len, embed_dim) embeddings with positional information
        """
        seq_len = x.shape[1]
        return x + self.pos_emb[:, :seq_len, :]
