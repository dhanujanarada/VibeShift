import torch
import torch.nn as nn
from omegaconf import DictConfig
from utills.embedding import RoPEEmbedding, GenreEmbedding , TimeEmbedding, PatchEmbedding
from models.film_conditioner import FiLMConditioner

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, num_genres, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Attention with RoPE (built-in now)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEEmbedding(dim, num_heads, dropout) 
        
        # FiLM conditioning
        self.norm2 = nn.LayerNorm(dim)
        self.film = FiLMConditioner(dim, num_genres)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, t, genre_ids, attn_mask=None):
        # Self-attention (RoPE is inside now)
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, attn_mask)  
        x = x + attn_out
        
        # FiLM conditioning
        scale, shift = self.film(t, genre_ids)
        x_norm = self.norm2(x)
        x = x + x_norm * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x

class DiT(nn.Module):
    """
    Full Diffusion Transformer for genre transformation with flow matching.
    """
    
    def __init__(
        self,
        patch_dim=256,
        embed_dim=512,
        num_blocks=12,
        num_heads=8,
        num_genres=10,
        hidden_dim=2048,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_genres = num_genres
        
        # Patch projection (adapt this to your input)
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        
        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, num_genres, hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, patch_dim)
    
    def forward(self, x, t, genre_ids):
        """
        Args:
            x: (B, L, patch_dim) or (B, L, embed_dim) input
            t: (B,) timesteps in [0, 1]
            genre_ids: (B,) or (B, 2) genre indices
        
        Returns:
            (B, L, patch_dim) transformed features
        """
        # Project to embedding dimension
        if x.shape[-1] != self.embed_dim:
            x = self.patch_proj(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t, genre_ids)
        
        # Output projection
        x = self.norm(x)
        x = self.out_proj(x)
        
        return x
