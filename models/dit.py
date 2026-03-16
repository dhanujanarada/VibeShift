import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from utills.embedding import RoPEEmbedding, GenreEmbedding, TimeEmbedding
from models.film_conditioner import FiLMConditioner
import os

# Load configuration with error handling
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dit.yaml')
try:
    dit_config = OmegaConf.load(CONFIG_PATH)
except FileNotFoundError:
    print(f"Warning: Config file not found at {CONFIG_PATH}. Using default config.")
    dit_config = OmegaConf.create({
        "dit_model": {
            "embed_dim": 1024,      
            "num_blocks": 16,       
            "num_heads": 16,        
            "num_genres": 2,        
                                    
            "hidden_dim": 4096,     
            "dropout": 0.1,         
        }
    })


class DiTBlock(nn.Module):
    """Transformer block with RoPE attention and FiLM conditioning"""
    
    def __init__(self, dim, num_heads, num_genres, hidden_dim=2048, dropout=0.1, film_shallow=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Attention with RoPE
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEEmbedding(dim, num_heads, dropout) 
        
        # FiLM conditioning (takes pre-embedded time)
        self.norm2 = nn.LayerNorm(dim)
        self.film = FiLMConditioner(dim, num_genres, shallow=film_shallow)
        
        # MLP
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, t_emb, genre_ids, attn_mask=None):
        """
        Args:
            x: (B, T, D) input sequence
            t_emb: (B, D) pre-embedded time from TimeEmbedding
            genre_ids: (B,) genre indices
            attn_mask: optional attention mask
        
        Returns:
            (B, T, D) output
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), attn_mask)
        
        x_norm = self.norm2(x)
        scale, shift = self.film(t_emb, genre_ids)
        x_modulated = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x + x_modulated

        # MLP with residual
        x = x + self.mlp(self.norm3(x))
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for genre transformation with flow matching.
    Designed for DAC embeddings: (B, T, latent_dim) where latent_dim=64 for all DAC variants.
    
    Args:
        input_dim: Dimension of the DAC quantized latent z. Fixed at 64 for all DAC
                   model variants (44khz, 24khz, 16khz). Do not change unless using
                   a different codec.
    """
    
    def __init__(
        self,
        input_dim=64,        # DAC quantized latent dimension: D=64 for all DAC variants
        embed_dim=None,
        num_blocks=None,
        num_heads=None,
        num_genres=None,
        hidden_dim=None,
        dropout=None,
        film_shallow=False,  # True for legacy checkpoints without the middle hidden layer
        config=None,
    ):
        super().__init__()
        
        # Use provided config or load from file
        if config is None:
            config = dit_config.dit_model
        
        # Use provided parameters or fall back to config
        self.input_dim = input_dim  # DAC latent dim
        self.embed_dim = embed_dim or config.embed_dim
        self.num_blocks = num_blocks or config.num_blocks
        self.num_heads = num_heads or config.num_heads
        self.num_genres = num_genres or config.num_genres
        self.hidden_dim = hidden_dim or config.hidden_dim
        self.dropout = dropout or config.dropout
        self.film_shallow = film_shallow
        
        # Time embedding for diffusion timesteps
        self.time_emb = TimeEmbedding(self.embed_dim)
        
        # Input projection: DAC latent dim -> embed_dim
        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        
        # Output projection: embed_dim -> DAC latent dim
        self.output_proj = nn.Linear(self.embed_dim, self.input_dim)
        
        # CFG: reserve one extra genre slot as the null / unconditional token.
        # null_genre_id == num_genres (one past the last real genre).
        self.null_genre_id: int = self.num_genres

        # Transformer blocks — embedding table has num_genres + 1 entries
        self.blocks = nn.ModuleList([
            DiTBlock(self.embed_dim, self.num_heads, self.num_genres + 1, self.hidden_dim, self.dropout, film_shallow=self.film_shallow)
            for _ in range(self.num_blocks)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x, t, genre_ids):
       
        # Project input to embed_dim
        x = self.input_proj(x)  # (B, T, embed_dim)
        
        # Embed time
        t_emb = self.time_emb(t)  # (B, embed_dim)
        
        # Add time embedding to all frames
        x = x + t_emb.unsqueeze(1)  # Broadcast to (B, T, embed_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, genre_ids)
        
        # Output projection back to DAC latent space
        x = self.norm(x)
        x = self.output_proj(x)  # (B, T, latent_dim)
        
        return x