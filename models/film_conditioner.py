print("film_conditioner.py STARTED")

import torch
import torch.nn as nn
from utills.embedding import GenreEmbedding

class FiLMConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) for conditioning on time and genre.
    Takes pre-embedded time and genre, outputs scale and shift parameters.
    """
    
    def __init__(self, embed_dim, num_genres, hidden_dim=512):
        super().__init__()
        
        # Genre embedding
        self.genre_emb = GenreEmbedding(num_genres, embed_dim, use_sinusoidal=True)
        
        # Project concatenated embeddings to scale/shift
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
    
    def forward(self, time_emb, genre_ids):
        """
        Args:
            time_emb: (B, embed_dim) pre-embedded time from TimeEmbedding
            genre_ids: (B,) genre indices
        
        Returns:
            scale: (B, embed_dim)
            shift: (B, embed_dim)
        """
        # Get genre embedding
        genre_emb = self.genre_emb(genre_ids)  # (B, embed_dim)
        
        # Concatenate time and genre embeddings
        cond = torch.cat([time_emb, genre_emb], dim=-1)  # (B, embed_dim * 2)
        
        # Project to scale and shift
        out = self.proj(cond)  # (B, embed_dim * 2)
        scale, shift = out.chunk(2, dim=-1)  # Each (B, embed_dim)
        
        return scale, shift