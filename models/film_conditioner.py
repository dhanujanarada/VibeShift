import torch
import torch.nn as nn
from utills.embedding import GenreEmbedding, TimeEmbedding


class FiLMConditioner(nn.Module):
    def __init__(self, embed_dim, num_genres, hidden_dim=512):
        super().__init__()
        self.time_emb = TimeEmbedding(embed_dim)
        self.genre_emb = GenreEmbedding(num_genres, embed_dim, use_sinusoidal=True)
        
        # Combine time + genre embeddings
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
    
    def forward(self, t, genre_ids):
        time_emb = self.time_emb(t)
        genre_emb = self.genre_emb(genre_ids)
        
        cond = torch.cat([time_emb, genre_emb], dim=-1)
        out = self.proj(cond)
        
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift