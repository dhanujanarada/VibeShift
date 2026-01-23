import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    Positional embedding layer for transformers.
    Adds learnable or sinusoidal position embeddings to token embeddings.
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
