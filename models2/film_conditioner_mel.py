import torch
import torch.nn as nn
from utills.embedding import GenreEmbedding


class FiLMConditionerMel(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioner for mel-spectrogram models.

    Mirrors models/film_conditioner.py but is a clean standalone class so
    models2 has no circular dependency on models/.

    Conditions each DiT block on:
        - pre-embedded timestep  (B, embed_dim)
        - genre id               (B,) long

    Args:
        embed_dim:  DiT hidden dimension
        num_genres: number of real genre classes  (null CFG token added automatically
                    at index num_genres, same convention as models/dit.py)
        hidden_dim: intermediate MLP width
        shallow:    True → legacy 2-Linear arch (no middle hidden→hidden layer)
    """

    def __init__(
        self,
        embed_dim: int,
        num_genres: int,
        hidden_dim: int = 512,
        shallow: bool = False,
    ):
        super().__init__()

        # Genre embedding — num_genres + 1 entries (last = CFG null token)
        self.genre_emb = GenreEmbedding(num_genres, embed_dim, use_sinusoidal=False)

        if shallow:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim * 2),
                nn.GELU(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim * 2),
            )

    def forward(
        self,
        time_emb: torch.Tensor,
        genre_ids: torch.Tensor,
    ):
        """
        Args:
            time_emb:  (B, embed_dim)
            genre_ids: (B,) long

        Returns:
            scale: (B, embed_dim)
            shift: (B, embed_dim)
        """
        genre_emb = self.genre_emb(genre_ids)                 # (B, embed_dim)
        cond = torch.cat([time_emb, genre_emb], dim=-1)       # (B, embed_dim*2)
        out = self.proj(cond)                                   # (B, embed_dim*2)
        scale, shift = out.chunk(2, dim=-1)                    # each (B, embed_dim)
        return scale, shift
