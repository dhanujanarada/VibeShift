import torch
import torch.nn as nn

from utills.embedding import MelPatchEmbedding, RoPEEmbedding, TimeEmbedding
from models2.film_conditioner_mel import FiLMConditionerMel


class DiTBlockMel(nn.Module):
    """
    Single DiT block for mel spectrograms.

    Identical structure to models/dit.py DiTBlock but uses FiLMConditionerMel
    and operates on flattened mel-patch token sequences.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_genres: int,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention with RoPE
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEEmbedding(embed_dim, num_heads, dropout)

        # FiLM conditioning
        self.norm2 = nn.LayerNorm(embed_dim)
        self.film = FiLMConditionerMel(embed_dim, num_genres, hidden_dim)

        # Feed-forward
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        genre_ids: torch.Tensor,
        attn_mask=None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, N, embed_dim)  — patch token sequence
            t_emb:     (B, embed_dim)     — pre-embedded timestep
            genre_ids: (B,) long
            attn_mask: optional

        Returns:
            (B, N, embed_dim)
        """
        # Self-attention
        x = x + self.attn(self.norm1(x), attn_mask)

        # FiLM modulation
        x_norm = self.norm2(x)
        scale, shift = self.film(t_emb, genre_ids)                        # (B, D)
        x = x + x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        return x


class DiTMel(nn.Module):
    """
    Diffusion Transformer that operates directly on mel spectrograms.

    Input / output: (B, 1, n_mels, T)

    Pipeline:
        mel  →  MelPatchEmbedding  →  N tokens  →  DiTBlocks  →  unpatch  →  mel velocity
    """

    # Default mel params matching utills/mel.py (sr=24000, n_mels=100, hop=256)
    DEFAULT_N_MELS       = 100
    DEFAULT_PATCH_HEIGHT = 10    # splits 100 mel bins into 10 freq patches
    DEFAULT_PATCH_WIDTH  = 16    # 16-frame time patches

    def __init__(
        self,
        n_mels: int = DEFAULT_N_MELS,
        patch_height: int = DEFAULT_PATCH_HEIGHT,
        patch_width: int = DEFAULT_PATCH_WIDTH,
        embed_dim: int = 512,
        num_blocks: int = 8,
        num_heads: int = 8,
        num_genres: int = 2,     # real genres; null token added internally
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        in_channels: int = 1,
    ):
        super().__init__()

        assert n_mels % patch_height == 0, (
            f"n_mels ({n_mels}) must be divisible by patch_height ({patch_height})"
        )

        self.n_mels        = n_mels
        self.patch_height  = patch_height
        self.patch_width   = patch_width
        self.embed_dim     = embed_dim
        self.num_blocks    = num_blocks
        self.num_genres    = num_genres
        self.in_channels   = in_channels

        # Number of frequency patches (fixed)
        self.n_freq_patches = n_mels // patch_height

        # ── Sub-modules ──────────────────────────────────────────────────────

        # Patch tokeniser: (B, 1, n_mels, T) → (B, N, embed_dim)
        self.patch_embed = MelPatchEmbedding(
            patch_height=patch_height,
            patch_width=patch_width,
            embed_dim=embed_dim,
            in_channels=in_channels,
        )

        # Timestep embedding
        self.time_emb = TimeEmbedding(embed_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlockMel(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_genres=num_genres,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project tokens back to patch pixels  (B, N, patch_dim)
        patch_dim = in_channels * patch_height * patch_width
        self.output_proj = nn.Linear(embed_dim, patch_dim)

        # Zero-init output so model starts as identity at t=0
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ------------------------------------------------------------------
    def _unpatchify(
        self,
        tokens: torch.Tensor,
        n_time_patches: int,
    ) -> torch.Tensor:
        """
        Reconstruct mel from patch tokens.

        Args:
            tokens:        (B, N, C*ph*pw)  where N = n_freq_patches * n_time_patches
            n_time_patches: number of time patches

        Returns:
            (B, C, n_mels, T_trim)
        """
        B = tokens.shape[0]
        ph  = self.patch_height
        pw  = self.patch_width
        nfp = self.n_freq_patches
        C   = self.in_channels

        tokens = tokens.view(B, nfp, n_time_patches, C, ph, pw)
        # → (B, nfp, ph, n_time_patches, pw) after reordering
        tokens = tokens.permute(0, 1, 3, 4, 2, 5).contiguous()
        # → (B, nfp*n_time_patches, ph, pw)  nope — let's do it step by step
        tokens = tokens.permute(0, 1, 4, 2, 5, 3).contiguous()
        # (B, nfp, ph, n_tp, pw, C) — too complex; use explicit reshape

        # Simpler: treat each patch independently then stitch
        tokens = tokens.view(B, nfp, n_time_patches, C, ph, pw)
        # Stitch frequency axis:  (B, nfp*ph, n_tp, C, pw)
        mel = tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        # (B, C, nfp, ph, n_tp, pw)
        mel = mel.view(B, C, nfp * ph, n_time_patches * pw)
        # (B, C, n_mels, T_trim)
        return mel

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        genre_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, 1, n_mels, T)  — noisy/interpolated mel at timestep t
            t:         (B,) floats in [0, 1]
            genre_ids: (B,) long

        Returns:
            predicted velocity: (B, 1, n_mels, T_trim)
                T_trim = T rounded down to nearest multiple of patch_width
        """
        B, C, H, W = x.shape

        # Trim time to multiple of patch_width (MelPatchEmbedding does same internally)
        W_trim = (W // self.patch_width) * self.patch_width
        x = x[:, :, :, :W_trim]
        n_time_patches = W_trim // self.patch_width

        # ── Tokenise ─────────────────────────────────────────────────────────
        tokens = self.patch_embed(x)                    # (B, N, embed_dim)

        # ── Conditioning ─────────────────────────────────────────────────────
        t_emb = self.time_emb(t)                        # (B, embed_dim)
        # Inject timestep into all tokens as a bias (same as models/dit.py)
        tokens = tokens + t_emb.unsqueeze(1)

        # ── Transformer blocks ────────────────────────────────────────────────
        for block in self.blocks:
            tokens = block(tokens, t_emb, genre_ids)

        # ── Output projection ─────────────────────────────────────────────────
        tokens = self.norm(tokens)                      # (B, N, embed_dim)
        tokens = self.output_proj(tokens)               # (B, N, C*ph*pw)

        # ── Unpatch back to mel ───────────────────────────────────────────────
        out = self._unpatchify(tokens, n_time_patches)  # (B, 1, n_mels, T_trim)
        return out
