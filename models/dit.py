import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from utills.embedding import RoPEEmbedding, GenreEmbedding, TimeEmbedding, MelPatchEmbedding
from models.film_conditioner import FiLMConditioner
import os

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'dit.yaml')
dit_config = OmegaConf.load(CONFIG_PATH)

class DiTBlock(nn.Module):
    """Transformer block with RoPE attention and FiLM conditioning"""
    
    def __init__(self, dim, num_heads, num_genres, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Attention with RoPE
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEEmbedding(dim, num_heads, dropout) 
        
        # FiLM conditioning (takes pre-embedded time)
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
    
    def forward(self, x, t_emb, genre_ids, attn_mask=None):
        """
        Args:
            x: (B, L, D) input
            t_emb: (B, D) pre-embedded time from TimeEmbedding
            genre_ids: (B,) genre indices
            attn_mask: optional attention mask
        
        Returns:
            (B, L, D) output
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), attn_mask)
        
        # FiLM conditioning with pre-embedded time
        scale, shift = self.film(t_emb, genre_ids)  # Pass embedded time
        x_norm = self.norm2(x)
        x = x + x_norm * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
        
        # MLP with residual
        x = x + self.mlp(self.norm3(x))
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for genre transformation with flow matching.
    Inputs and outputs mel spectrograms: (B, C, n_mels, time_steps)
    """
    
    def __init__(
        self,
        patch_dim=None,
        embed_dim=None,
        num_blocks=None,
        num_heads=None,
        num_genres=None,
        hidden_dim=None,
        dropout=None,
        config=None,
        use_mel_patches=True,
        patch_height=8,
        patch_width=8,
        in_channels=1
    ):
        super().__init__()
        
        # Use provided config or load from file
        if config is None:
            config = dit_config.dit_model
        
        # Use provided parameters or fall back to config
        self.patch_dim = patch_dim or config.patch_dim
        self.embed_dim = embed_dim or config.embed_dim
        self.num_blocks = num_blocks or config.num_blocks
        self.num_heads = num_heads or config.num_heads
        self.num_genres = num_genres or config.num_genres
        self.hidden_dim = hidden_dim or config.hidden_dim
        self.dropout = dropout or config.dropout
        self.use_mel_patches = use_mel_patches
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.in_channels = in_channels
        
        # Time embedding for diffusion timesteps (SINGLE SOURCE OF TRUTH)
        self.time_emb = TimeEmbedding(self.embed_dim)
        
        # Mel patch embedding
        if use_mel_patches:
            self.patch_embedding = MelPatchEmbedding(
                patch_height=patch_height,
                patch_width=patch_width,
                embed_dim=self.embed_dim,
                in_channels=in_channels
            )
            # Output projection back to patch space
            self.out_proj = nn.Linear(self.embed_dim, patch_height * patch_width * in_channels)
        else:
            # Linear projection for pre-patchified inputs
            self.patch_proj = nn.Linear(self.patch_dim, self.embed_dim)
            self.out_proj = nn.Linear(self.embed_dim, self.patch_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.embed_dim, self.num_heads, self.num_genres, self.hidden_dim, self.dropout)
            for _ in range(self.num_blocks)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x, t, genre_ids):
        """
        Args:
            x: (B, C, n_mels, time_steps) mel spectrogram
            t: (B,) timesteps in [0, 1]
            genre_ids: (B,) genre indices for target genre (e.g., 1 for rock)
        
        Returns:
            (B, C, n_mels, time_steps) transformed mel spectrogram
        """
        # Store original shape for reconstruction
        original_shape = None
        
        if self.use_mel_patches:
            B, C, H, W = x.shape
            original_shape = (B, C, H, W)
            
            # Convert mel to patches
            x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        else:
            # Linear projection for pre-patchified inputs
            if x.shape[-1] != self.embed_dim:
                x = self.patch_proj(x)
        
        # Embed time ONCE (single source of truth)
        t_emb = self.time_emb(t)  # (B, embed_dim)
        
        # Add time embedding to all patches
        x = x + t_emb.unsqueeze(1)  # Broadcast to (B, num_patches, embed_dim)
        
        # Apply transformer blocks with pre-embedded time
        for block in self.blocks:
            x = block(x, t_emb, genre_ids)  # Pass t_emb, not raw t
        
        # Output projection
        x = self.norm(x)
        x = self.out_proj(x)  # (B, num_patches, patch_dim)
        
        # Reconstruct mel spectrogram from patches
        if self.use_mel_patches and original_shape is not None:
            B, C, H, W = original_shape
            num_patches_h = H // self.patch_height
            num_patches_w = W // self.patch_width
            
            # Reshape from (B, num_patches, patch_size) to (B, H', W', C, patch_h, patch_w)
            x = x.reshape(B, num_patches_h, num_patches_w, C, self.patch_height, self.patch_width)
            
            # Permute to (B, C, H', patch_h, W', patch_w)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            
            # Reshape to original mel shape (B, C, H, W)
            x = x.reshape(B, C, H, W)
        
        return x