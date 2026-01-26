import math
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t):
        
        # Scale to reasonable range
        t = t * 1000  # Scale up for sinusoidal
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
        
        return emb

class PatchEmbedding(nn.Module):
   
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        
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
       
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x


class RoPEEmbedding(nn.Module):
    """Multi-head attention with Rotary Position Embeddings"""
    
    def __init__(self, dim, num_heads, dropout=0.1, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _compute_rope(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()
    
    def _apply_rope(self, x, cos_emb, sin_emb):
        cos_emb = cos_emb[None, None, :, :]
        sin_emb = sin_emb[None, None, :, :]
        x1, x2 = x[..., :self.head_dim//2], x[..., self.head_dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_emb + rotated * sin_emb
    
    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        cos_emb, sin_emb = self._compute_rope(L, x.device)
        q = self._apply_rope(q, cos_emb, sin_emb)
        k = self._apply_rope(k, cos_emb, sin_emb)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        return out


class GenreEmbedding(nn.Module):
  
    
    def __init__(self, num_genres, embed_dim, use_sinusoidal=True):
        super().__init__()
        self.num_genres = num_genres
        self.embed_dim = embed_dim
        self.use_sinusoidal = use_sinusoidal
        
        if use_sinusoidal:
            # Sinusoidal embeddings for smooth genre space
            self.register_buffer('genre_emb', self._create_sinusoidal_embeddings(num_genres, embed_dim))
        else:
            # Learnable embeddings
            self.genre_emb = nn.Parameter(torch.randn(num_genres, embed_dim) * 0.02)
    
    def _create_sinusoidal_embeddings(self, num_genres, embed_dim):
        """Create sinusoidal embeddings for genre space."""
        pos = torch.arange(num_genres, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe = torch.zeros(num_genres, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe
    
    def forward(self, genre_ids, interp_weight=None):
        """
        Args:
            genre_ids: (B,) for single genre or (B, 2) for interpolation
            interp_weight: interpolation weight (default 0.5 if None)
        """
        if genre_ids.dim() == 1:
            # Single genre
            return self.genre_emb[genre_ids]
        
        elif genre_ids.dim() == 2 and genre_ids.shape[1] == 2:
            # Interpolation between two genres
            g1, g2 = genre_ids[:, 0], genre_ids[:, 1]
            emb1 = self.genre_emb[g1]
            emb2 = self.genre_emb[g2]
            
            # Use provided weight or default to 0.5
            if interp_weight is None:
                t = 0.5
            else:
                if isinstance(interp_weight, (int, float)):
                    t = interp_weight
                else:
                    t = interp_weight.unsqueeze(-1)
            
            return emb1 * (1 - t) + emb2 * t
        
        else:
            raise ValueError("genre_ids must be (B,) or (B, 2)")
        


class PositionalEmbedding(nn.Module):
    
    def __init__(self, num_positions, embed_dim, sinusoidal=False):
       
        super().__init__()
        self.embed_dim = embed_dim
        self.sinusoidal = sinusoidal
        
        if sinusoidal:
            self.register_buffer('pos_emb', self._create_sinusoidal_embeddings(num_positions, embed_dim))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
            nn.init.normal_(self.pos_emb, std=0.02)
    
    def _create_sinusoidal_embeddings(self, num_positions, embed_dim):
        
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
       
        seq_len = x.shape[1]
        return x + self.pos_emb[:, :seq_len, :]