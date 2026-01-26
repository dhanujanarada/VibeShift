import torch
import sys
sys.path.append('c:\\Users\\Dhanuja\\Desktop\\Vibeshift\\VibeShift')

from models.dit import DiT

# Initialize
model = DiT(
    patch_dim=256,
    embed_dim=512,
    num_blocks=12,
    num_heads=8,
    num_genres=10,
    hidden_dim=2048,
    dropout=0.1
)

# Test forward pass
batch_size = 2
seq_length = 100

x = torch.randn(batch_size, seq_length, 256)
t = torch.rand(batch_size)
genre_ids = torch.randint(0, 10, (batch_size,))

output = model(x, t, genre_ids)

print(f"✅ Input shape: {x.shape}")
print(f"✅ Output shape: {output.shape}")
print(f"✅ Model works! No errors.")