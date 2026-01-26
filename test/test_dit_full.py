import torch
import torchaudio
import sys
sys.path.append('c:\\Users\\Dhanuja\\Desktop\\Vibeshift\\VibeShift')

from models.dit import DiT
from utills.embedding import PatchEmbedding, RoPEEmbedding

# Load a sample audio file (create one or use existing)
# For now, we'll simulate audio
audio_path = "data\\nonrock_audio_files\\00000.wav"  # Replace with actual audio

try:
    waveform, sr = torchaudio.load(audio_path)
    print(f"✅ Loaded audio: {waveform.shape}")
except:
    print("⚠️ No audio file found. Using synthetic audio.")
    waveform = torch.randn(1, 16000)  # 1 second at 16kHz

# Convert to mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
mel_spec = mel_transform(waveform)
print(f"✅ Mel spectrogram: {mel_spec.shape}")

# Add channel dimension: (B, C, H, W) = (1, 1, 128, 16000)
mel_spec = mel_spec.unsqueeze(1)

# Create patches with larger patch_size to reduce sequence length
patch_embedding = PatchEmbedding(mel_spec.shape[-1], patch_size=16, in_channels=1, embed_dim=256)
patches = patch_embedding(mel_spec)
print(f"✅ Patches: {patches.shape}")

# Initialize DiT
model = DiT(
    patch_dim=256,
    embed_dim=512,
    num_blocks=12,
    num_heads=8,
    num_genres=10,
    hidden_dim=2048
)

# Test different timesteps and genres
print("\n🔄 Testing different configurations:")
for t_val in [0.0, 0.5, 1.0]:
    for genre_id in [0, 5, 9]:
        t = torch.tensor([t_val])
        genre = torch.tensor([genre_id])
        
        output = model(patches, t, genre)
        print(f"  t={t_val}, genre={genre_id} → output shape: {output.shape} ✅")

print("\n✅ All tests passed!")