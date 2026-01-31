"""
Training script for VibeShift: Genre transformation using flow matching
Transforms mel spectrograms from any genre to rock
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import librosa
from tqdm import tqdm

from models.dit import DiT
from models.flow import FlowMatching

# Load config
CONFIG_PATH = Path(__file__).parent / 'configs' / 'dit.yaml'
config = OmegaConf.load(CONFIG_PATH)


class MelSpectrogramDataset(Dataset):
    """Dataset that loads paired mel spectrograms (source -> rock)"""
    
    def __init__(
        self,
        source_dir,
        target_dir,
        n_mels=128,
        max_time_steps=512,
        normalize=True
    ):
        """
        Args:
            source_dir: Directory containing source genre mel .npy files
            target_dir: Directory containing target rock mel .npy files
            n_mels: Number of mel bins
            max_time_steps: Maximum time steps (will pad/crop)
            normalize: Whether to normalize mel specs
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.n_mels = n_mels
        self.max_time_steps = max_time_steps
        self.normalize = normalize
        
        # Collect paired files
        self.source_files = sorted(self.source_dir.glob('**/*.npy'))
        self.target_files = sorted(self.target_dir.glob('**/*.npy'))
        
        # For unpaired training, use same length
        self.length = min(len(self.source_files), len(self.target_files))
        
        print(f"Dataset: {self.length} pairs")
    
    def _process_mel(self, mel_path):
        """Load and process mel spectrogram"""
        mel_spec = np.load(mel_path)  # (n_mels, time_steps)
        
        # Pad or crop to fixed length
        if mel_spec.shape[1] < self.max_time_steps:
            # Pad with zeros
            pad_width = self.max_time_steps - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Crop to max length
            mel_spec = mel_spec[:, :self.max_time_steps]
        
        # Normalize to [-1, 1]
        if self.normalize:
            mel_min, mel_max = mel_spec.min(), mel_spec.max()
            if mel_max > mel_min:
                mel_spec = 2 * (mel_spec - mel_min) / (mel_max - mel_min + 1e-8) - 1
        
        # Add channel dimension
        mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0)  # (1, n_mels, time_steps)
        
        return mel_spec
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load source and target mels
        source_mel = self._process_mel(self.source_files[idx])
        target_mel = self._process_mel(self.target_files[idx % len(self.target_files)])
        
        return {
            'source': source_mel,  # (1, n_mels, time_steps)
            'target': target_mel,  # (1, n_mels, time_steps)
            'genre_id': torch.tensor(1, dtype=torch.long)  # 1 = rock
        }


class VibeShiftTrainer:
    """Trainer for genre transformation using flow matching"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # Initialize DiT model with mel patch support
        dit_model = DiT(
            config=config.dit_model,
            use_mel_patches=True,
            patch_height=config.training.patch_height,
            patch_width=config.training.patch_width,
            in_channels=1
        ).to(device)
        
        # Wrap in FlowMatching
        self.model = FlowMatching(dit_model).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6
        )
        
        self.best_loss = float('inf')
        self.epoch = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            source_mel = batch['source'].to(self.device)  # (B, 1, n_mels, time_steps)
            target_mel = batch['target'].to(self.device)  # (B, 1, n_mels, time_steps)
            genre_ids = batch['genre_id'].to(self.device)  # (B,)
            
            # Compute flow matching loss
            loss = self.model(source_mel, target_mel, genre_ids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            source_mel = batch['source'].to(self.device)
            target_mel = batch['target'].to(self.device)
            genre_ids = batch['genre_id'].to(self.device)
            
            loss = self.model(source_mel, target_mel, genre_ids)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    @torch.no_grad()
    def generate_sample(self, source_mel, genre_id=1, num_steps=50):
        """Generate rock version of source mel"""
        self.model.eval()
        
        # Add batch dimension if needed
        if source_mel.dim() == 3:
            source_mel = source_mel.unsqueeze(0)
        
        source_mel = source_mel.to(self.device)
        
        # Sample using Heun's method
        rock_mel = self.model.sample_heun(source_mel, genre_id, num_steps)
        
        return rock_mel
    
    def save_checkpoint(self, path, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, train_loader, val_loader, save_dir='checkpoints'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n{'='*60}")
        print(f"Starting VibeShift Training")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epoch, self.config.training.epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                print(f"✓ New best model! (val_loss: {val_loss:.4f})")
            
            self.save_checkpoint(
                save_dir / f'checkpoint_epoch_{epoch+1}.pt',
                is_best=is_best
            )
            
            print(f"{'-'*60}\n")


def convert_audio_to_mel(audio_dir, output_dir, n_mels=128, n_fft=2048, hop_length=512, sr=22050):
    """Convert audio files to mel spectrograms"""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(audio_dir.glob('**/*.wav')) + list(audio_dir.glob('**/*.mp3'))
    
    print(f"Converting {len(audio_files)} audio files to mel spectrograms...")
    
    for audio_file in tqdm(audio_files):
        # Load audio
        y, _ = librosa.load(str(audio_file), sr=sr)
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save
        relative_path = audio_file.relative_to(audio_dir)
        output_path = output_dir / relative_path.with_suffix('.npy')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, mel_spec_db)
    
    print(f"✓ Saved mel spectrograms to {output_dir}")


if __name__ == '__main__':
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Optional: Convert audio to mel spectrograms first
    # convert_audio_to_mel(
    #     audio_dir='data/nonrock_audio_files',
    #     output_dir='data/mel_specs/source',
    #     n_mels=config.training.n_mels
    # )
    # convert_audio_to_mel(
    #     audio_dir='data/rock_audio_files',
    #     output_dir='data/mel_specs/target',
    #     n_mels=config.training.n_mels
    # )
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MelSpectrogramDataset(
        source_dir='data/mel_specs/source/train',
        target_dir='data/mel_specs/target/train',
        n_mels=config.training.n_mels,
        max_time_steps=512
    )
    
    val_dataset = MelSpectrogramDataset(
        source_dir='data/mel_specs/source/val',
        target_dir='data/mel_specs/target/val',
        n_mels=config.training.n_mels,
        max_time_steps=512
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    trainer = VibeShiftTrainer(config, device=device)
    trainer.train(train_loader, val_loader, save_dir='checkpoints/vibeshift')
    
    print("\n✓ Training complete!")
