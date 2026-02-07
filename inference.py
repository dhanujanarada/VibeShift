import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Tuple
import dac
from models.dit import DiT
from models.flow import FlowMatching
from omegaconf import OmegaConf


class VibeShiftInference:
    """Inference pipeline for VibeShift genre transformation"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load DAC codec
        print("Loading DAC codec...")
        model_path = dac.utils.download(model_type="44khz")
        self.dac_model = dac.DAC.load(model_path).to(device)
        self.dac_model.eval()
        
        # Initialize DiT model
        print("Loading DiT model...")
        config_path = Path(__file__).parent / "configs" / "dit.yaml"
        config = OmegaConf.load(config_path)
        
        self.dit_model = DiT(
            input_dim=config.dit_model.input_dim,
            embed_dim=config.dit_model.embed_dim,
            num_blocks=config.dit_model.num_blocks,
            num_heads=config.dit_model.num_heads,
            num_genres=config.dit_model.num_genres,
            hidden_dim=config.dit_model.hidden_dim,
            dropout=config.dit_model.dropout
        ).to(device)
        
        # Initialize Flow Matching model
        self.flow_model = FlowMatching(self.dit_model).to(device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'flow_model_state' in checkpoint:
            self.flow_model.load_state_dict(checkpoint['flow_model_state'])
        elif 'model_state_dict' in checkpoint:
            self.flow_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.flow_model.load_state_dict(checkpoint)
        
        self.flow_model.eval()
        print("✓ Model loaded successfully!")
    
    def load_audio(self, audio_path: str, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
            sr = target_sr
        
        return waveform, sr
    
    def encode_to_dac(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Encode audio to DAC latent space.
        
        Args:
            waveform: Audio waveform tensor (1, samples)
            sr: Sample rate
            
        Returns:
            DAC latent embeddings (1, T, 768)
        """
        with torch.no_grad():
            # Prepare audio for DAC
            waveform = waveform.to(self.device)
            
            # DAC expects specific format
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # Add batch dimension
            
            # Encode to latent space
            z, codes, latents, commitment_loss, codebook_loss = self.dac_model.encode(waveform, sr)
            
            # Get the latent embeddings (before quantization)
            # z shape: (B, latent_dim, T)
            latents = z.transpose(1, 2)  # (B, T, latent_dim)
            
        return latents
    
    def decode_from_dac(self, latents: torch.Tensor, sr: int = 44100) -> torch.Tensor:
        """
        Decode DAC latents back to audio.
        
        Args:
            latents: DAC latent embeddings (1, T, 768)
            sr: Sample rate for output
            
        Returns:
            Audio waveform (1, 1, samples)
        """
        with torch.no_grad():
            # Transpose back to DAC format: (B, latent_dim, T)
            z = latents.transpose(1, 2)
            
            # Decode from latent space
            audio = self.dac_model.decode(z)
            
        return audio
    
    def transform_audio(
        self,
        audio_path: str,
        output_path: str,
        target_genre: int = 1,  # 1 = rock
        num_steps: int = 50,
        method: str = "heun"
    ) -> str:
        """
        Transform audio to target genre.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save output audio
            target_genre: Target genre ID (0=classical, 1=rock, 2=unknown)
            num_steps: Number of sampling steps
            method: Sampling method ('euler' or 'heun')
            
        Returns:
            Path to output audio file
        """
        print(f"Transforming {audio_path} to genre {target_genre}...")
        
        # Load audio
        print("Loading audio...")
        waveform, sr = self.load_audio(audio_path, target_sr=44100)
        
        # Encode to DAC latents
        print("Encoding to DAC latents...")
        source_latents = self.encode_to_dac(waveform, sr)
        
        # Transform using flow matching
        print(f"Transforming with {method} method ({num_steps} steps)...")
        with torch.no_grad():
            if method == "heun":
                transformed_latents = self.flow_model.sample_heun(
                    source_latents, target_genre, num_steps=num_steps
                )
            else:
                transformed_latents = self.flow_model.sample_euler(
                    source_latents, target_genre, num_steps=num_steps
                )
        
        # Decode back to audio
        print("Decoding to audio...")
        output_audio = self.decode_from_dac(transformed_latents, sr)
        
        # Save output
        print(f"Saving to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove batch dimension and save
        torchaudio.save(
            str(output_path),
            output_audio.squeeze(0).cpu(),
            sr
        )
        
        print("✓ Transformation complete!")
        return str(output_path)


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint_path> <audio_path> [output_path]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output_rock.wav"
    
    # Initialize inference pipeline
    inference = VibeShiftInference(checkpoint_path)
    
    # Transform audio
    result = inference.transform_audio(
        audio_path,
        output_path,
        target_genre=1,  # Rock
        num_steps=50,
        method="heun"
    )
    
    print(f"✓ Output saved to: {result}")


if __name__ == "__main__":
    main()
