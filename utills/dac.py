import gc
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _this_dir]

import torch
from torch.nn.utils import parametrizations
import torch.nn.utils as nn_utils

# Force new weight_norm API for dependencies that still call the deprecated helper.
nn_utils.weight_norm = parametrizations.weight_norm

import dac
from audiotools import AudioSignal
from typing import Optional, List
import numpy as np

class DACLatentProcessor:
    """
    Batch process audio files to DAC latent representations.
    """
    
    def __init__(
        self, 
        model_type: str = "44khz",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_quantizers: Optional[int] = None
    ):
        """
        Initialize DAC processor.
        
        Args:
            model_type: DAC model variant ('44khz', '24khz', '16khz')
            device: 'cuda' or 'cpu'
            n_quantizers: Number of RVQ codebooks to use (None = all)
        """
        self.model_type = model_type
        self.device = device
        self.n_quantizers = n_quantizers
        
        # Load pretrained DAC model
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        
    def encode_file(self, audio_path: Path) -> dict:
        """
        Encode single audio file to DAC latents.
        
        """
        signal = AudioSignal(str(audio_path))
        
        with torch.no_grad():
            signal = signal.to(self.device)
            x = self.model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = self.model.encode(x, self.n_quantizers)
            
        return {
            'z': z.cpu().numpy(),           # Quantized latents (B, D, T)
            'codes': codes.cpu().numpy(),   # Discrete codes (B, num_codebooks, T)
            'latents': latents.cpu().numpy(), # Pre-quantization latents
            'sample_rate': signal.sample_rate,
            'original_length': signal.signal_length
        }
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        extensions: List[str] = ['.wav', '.mp3', '.flac']
    ):
    
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Gather all audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.glob(f'**/*{ext}'))
        
        print(f"Found {len(audio_files)} audio files")
        
        for i, file_path in enumerate(audio_files):
            try:
                print(f"[{i+1}/{len(audio_files)}] Processing {file_path.name}...", end=" ")
                
                # Encode to latents
                latent_data = self.encode_file(file_path)
                
                # Save as .npz (preserves directory structure)
                relative_path = file_path.relative_to(input_path)
                output_file = output_path / relative_path.with_suffix('.pt')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert DAC latents from (B, D, T) to (B, T, D) for consistency with DiT expectations
                z_tensor = torch.from_numpy(latent_data["z"])  # (B, D, T)
                z_tensor = z_tensor.transpose(1, 2)  # Convert to (B, T, D)
                
                torch.save(
                    {
                        "z": z_tensor,
                        "codes": torch.from_numpy(latent_data["codes"]),
                        "latents": torch.from_numpy(latent_data["latents"]),
                        "sample_rate": latent_data["sample_rate"],
                        "original_length": latent_data["original_length"],
                    },
                    output_file.with_suffix(".pt"),
                )
                
                print(f"Saved to {output_file.name}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        print(f"\nCompleted! Latents saved to {output_dir}")
    
    def load_latents(self, latent_path: str) -> dict:
        """
        Load saved latents from .pt file.
        """
        data = torch.load(latent_path, map_location="cpu")
        return {
            "z": data["z"],
            "codes": data["codes"],
            "latents": data["latents"],
            "sample_rate": int(data["sample_rate"]),
            "original_length": int(data["original_length"]),
        }
    
    def decode_latents(self, z: torch.Tensor) -> np.ndarray:
        """
        Decode latents back to audio waveform.
        
        Args:
            z: Quantized latents (B, D, T) or (B, T, D) - will transpose if needed
        
        Returns:
            Audio waveform as numpy array
        """
        with torch.no_grad():
            z = z.to(self.device)
            # Ensure z is in (B, D, T) format for DAC decoder
            if z.dim() == 3 and z.shape[2] < z.shape[1]:
                # Likely (B, T, D) format, transpose to (B, D, T)
                z = z.transpose(1, 2)
            audio = self.model.decode(z)
        return audio.cpu().numpy()

    def _unload_dac(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
