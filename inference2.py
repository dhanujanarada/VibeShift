import logging
logger = logging.getLogger("VibeShiftInference")
logging.basicConfig(level=logging.INFO)
"""
VibeShift Genre Transfer Inference Pipeline
============================================
Complete pipeline: Audio → MIDI → Synth → DAC latent → Genre transfer → Audio
Optimized for CPU inference with memory management.
"""

import torch
import torch.nn as nn
from pathlib import Path
import librosa
import soundfile as sf
import torchaudio
from typing import Optional, Dict, Tuple, Union
from tqdm import tqdm
import gc
import warnings

# Import project utilities
from utills.audio_midi_converter import AudioMidiConverter
from models.flow import FlowMatching
from models.dit import DiT
import dac
from dac.utils import load_model


class VibeShiftInference:
    """
    Complete inference pipeline for music genre transfer.

    Pipeline stages:
    1. Audio → MIDI conversion (using basic-pitch)
    2. MIDI → Synthesized audio (using soundfont)
    3. Audio → DAC latent encoding
    4. Genre transfer using flow matching
    5. DAC latent → Audio decoding

    Optimized for CPU inference with automatic memory management.
    """

    # Genre mapping
    GENRE_MAP = {
        1: "Rock",
        0: "Classical",
        2: "Synth"
    }

    def __init__(
        self,
        checkpoint_path: str,
        soundfont_path: str,
        dit_config: Optional[Dict] = None,
        device: str = "cpu"
    ):
        """
        Initialize inference pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            soundfont_path: Path to soundfont file for MIDI synthesis
            dit_config: DiT model configuration dict (must match training)
            device: Device to use (default: "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.soundfont_path = Path(soundfont_path)
        self.device = device

        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.soundfont_path.exists():
            raise FileNotFoundError(f"Soundfont not found: {self.soundfont_path}")

        # Default DiT configuration (must match training)
        self.dit_config = dit_config or {
            "input_dim": 1024,
            "embed_dim": 512,
            "num_blocks": 8,
            "num_heads": 8,
            "num_genres": 3,
            "hidden_dim": 2048
        }

        # Model components (lazy loading for memory efficiency)
        self.midi_converter = None
        self.dac_model = None
        self.flow_model = None

        logger.info(f"✓ VibeShiftInference initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Checkpoint: {self.checkpoint_path.name}")
        logger.info(f"  Soundfont: {self.soundfont_path.name}")

    def _load_midi_converter(self):
        """Load MIDI converter (lightweight component)."""
        if self.midi_converter is None:
            logger.info("Loading MIDI converter...")
            self.midi_converter = AudioMidiConverter(
                soundfont_path=str(self.soundfont_path)
            )
            logger.info("✓ MIDI converter loaded")

    def _load_dac_model(self):
        """Load DAC encoder/decoder model."""
        if self.dac_model is None:
            logger.info("Loading DAC model (44kHz)...")
            self.dac_model = load_model(model_type="44khz")
            self.dac_model.to(self.device)
            self.dac_model.eval()

            # Disable gradients for inference
            for param in self.dac_model.parameters():
                param.requires_grad = False

            logger.info("✓ DAC model loaded")

    def _unload_dac_model(self):
        """Unload DAC model to free memory."""
        if self.dac_model is not None:
            logger.info("Unloading DAC model...")
            del self.dac_model
            self.dac_model = None
            gc.collect()
            logger.info("✓ DAC model unloaded")

    def _load_flow_model(self):
        """Load trained genre transfer flow matching model."""
        if self.flow_model is None:
            logger.info(f"Loading genre transfer model...")
            logger.info(f"  Checkpoint: {self.checkpoint_path}")

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )

            # Extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', 'unknown')
                logger.info(f"  Epoch: {epoch}")
                logger.info(f"  Training loss: {loss}")
            else:
                state_dict = checkpoint

            # Create DiT and FlowMatching models
            dit = DiT(**self.dit_config)
            self.flow_model = FlowMatching(dit)

            # Load weights
            self.flow_model.load_state_dict(state_dict)
            self.flow_model.to(self.device)
            self.flow_model.eval()

            # Disable gradients for inference
            for param in self.flow_model.parameters():
                param.requires_grad = False

            n_params = sum(p.numel() for p in self.flow_model.parameters()) / 1e6
            logger.info(f"✓ Flow model loaded ({n_params:.2f}M parameters)")

    def _unload_flow_model(self):
        """Unload flow model to free memory."""
        if self.flow_model is not None:
            logger.info("Unloading flow model...")
            del self.flow_model
            self.flow_model = None
            gc.collect()
            logger.info("✓ Flow model unloaded")

    def preprocess_audio(
        self,
        audio_path: str,
        target_sr: int = 44100
    ) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio file.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: 44100 Hz)

        Returns:
            audio_tensor: (1, T) mono audio tensor
            sample_rate: Sample rate
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio: {audio_path.name}")

        # Load audio
        audio_tensor, sr = torchaudio.load(str(audio_path))
        logger.info(f"  Original: {audio_tensor.shape[0]} channels, {sr} Hz")

        # Resample if needed
        if sr != target_sr:
            logger.info(f"  Resampling to {target_sr} Hz...")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            audio_tensor = resampler(audio_tensor)
            sr = target_sr

        # Convert to mono
        if audio_tensor.shape[0] > 1:
            logger.info(f"  Converting to mono...")
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        duration = audio_tensor.shape[1] / sr
        logger.info(f"✓ Preprocessed: mono, {sr} Hz, {duration:.2f}s")

        return audio_tensor, sr

    def audio_to_synth(
        self,
        audio_path: str,
        output_dir: str
    ) -> Tuple[str, str]:
        """
        Convert audio to MIDI and synthesize back to audio.

        Args:
            audio_path: Input audio file path
            output_dir: Directory to save outputs

        Returns:
            midi_path: Path to generated MIDI file
            synth_audio_path: Path to synthesized audio file
        """
        self._load_midi_converter()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_name = Path(audio_path).stem

        logger.info(f"{'='*70}")
        logger.info("STEP 1: Audio → MIDI → Synthesized Audio")
        logger.info(f"{'='*70}")

        # Audio → MIDI
        logger.info(f"[1/2] Converting audio to MIDI...")
        midi_path = self.midi_converter.audio_to_midi(
            audio_path=audio_path,
            output_dir=str(output_dir)
        )
        logger.info(f"✓ MIDI saved: {midi_path}")

        # MIDI → Synth audio
        logger.info(f"[2/2] Synthesizing MIDI with soundfont...")
        synth_audio_path = output_dir / f"{input_name}_synth.wav"
        self.midi_converter.midi_to_audio(
            midi_path=midi_path,
            output_path=str(synth_audio_path)
        )
        logger.info(f"✓ Synth audio saved: {synth_audio_path}")

        return str(midi_path), str(synth_audio_path)

    def encode_audio_to_latent(
        self,
        audio_path: str
    ) -> torch.Tensor:
        """
        Encode audio to DAC latent representation.

        Args:
            audio_path: Path to audio file

        Returns:
            latent: (B, T, D) latent tensor
        """
        self._load_dac_model()

        logger.info(f"{'='*70}")
        logger.info("STEP 2: Audio → DAC Latent Encoding")
        logger.info(f"{'='*70}")

        # Load and preprocess
        audio_tensor, sr = self.preprocess_audio(audio_path)
        audio_tensor = audio_tensor.to(self.device)

        # Add batch dimension: (C, T) → (B, C, T)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Encode
        logger.info(f"Encoding to latent representation...")
        with torch.no_grad():
            dac_output = self.dac_model.encode(audio_tensor)
            z = dac_output[0]  # (B, D, T)

            # Reshape to (B, T, D) for model compatibility
            if z.dim() == 3:
                latent = z.permute(0, 2, 1)
            else:
                latent = z

        logger.info(f"✓ Latent encoded")
        logger.info(f"  Shape: {latent.shape}")
        logger.info(f"  (Batch, Time steps, Embedding dim)")

        return latent

    @torch.no_grad()
    def transfer_genre(
        self,
        source_latent: torch.Tensor,
        target_genre: Union[int, str],
        num_steps: int = 150,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Transfer latent to target genre using flow matching.

        Args:
            source_latent: (B, T, D) source latent tensor
            target_genre: Target genre (int ID or string name)
            num_steps: Number of flow matching steps
            show_progress: Show progress bar

        Returns:
            transferred_latent: (B, T, D) transferred latent
        """
        self._load_flow_model()

        logger.info(f"{'='*70}")
        logger.info("STEP 3: Genre Transfer via Flow Matching")
        logger.info(f"{'='*70}")

        # Convert genre name to ID if needed
        if isinstance(target_genre, str):
            genre_name_to_id = {v: k for k, v in self.GENRE_MAP.items()}
            if target_genre not in genre_name_to_id:
                raise ValueError(
                    f"Unknown genre '{target_genre}'. "
                    f"Valid genres: {list(genre_name_to_id.keys())}"
                )
            target_genre_id = genre_name_to_id[target_genre]
            genre_name = target_genre
        else:
            target_genre_id = target_genre
            genre_name = self.GENRE_MAP.get(target_genre_id, str(target_genre_id))

        logger.info(f"Target genre: {genre_name} (ID: {target_genre_id})")
        logger.info(f"Flow steps: {num_steps}")
        logger.info(f"Source latent: {source_latent.shape}")

        B, T, D = source_latent.shape
        source_latent = source_latent.to(self.device)

        # Genre tensor
        genre_ids = torch.full(
            (B,),
            target_genre_id,
            dtype=torch.long,
            device=self.device
        )

        # Flow matching: interpolate from source (t=0) to target (t=1)
        dt = 1.0 / num_steps
        x = source_latent.clone()

        iterator = tqdm(
            range(num_steps),
            desc=f"Transferring to {genre_name}",
            disable=not show_progress
        )

        for step in iterator:
            t = torch.full((B,), step * dt, device=self.device)

            # Predict velocity field
            v_pred = self.flow_model.dit(x, t, genre_ids)

            # Euler integration step
            x = x + v_pred * dt

        logger.info(f"✓ Genre transfer complete")
        logger.info(f"  Output latent: {x.shape}")

        return x

    def decode_latent_to_audio(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Decode DAC latent to audio waveform.

        Args:
            latent: (B, T, D) latent tensor

        Returns:
            audio: (T,) audio numpy array
            sample_rate: Sample rate
        """
        self._load_dac_model()

        logger.info(f"{'='*70}")
        logger.info("STEP 4: DAC Latent → Audio Decoding")
        logger.info(f"{'='*70}")

        latent = latent.to(self.device)

        # Reshape to (B, D, T) for decoder
        if latent.dim() == 3:
            z_for_decode = latent.permute(0, 2, 1)
        else:
            z_for_decode = latent

        logger.info(f"Decoding latent to audio...")
        logger.info(f"  Input latent: {latent.shape}")

        with torch.no_grad():
            decoded_audio = self.dac_model.decode(z_for_decode)

            # Convert to mono numpy: (B, C, T) → (T,)
            if decoded_audio.dim() == 3:
                decoded_audio = decoded_audio.squeeze(0)  # Remove batch
            if decoded_audio.dim() == 2:
                decoded_audio = decoded_audio.mean(dim=0)  # Convert to mono

            audio_np = decoded_audio.cpu().numpy()

        sample_rate = self.dac_model.sample_rate
        duration = len(audio_np) / sample_rate

        logger.info(f"✓ Audio decoded")
        logger.info(f"  Samples: {len(audio_np)}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Duration: {duration:.2f}s")

        return audio_np, sample_rate

    def process(
        self,
        input_audio_path: str,
        target_genre: Union[int, str],
        output_dir: str,
        num_steps: int = 50,
        save_intermediates: bool = True
    ) -> Dict[str, str]:
        """
        Run complete genre transfer pipeline.

        Args:
            input_audio_path: Path to input audio file
            target_genre: Target genre (int ID or string name)
                         0/"synth", 1/"classical", 2/"rock"
            output_dir: Directory to save all outputs
            num_steps: Number of flow matching steps (default: 50)
            save_intermediates: Save intermediate files (MIDI, synth, latents)

        Returns:
            output_paths: Dictionary of all generated file paths
        """
        logger.info(f"{'='*70}")
        logger.info("VIBESHIFT GENRE TRANSFER PIPELINE")
        logger.info(f"{'='*70}")
        logger.info(f"Input: {input_audio_path}")
        logger.info(f"Target genre: {target_genre}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'='*70}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_name = Path(input_audio_path).stem

        # Determine genre name for file naming
        if isinstance(target_genre, str):
            genre_name = target_genre
        else:
            genre_name = self.GENRE_MAP.get(target_genre, str(target_genre))

        output_paths = {}

        # STEP 1: Audio → MIDI → Synth
        midi_path, synth_audio_path = self.audio_to_synth(
            audio_path=input_audio_path,
            output_dir=str(output_dir)
        )

        if save_intermediates:
            output_paths['midi'] = midi_path
            output_paths['synth_audio'] = synth_audio_path

        # STEP 2: Synth → DAC Latent
        source_latent = self.encode_audio_to_latent(synth_audio_path)

        if save_intermediates:
            synth_latent_path = output_dir / f"{input_name}_synth_latent.pt"
            torch.save(source_latent.cpu(), synth_latent_path)
            output_paths['synth_latent'] = str(synth_latent_path)
            logger.info(f"✓ Synth latent saved: {synth_latent_path.name}")

        # Unload DAC before loading flow model
        self._unload_dac_model()

        # STEP 3: Genre Transfer
        transferred_latent = self.transfer_genre(
            source_latent=source_latent,
            target_genre=target_genre,
            num_steps=num_steps
        )

        if save_intermediates:
            transferred_latent_path = output_dir / f"{input_name}_to_{genre_name}_latent.pt"
            torch.save(transferred_latent.cpu(), transferred_latent_path)
            output_paths['transferred_latent'] = str(transferred_latent_path)
            logger.info(f"✓ Transferred latent saved: {transferred_latent_path.name}")

        # Unload flow model before reloading DAC
        self._unload_flow_model()

        # STEP 4: Latent → Audio
        audio_np, sample_rate = self.decode_latent_to_audio(transferred_latent)

        # Save final audio
        output_audio_path = output_dir / f"{input_name}_to_{genre_name}.wav"
        sf.write(str(output_audio_path), audio_np, sample_rate)
        output_paths['output_audio'] = str(output_audio_path)

        logger.info(f"✓ Final audio saved: {output_audio_path.name}")

        # Cleanup DAC
        self._unload_dac_model()

        # Print summary
        logger.info(f"{'='*70}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Generated files:")
        for key, path in output_paths.items():
            path_obj = Path(path)
            size_mb = path_obj.stat().st_size / (1024 * 1024)
            logger.info(f"  {key:20s}: {path_obj.name:40s} ({size_mb:6.2f} MB)")

        logger.info(f"All outputs saved to: {output_dir}")

        return output_paths

    def cleanup(self):
        """
        Unload all models and free memory.
        Call this when done with all inference tasks.
        """
        logger.info(f"{'='*70}")
        logger.info("Cleaning up resources...")
        logger.info(f"{'='*70}")

        self._unload_dac_model()
        self._unload_flow_model()

        if self.midi_converter is not None:
            del self.midi_converter
            self.midi_converter = None

        gc.collect()
        logger.info("✓ Cleanup complete")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.cleanup()


