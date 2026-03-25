"""
vibeshift_inference.py
======================
Self-contained inference class for VibeShift genre transfer.

Full pipeline:
  WAV / MP3  →  basic_pitch  →  MIDI
             →  FluidSynth + TimGM6mb.sf2  →  synth.wav
             →  DAC encode  →  latent (1, T, 1024)
             →  FlowMatching (Heun)
             →  DAC decode  →  output.wav

Accepted input formats:  .mp3  .wav  .flac  .ogg  .m4a

Usage
-----
    pipeline = VibeshiftInference(hf_repo_id="Nyarada/vibeshift-checkpoints")
    result = pipeline.generate("input.mp3", "outputs/job_001")
    # result["input_wav"], result["synth_wav"], result["output_wav"]
"""

import pathlib
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

import dac
from dac.utils import download as dac_download
from huggingface_hub import hf_hub_download

from models.dit import DiT
from models.flow import FlowMatching
from utills.audio_midi_converter import AudioMidiConverter


# ── Constants ────────────────────────────────────────────────────────────────

DAC_NATIVE_DIM  = 1024
TARGET_GENRE_ID = 1         
HEUN_STEPS      = 50

MODEL_CONFIG = dict(
    input_dim  = 768,
    embed_dim  = 512,
    num_blocks = 8,
    num_heads  = 8,
    num_genres = 3,
    hidden_dim = 2048,
    dropout    = 0.1,
)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ── Main class ───────────────────────────────────────────────────────────────

class VibeshiftInference:
    """
    Loads the flow model and DAC codec once at construction time,
    then exposes a single generate() method for any audio input.
    """

    def __init__(
        self,
        hf_repo_id: str,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        hf_repo_id : str
            HuggingFace Hub repo ID, e.g. "Nyarada/vibeshift-checkpoints".
            Two files are downloaded automatically: ``best.pt`` (flow model
            checkpoint) and ``TimGM6mb.sf2`` (FluidSynth soundfont).
        device : str | None
            "cuda" / "cpu" — auto-detected if None.
        """
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_repo_id  = hf_repo_id

        checkpoint_path     = self._hub_download("best_final.pt")
        self.soundfont_path = self._hub_download("TimGM6mb.sf2")

        self._load_flow_model(checkpoint_path)
        self._load_dac()
        self.converter = AudioMidiConverter(self.soundfont_path)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[VibeShift] Ready on {self.device}  ({n_params/1e6:.1f}M params)")

    # ── Hub helper ───────────────────────────────────────────────────────────

    def _hub_download(self, filename: str) -> str:
        """Download *filename* from self.hf_repo_id via the HuggingFace Hub cache."""
        print(f"[VibeShift] Fetching {filename} from {self.hf_repo_id} …")
        local_path = hf_hub_download(repo_id=self.hf_repo_id, filename=filename)
        print(f"[VibeShift] {filename} → {local_path}")
        return local_path

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_flow_model(self, checkpoint_path: str):
        """Load DiT + FlowMatching from checkpoint — called once at startup."""
        ckpt_path = pathlib.Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt       = torch.load(str(ckpt_path), map_location=self.device)
        dit        = DiT(**MODEL_CONFIG)
        self.model = FlowMatching(dit)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        epoch     = ckpt.get("epoch", "?")
        best_loss = ckpt.get("best_loss", float("nan"))
        print(f"[VibeShift] Flow model loaded  epoch={epoch}  best_loss={best_loss:.4f}")

    def _load_dac(self):
        """Download (cached) and load the DAC 44 kHz codec — called once."""
        print("[VibeShift] Loading DAC 44 kHz codec …")
        dac_path       = dac_download(model_type="44khz")
        self.dac_model = dac.DAC.load(dac_path)
        self.dac_model.to(self.device).eval()

        for p in self.dac_model.parameters():
            p.requires_grad_(False)

        self.sr = self.dac_model.sample_rate   # 44100
        print(f"[VibeShift] DAC loaded  sample_rate={self.sr}")

    # ── Audio loading ────────────────────────────────────────────────────────

    def _load_audio(self, audio_path: str):
        """
        Load any audio file to a mono float32 waveform tensor.

        Returns
        -------
        waveform : torch.Tensor  shape (1, T)  on CPU
        sr       : int
        """
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sr

    # ── DAC encode / decode ──────────────────────────────────────────────────

    def _encode(self, waveform: torch.Tensor, sr: int):
        """
        Encode a (1, T) waveform to a DAC latent.

        Returns
        -------
        x0    : (1, T_lat, 1024)  on self.device
        codes : (1, n_codebooks, T_lat)
        """
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)

        wav_in = waveform.unsqueeze(0).to(self.device)   # (1, 1, T)

        with torch.no_grad():
            z, codes, *_ = self.dac_model.encode(wav_in)
            # z     : (1, 1024, T_lat)
            # codes : (1, n_codebooks, T_lat)

        x0 = z.permute(0, 2, 1).float()   # (1, T_lat, 1024)
        return x0, codes

    def _get_decode_weight(
        self,
        x0_flat: torch.Tensor,   # (T, D)  truncated latent on CPU
        codes: torch.Tensor,     # (1, n_cb, T)
    ) -> torch.Tensor:
        """
        Compute pseudo-inverse W_inv (D, 1024) so that:
            x0_flat @ W_inv  ≈  z_native  (T, 1024)

        Only needed when model input_dim < 1024.
        """
        codes = codes.to(self.device)
        with torch.no_grad():
            z_native, _, _ = self.dac_model.quantizer.from_codes(codes)  # (1, 1024, T)
        z_native_t = z_native.permute(0, 2, 1).squeeze(0).float()        # (T, 1024)
        W_inv = torch.linalg.pinv(x0_flat) @ z_native_t                  # (D, 1024)
        return W_inv

    def _decode_latent(
        self,
        lat: torch.Tensor,               # (1, T, D)
        w_inv: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Decode a (1, T, D) flow-model latent to a mono audio numpy array.

        D == 1024  → direct transpose and DAC decode.
        D != 1024  → project to 1024 via w_inv, then decode.

        Returns
        -------
        np.ndarray  shape (T_audio,)  float32 mono
        """
        D = lat.shape[-1]

        if D == DAC_NATIVE_DIM:
            z_dec = lat.permute(0, 2, 1).to(self.device)

        elif w_inv is not None:
            lat_flat = lat.squeeze(0).float()
            z_native = lat_flat @ w_inv
            z_dec    = z_native.T.unsqueeze(0).to(self.device)

        else:
            raise RuntimeError(
                f"Latent dim {D} != {DAC_NATIVE_DIM} and no pseudo-inverse supplied."
            )

        with torch.no_grad():
            audio_out = self.dac_model.decode(z_dec)

        audio_np = audio_out.squeeze().cpu().float().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=0)

        return audio_np

    # ── Main entry point ─────────────────────────────────────────────────────

    def generate(self, input_path: str, output_dir: str) -> dict:
        """
        Full pipeline:
          audio  →  basic_pitch  →  MIDI
                 →  FluidSynth + soundfont  →  synth.wav
                 →  DAC encode
                 →  FlowMatching (Heun)
                 →  DAC decode  →  output.wav

        Also saves the original input as input.wav.

        Returns
        -------
        dict with keys: input_wav, synth_wav, output_wav, sample_rate
        """
        input_path = pathlib.Path(input_path)
        out_dir    = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = input_path.suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Accepted: {sorted(AUDIO_EXTENSIONS)}"
            )

        stem = input_path.stem

        # ── 1. Save original input as WAV ────────────────────────────────────
        input_wav_path = str(out_dir / f"{stem}_input.wav")
        waveform_raw, sr_raw = self._load_audio(str(input_path))
        sf.write(input_wav_path, waveform_raw.squeeze(0).numpy(), sr_raw)
        print(f"[VibeShift] Input saved: {input_wav_path}")

        # ── 2. Audio → MIDI (basic_pitch) ────────────────────────────────────
        print(f"[VibeShift] Audio → MIDI (basic_pitch) …")
        midi_dir  = str(out_dir / "midi")
        midi_path = self.converter.audio_to_midi(str(input_path), midi_dir)

        # ── 3. MIDI → synthesized WAV (FluidSynth) ───────────────────────────
        print(f"[VibeShift] MIDI → synth WAV (FluidSynth) …")
        synth_wav_path = str(out_dir / f"{stem}_synth.wav")
        self.converter.midi_to_audio(midi_path, synth_wav_path)

        # ── 4. Encode synth audio ─────────────────────────────────────────────
        print(f"[VibeShift] Encoding synth audio …")
        waveform, sr  = self._load_audio(synth_wav_path)
        synth_samples = waveform.shape[-1]
        print(f"[VibeShift] {synth_samples / sr:.2f}s synth audio loaded")

        genre_id  = torch.tensor([TARGET_GENRE_ID], device=self.device, dtype=torch.long)
        model_dim = MODEL_CONFIG['input_dim']

        x0, codes = self._encode(waveform, sr)         # (1, T_lat, 1024)

        if x0.shape[-1] != model_dim:
            x0 = x0[..., :model_dim]                   # (1, T_lat, model_dim)

        w_inv = None
        if model_dim != DAC_NATIVE_DIM:
            w_inv = self._get_decode_weight(x0.squeeze(0).cpu().float(), codes)

        # ── 5. FlowMatching (Heun) ────────────────────────────────────────────
        print(f"[VibeShift] Running flow model (Heun, {HEUN_STEPS} steps) …")
        with torch.no_grad():
            x_out = self.model.sample_heun(
                x0, genre_id, num_steps=HEUN_STEPS
            ).float()

        sim = torch.nn.functional.cosine_similarity(
            x0.flatten(), x_out.flatten(), dim=0
        ).item()
        print(f"[VibeShift] x0 vs x_out cosine sim: {sim:.4f}  (1.0=identical, 0=orthogonal)")

        # ── 6. Decode and save output ─────────────────────────────────────────
        audio_output = self._decode_latent(x_out, w_inv)
        audio_output = audio_output[:synth_samples]    # trim any DAC padding

        output_wav_path = str(out_dir / f"{stem}_output.wav")
        sf.write(output_wav_path, audio_output, self.sr)
        print(f"[VibeShift] Output saved: {output_wav_path}  ({len(audio_output)/self.sr:.2f}s)")

        return {
            "input_wav":   input_wav_path,
            "synth_wav":   synth_wav_path,
            "output_wav":  output_wav_path,
            "sample_rate": self.sr,
        }