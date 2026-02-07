"""Chroma-based neutral audio resynthesis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import librosa
import numpy as np
import soundfile as sf


@dataclass
class ChromaResynthesizer:
    """Resynthesize neutral audio from chroma + coarse timing."""

    sr: int = 44
    hop_length: int = 512
    n_mels: int = 80
    octave: int = 4
    normalize_chroma: bool = True

    def __post_init__(self) -> None:
        self.note_freqs = self._note_freqs_for_octave(self.octave)

    @staticmethod
    def _note_freqs_for_octave(octave: int) -> np.ndarray:
        base = np.array(
            [
                261.63,  # C4
                277.18,  # C#4
                293.66,  # D4
                311.13,  # D#4
                329.63,  # E4
                349.23,  # F4
                369.99,  # F#4
                392.00,  # G4
                415.30,  # G#4
                440.00,  # A4
                466.16,  # A#4
                493.88,  # B4
            ],
            dtype=np.float32,
        )
        ratio = 2 ** (octave - 4)
        return base * ratio

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return y, sr

    def compute_chroma(self, y: np.ndarray) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
        )
        if self.normalize_chroma:
            chroma = chroma / (np.max(chroma, axis=0, keepdims=True) + 1e-8)
        return chroma

    def compute_rms(self, y: np.ndarray) -> np.ndarray:
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms = rms / (rms.max() + 1e-8)
        return rms

    def resynthesize(self, chroma: np.ndarray, rms: np.ndarray) -> np.ndarray:
        total_frames = chroma.shape[1]
        if rms.shape[0] != total_frames:
            rms = librosa.util.fix_length(rms, size=total_frames)

        frame_len = self.hop_length
        time = np.arange(frame_len, dtype=np.float32) / float(self.sr)
        sine_bank = np.sin(2 * np.pi * self.note_freqs[:, None] * time[None, :])

        output = np.zeros(total_frames * frame_len, dtype=np.float32)
        for t in range(total_frames):
            frame = (chroma[:, t][:, None] * sine_bank).sum(axis=0)
            frame *= rms[t]
            start = t * frame_len
            output[start : start + frame_len] = frame

        output = output / (np.max(np.abs(output)) + 1e-8)
        return output

    def to_mel(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            f_min=0,
            f_max=8000,
        )
        return mel

    def process_path(
        self,
        audio_path: str,
        output_wav_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        y, sr = self.load_audio(audio_path)
        chroma = self.compute_chroma(y)
        rms = self.compute_rms(y)
        neutral_audio = self.resynthesize(chroma, rms)

        if output_wav_path:
            save_path = output_wav_path
            if os.path.isdir(output_wav_path):
                save_path = os.path.join(output_wav_path, "neutral.wav")
            sf.write(save_path, neutral_audio, sr)

        neutral_mel = self.to_mel(neutral_audio)
        original_mel = self.to_mel(y)
        return neutral_audio, neutral_mel, original_mel, sr
