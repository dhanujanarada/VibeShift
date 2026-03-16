"""
models2 — VibeShift models operating directly on mel spectrograms.

Input / output shape throughout: (B, 1, n_mels, T)
    B       — batch size
    1       — mono channel
    n_mels  — frequency bins (default 100, matches utills/mel.py)
    T       — time frames

Genre IDs (match utills/mel.py and configs/genres.yaml):
    0 — non-rock / classical
    1 — rock / punk
    2 — synth
"""

from .film_conditioner_mel import FiLMConditionerMel
from .dit_mel import DiTMel, DiTBlockMel
from .flow_mel import FlowMatchingMel

__all__ = [
    "FiLMConditionerMel",
    "DiTBlockMel",
    "DiTMel",
    "FlowMatchingMel",
]
