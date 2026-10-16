"""Audio loading shared by every stage."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_mono(path: str | Path, sr: int, offset: float | None = None,
              duration: float | None = None) -> np.ndarray:
    """Mono float32 waveform at `sr`, optionally the [offset, offset+duration) span."""
    y, _ = librosa.load(str(path), sr=sr, mono=True, offset=offset or 0.0, duration=duration)
    return np.ascontiguousarray(y, dtype=np.float32)


def resample(y: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return y
    return np.ascontiguousarray(librosa.resample(y, orig_sr=sr, target_sr=target), dtype=np.float32)
