"""Frozen audio encoders that turn a clip into one pooled vector per layer.

An encoder is registered under a name and must provide `sample_rate`, `load(device)`
and `layer_features(wav) -> (L, D)` (every hidden layer averaged over time). The
instrument tagger and the encoder benchmark only depend on that interface, so a new
encoder is one registry entry.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import torch


class Encoder(Protocol):
    name: str
    sample_rate: int

    def load(self, device: str) -> "Encoder": ...

    def layer_features(self, wav: np.ndarray) -> torch.Tensor: ...

    def describe(self) -> dict: ...


def _matpac(variant: str) -> Callable[[], Encoder]:
    def make():
        from .matpac import Matpac
        return Matpac(variant)
    return make


def _hidden(key: str) -> Callable[[], Encoder]:
    def make():
        from .hidden import ENCODERS
        return ENCODERS[key]()
    return make


REGISTRY: dict[str, Callable[[], Encoder]] = {
    "matpac-music": _matpac("music"),
    "matpac-general": _matpac("general_audio"),
    **{k: _hidden(k) for k in ("mert", "muq", "pupum2d", "af3-whisper", "afnext-whisper",
                               "qwen3-omni-aut")},
}


def build(name: str) -> Encoder:
    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()
