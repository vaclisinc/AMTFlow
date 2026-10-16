"""MATPAC++ (Quelennec et al. 2025) through its transformers port on the HF Hub."""

from __future__ import annotations

import numpy as np
import torch

REPOS = {
    "music": ("auriankelen/matpac_music", "a7e10eb8d14c764673f4364884fb06c9f9f88887"),
    "general_audio": ("auriankelen/matpac_general_audio", "c19b99f7a85d3805d0eafbfde69f986f4ab76566"),
}


class Matpac:
    sample_rate = 16000

    def __init__(self, variant: str = "music"):
        self.variant = variant
        self.repo, self.revision = REPOS[variant]
        self.name = f"matpac-{variant.split('_')[0]}"
        self.model = None
        self.device = "cpu"

    def load(self, device: str) -> "Matpac":
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.repo, revision=self.revision,
                                               trust_remote_code=True).eval().to(device)
        self.device = device
        return self

    @torch.inference_mode()
    def layer_features(self, wav: np.ndarray) -> torch.Tensor:
        """(12, 3840): each block's output, mean over time ("precise" inference)."""
        x = torch.from_numpy(wav)[None].to(self.device)
        out = self.model(input_values=x, inference_type="precise", pull_time_dimension=True)
        return torch.stack(out.hidden_states)[:, 0]

    def describe(self) -> dict:
        return {"name": self.name, "repo": self.repo, "revision": self.revision,
                "sample_rate": self.sample_rate, "n_layers": 12, "dim": 3840,
                "pooling": "mean over time per layer, inference_type=precise"}
