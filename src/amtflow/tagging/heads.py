"""OpenMIC instrument heads on frozen encoders: export, calibrate, load.

A head file (`.pt`) holds everything `ProbeTagger` needs: encoder name, layer,
train-set feature normalization, head weights (linear or MARBLE MLP), class names,
per-class thresholds calibrated on OpenMIC validation, and provenance.

Calibration per class (a recall-weighted weak-label recipe): over 201 evenly spaced
candidate thresholds, maximize F-beta (beta 2, recall-weighted) among thresholds with
precision >= 0.25 (all thresholds if none qualify); ties prefer higher recall, then a
lower threshold. Only observed (clip, class) labels count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BETA, MIN_PRECISION, CANDIDATES = 2.0, 0.25, 201
# Published heads live in the AMTFlow Hub repo; the revision is pinned for reproducibility.
HUB_REPO = "vaclis/amtflow"
HUB_REVISION = "4fdb58169d8ff405c31e3264daa88b749b6b52a8"


def threshold_fbeta(scores: np.ndarray, positives: np.ndarray, beta: float = BETA,
                    minimum_precision: float = MIN_PRECISION,
                    candidate_count: int = CANDIDATES) -> dict:
    lo, hi = float(scores.min()), float(scores.max())
    cands = [lo] if lo == hi else np.linspace(lo, hi, candidate_count).tolist()
    b2, results = beta * beta, []
    for t in cands:
        pred = scores >= t
        tp = int((pred & positives).sum())
        fp = int((pred & ~positives).sum())
        fn = int((~pred & positives).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fb = (1 + b2) * p * r / (b2 * p + r) if b2 * p + r else 0.0
        results.append({"threshold": t, "precision": p, "recall": r, "fbeta": fb})
    pool = [x for x in results if x["precision"] >= minimum_precision] or results
    return max(pool, key=lambda x: (x["fbeta"], x["recall"], -x["threshold"]))


@dataclass
class Head:
    encoder: str
    layer: int
    kind: str
    classes: list[str]
    thresholds: list[float]
    clip_seconds: float
    mu: torch.Tensor
    sd: torch.Tensor
    state: dict

    @torch.no_grad()
    def logits(self, layer_features: torch.Tensor) -> torch.Tensor:
        """(..., L, D) layer features -> (..., C) logits."""
        x = (layer_features[..., self.layer, :].float() - self.mu) / self.sd
        s = self.state
        if self.kind == "mlp":
            x = F.relu(x @ s["w1"] + s["b1"][0])
        return x @ s["w2"] + s["b2"][0]

    def __call__(self, layer_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(layer_features))


def load_head(spec: str | Path, device: str = "cpu") -> Head:
    """A head file path, or a published head name such as `openmic/pupum2d-mlp`:
    looked up as `$AMTFLOW_HEADS/<name>.safetensors`, else downloaded on first use from
    the AMTFlow Hub repo at its pinned revision."""
    import os

    path = Path(spec)
    if not path.exists():
        local = Path(os.environ.get("AMTFLOW_HEADS", "/nonexistent")) / f"{spec}.safetensors"
        if local.exists():
            path = local
        else:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(HUB_REPO, f"heads/{spec}.safetensors", revision=HUB_REVISION))
    d = read_head_file(path)
    return Head(d["encoder"], int(d["layer"]), d["head"], list(d["classes"]),
                [float(x) for x in d["thresholds"]], float(d.get("clip_seconds", 10.0)),
                d["mu"].to(device).float(), d["sd"].to(device).float(),
                {k: v.to(device).float() for k, v in d["state"].items()})


def read_head_file(path: Path) -> dict:
    """A head from `.safetensors` (tensors + JSON metadata; safe to load from the
    internet) or a local `.pt` written by older exports."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["amtflow"])
            t = {k: f.get_tensor(k) for k in f.keys()}
        return {**meta, "mu": t.pop("mu"), "sd": t.pop("sd"), "state": t}
    return torch.load(path, map_location="cpu", weights_only=False)


def write_head_file(bundle: dict, out: Path) -> None:
    from safetensors.torch import save_file
    tensors = {"mu": bundle["mu"], "sd": bundle["sd"],
               **{k: v.contiguous() for k, v in bundle["state"].items()}}
    meta = {k: v for k, v in bundle.items() if k not in ("mu", "sd", "state")}
    save_file({k: v.float().contiguous() for k, v in tensors.items()}, str(out),
              metadata={"amtflow": json.dumps(meta)})


def export_head(probe_file: Path, features: Path, meta: Path, relevance: np.ndarray,
                mask: np.ndarray, out: Path, clip_seconds: float = 10.0) -> dict:
    """Calibrate a benchmark probe on the validation split and write a head file."""
    d = torch.load(probe_file, map_location="cpu", weights_only=False)
    m = json.load(open(meta))
    val = np.array([s == "val" for s in m["splits"]])
    X = torch.from_numpy(np.asarray(np.load(features, mmap_mode="r")[val], dtype=np.float32))
    head = Head(d["encoder"], int(d["layer"]), d["head"], list(d["classes"]),
                [0.0] * len(d["classes"]), clip_seconds, d["mu"].float(), d["sd"].float(),
                {k: v.float() for k, v in d["state"].items()})
    probs = head(X).numpy()
    y, obs = relevance[val] >= 0.5, mask[val].astype(bool)
    calib = {}
    for j, c in enumerate(head.classes):
        o = obs[:, j]
        calib[c] = threshold_fbeta(probs[o, j], y[o, j]) if y[o, j].any() else {"threshold": 0.5}
    thresholds = [calib[c]["threshold"] for c in head.classes]
    bundle = {**{k: d[k] for k in ("layer", "head", "classes", "mu", "sd", "state", "seed")},
              "encoder": d["encoder"],
              "thresholds": thresholds, "clip_seconds": clip_seconds,
              "calibration": {"split": "openmic-2018 val (80/20 of official train, seed 0)",
                              "beta": BETA, "minimum_precision": MIN_PRECISION,
                              "candidates": CANDIDATES, "per_class": calib},
              "encoder_description": m["encoder"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".safetensors":
        write_head_file(bundle, out)
    else:
        torch.save(bundle, out)
    return {"encoder": bundle["encoder"], "layer": int(d["layer"]), "head": d["head"],
            "thresholds": dict(zip(head.classes, [round(t, 4) for t in thresholds]))}
