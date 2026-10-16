"""Source separators: a clip -> named stems, each with a presence measure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class Stem:
    name: str
    wav: np.ndarray        # mono float32
    sr: int
    coverage: float        # percent of 100 ms blocks above -40 dBFS
    missing: bool = False  # audible, but its audio is unavailable (precomputed stems)


def coverage(mono: np.ndarray, sr: int, floor_db: float = -40.0) -> float:
    block = int(sr * 0.1)
    frames = mono[: len(mono) // block * block].reshape(-1, block)
    if not len(frames):
        return 0.0
    rms = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
    return float((rms > floor_db).mean() * 100)


class NoSeparator:
    """The paper default: transcribe the mixture."""

    sample_rate = None

    def __init__(self, cfg: dict, device: str):
        pass

    def __call__(self, wav: np.ndarray, sr: int, row: dict | None = None) -> list[Stem]:
        return [Stem("mix", wav, sr, coverage(wav, sr))]


class BSRoformer:
    """BS-RoFormer, all stems kept. `checkpoint`/`config` default to jarredou's SW-Fixed
    six-stem model (weights CC-BY-NC-SA 4.0), downloaded on first use."""

    sample_rate = 44100

    def __init__(self, cfg: dict, device: str):
        import torch
        import yaml
        from bs_roformer.inference import SafeLoaderWithTuple
        from bs_roformer.utils import get_model_from_config, load_checkpoint_state
        from ml_collections import ConfigDict

        from .assets import resolve

        config_path = resolve(cfg.get("config"), "bs-roformer-sw-fixed.yaml")
        ckpt_path = resolve(cfg.get("checkpoint"), "bs-roformer-sw-fixed.ckpt")
        self.config = ConfigDict(yaml.load(config_path.read_text(), Loader=SafeLoaderWithTuple))
        # bs_roformer's iSTFT returns whole hops: keep chunks a multiple of the hop length
        # (MVSep's 53-stem config uses 441000 / 882000 samples with hop 512)
        hop = int(self.config.model.get("stft_hop_length", 512))
        for sec in ("audio", "inference"):
            if sec in self.config and "chunk_size" in self.config[sec]:
                self.config[sec].chunk_size = -(-int(self.config[sec].chunk_size) // hop) * hop
        model = get_model_from_config("bs_roformer", self.config)
        model.load_state_dict(load_checkpoint_state(ckpt_path, map_location=torch.device("cpu")))
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.first_chunk_time = None

    def __call__(self, wav: np.ndarray, sr: int, row: dict | None = None) -> list[Stem]:
        import torch
        from bs_roformer.utils import demix_track
        from .audio import resample

        y = resample(wav, sr, self.sample_rate)
        stereo = torch.from_numpy(np.stack([y, y]))
        with torch.no_grad():
            res, self.first_chunk_time = demix_track(self.config, self.model, stereo, self.device,
                                                     self.first_chunk_time)
        stems = []
        for name, value in res.items():
            mono = np.asarray(value).mean(axis=0).astype(np.float32)
            stems.append(Stem(name, mono, self.sample_rate, coverage(mono, self.sample_rate)))
        return stems


class Precomputed:
    """Stems already separated, e.g. by `amtflow separate` (see `save_pool`).

    Reads `<root>/clips/<id>.json` (per-stem `stem_presence`, `stem_paths`). A stem
    without audio whose coverage reached `saved_min_coverage` was audible but not
    saved (older pools lack `other`) and is returned as missing, never as silent.
    """

    def __init__(self, cfg: dict, device: str):
        self.root = Path(cfg["root"])
        self.saved_min = float(cfg.get("saved_min_coverage", 5.0))

    def __call__(self, wav: np.ndarray, sr: int, row: dict | None = None) -> list[Stem]:
        import json
        import soundfile as sf

        if not row or not row.get("id"):
            raise ValueError("precomputed stems need a manifest id")
        rec = json.load(open(self.root / "clips" / f"{row['id']}.json"))
        stems = []
        for name, pres in rec["stem_presence"].items():
            cov = float(pres["coverage_above_minus_40_percent"])
            path = rec.get("stem_paths", {}).get(name)
            if path:
                y, ssr = sf.read(path, dtype="float32", always_2d=False)
                mono = y if y.ndim == 1 else y.mean(1)
                # Saved metadata describes the original separation, not necessarily
                # the bytes on disk (resampling, stale metadata, or replaced audio).
                stems.append(Stem(name, mono, ssr, coverage(mono, ssr)))
            else:
                stems.append(Stem(name, np.zeros(0, np.float32), sr, cov, missing=cov >= self.saved_min))
        return stems


REGISTRY: dict[str, Callable] = {"none": NoSeparator, "bsroformer": BSRoformer,
                                 "precomputed": Precomputed}


def build_separator(cfg: dict, device: str):
    if cfg["name"] not in REGISTRY:
        raise KeyError(f"unknown separator {cfg['name']!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[cfg["name"]](cfg, device)


def save_pool(cfg: dict, manifest: Path, out: Path, device: str = "cuda",
              shard: int = 0, num_shards: int = 1) -> dict:
    """Separate every manifest clip once and write a pool `Precomputed` can read:
    `out/stems/<id>/<stem>.flac` (mono, separator rate) and `out/clips/<id>.json`.
    Every stem is saved, silent ones included. Existing clips are skipped (resume)."""
    import json
    import soundfile as sf

    from .audio import load_mono

    sep = build_separator(cfg, device)
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    rows = [r for i, r in enumerate(rows) if i % num_shards == shard]
    (out / "clips").mkdir(parents=True, exist_ok=True)
    done = 0
    for row in rows:
        target = out / "clips" / f"{row['id']}.json"
        if target.exists():
            continue
        sr = getattr(sep, "sample_rate", None) or 44100
        wav = load_mono(Path(row["audio"]), sr, row.get("offset"), row.get("duration"))
        stem_dir = out / "stems" / row["id"]
        stem_dir.mkdir(parents=True, exist_ok=True)
        rec = {"id": row["id"], "audio": row["audio"], "separator": cfg,
               "stem_presence": {}, "stem_paths": {}}
        for s in sep(wav, sr, row):
            path = stem_dir / f"{s.name}.flac"
            sf.write(path, s.wav, s.sr)
            rec["stem_presence"][s.name] = {"coverage_above_minus_40_percent": round(s.coverage, 2)}
            rec["stem_paths"][s.name] = str(path)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(target)
        done += 1
    return {"shard": shard, "done": done, "of": len(rows)}
