"""Pooled per-layer encoder features for a manifest of clips.

    <out>/<encoder>.npy    (N, L, D) float16 in manifest order
    <out>/<encoder>.json   ids, splits, encoder description, timing
    <out>/<encoder>.DONE
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..audio import load_mono
from ..encoders import build


class _Clips(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], root: Path, sr: int):
        self.rows, self.root, self.sr = rows, root, sr

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return i, load_mono(self.root / r["path"], self.sr, r.get("offset"), r.get("duration"))


def extract(encoder: str, rows: list[dict], audio_root: Path, out: Path,
            device: str = "cuda", workers: int = 8) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    safe = encoder.replace(":", "-")
    if (out / f"{safe}.DONE").exists():
        return out / f"{safe}.npy"
    enc = build(encoder).load(device)
    loader = torch.utils.data.DataLoader(_Clips(rows, audio_root, enc.sample_rate),
                                         batch_size=None, num_workers=workers,
                                         prefetch_factor=4 if workers else None)
    feats, t0 = None, time.time()
    for n, (i, y) in enumerate(loader):
        y = np.asarray(y, dtype=np.float32)          # the loader's collate returns a tensor
        if len(y) < enc.sample_rate * 0.1:
            raise RuntimeError(f"clip {rows[i]['id']} shorter than 0.1 s")
        pooled = enc.layer_features(y).float().cpu().numpy()
        if not np.isfinite(pooled).all():
            raise RuntimeError(f"non-finite features for {rows[i]['id']}")
        if feats is None:
            feats = np.lib.format.open_memmap(out / f"{safe}.npy.tmp", mode="w+",
                                              dtype=np.float16, shape=(len(rows),) + pooled.shape)
        feats[i] = pooled.astype(np.float16)
        if n % 1000 == 0:
            print(f"{encoder} {n}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    feats.flush()
    del feats
    (out / f"{safe}.npy.tmp").rename(out / f"{safe}.npy")
    json.dump({"encoder": enc.describe(), "ids": [r["id"] for r in rows],
               "splits": [r["split"] for r in rows], "seconds": round(time.time() - t0, 1)},
              open(out / f"{safe}.json", "w"))
    (out / f"{safe}.DONE").touch()
    print(f"{encoder}: {len(rows)} clips in {time.time() - t0:.0f}s", flush=True)
    return out / f"{safe}.npy"
