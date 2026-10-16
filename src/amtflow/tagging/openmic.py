"""OpenMIC-2018 (Humphrey et al. 2018) instrument-presence benchmark data.

Labels are the official aggregated relevances `Y_true` with observation mask
`Y_mask` from `openmic-2018.npz`: an observed entry is positive when its relevance is
>= 0.5 (the dataset's own convention) and unobserved entries are excluded from loss
and metrics. The official `split01` test partition is reserved for test; the official
train partition is split 80/20 by track into train/val (seed 0).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

CLASSES = ("accordion", "banjo", "bass", "cello", "clarinet", "cymbals", "drums",
           "flute", "guitar", "mallet_percussion", "mandolin", "organ", "piano",
           "saxophone", "synthesizer", "trombone", "trumpet", "ukulele", "violin", "voice")
VAL_FRACTION = 0.2
SPLIT_SEED = 0


def _partition(path: Path) -> list[str]:
    with open(path) as f:
        return [row[0] for row in csv.reader(f) if row and row[0] != "sample_key"]


def build_manifest(root: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Rows {id, path, split} plus (N, 20) relevance and (N, 20) observed mask."""
    data = np.load(root / "openmic-2018.npz", allow_pickle=True)
    keys = [str(k) for k in data["sample_key"]]
    class_map = json.load(open(root / "class-map.json"))
    order = [class_map[c] for c in CLASSES]           # npz column of each CLASSES entry
    y_true, y_mask = data["Y_true"][:, order], data["Y_mask"][:, order].astype(bool)
    index = {k: i for i, k in enumerate(keys)}

    train = _partition(root / "partitions" / "split01_train.csv")
    test = _partition(root / "partitions" / "split01_test.csv")
    tracks = sorted({k.split("_")[0] for k in train})
    rng = np.random.default_rng(SPLIT_SEED)
    val_tracks = set(rng.permutation(tracks)[: round(len(tracks) * VAL_FRACTION)].tolist())

    rows, rel, mask = [], [], []
    for split, members in (("train", train), ("test", test)):
        for key in members:
            s = "val" if split == "train" and key.split("_")[0] in val_tracks else split
            i = index[key]
            rows.append({"id": key, "path": f"audio/{key[:3]}/{key}.ogg", "split": s})
            rel.append(y_true[i])
            mask.append(y_mask[i])
    return rows, np.stack(rel).astype(np.float32), np.stack(mask)


def write_manifest(root: Path, out: Path) -> dict:
    rows, rel, mask = build_manifest(root)
    missing = [r["id"] for r in rows if not (root / r["path"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} OpenMIC audio files missing, e.g. {missing[:3]}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez(out / "labels.npz", relevance=rel, mask=mask, classes=np.array(CLASSES))
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    summary = {"root": str(root), "classes": list(CLASSES), "counts": counts,
               "positive_threshold": 0.5, "val_fraction": VAL_FRACTION,
               "split_seed": SPLIT_SEED, "observed_fraction": float(mask.mean())}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    return summary


def read_manifest(out: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in open(out / "manifest.jsonl")]
    labels = np.load(out / "labels.npz")
    return rows, labels["relevance"], labels["mask"]
