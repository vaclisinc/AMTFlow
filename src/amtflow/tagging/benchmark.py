"""OpenMIC-2018 encoder benchmark: manifest -> features -> probes -> report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from . import openmic
from .features import extract
from .probe import average_precision, f1_scores, probe


def t_ci95(values) -> float:
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def run_extract(encoder: str, manifest_dir: Path, audio_root: Path, out: Path,
                smoke: int = 0, workers: int = 8) -> None:
    rows, _, _ = openmic.read_manifest(manifest_dir)
    if smoke:
        rows = [r for s in ("train", "val", "test") for r in [r for r in rows if r["split"] == s][:smoke]]
    extract(encoder, rows, audio_root, out / "features", workers=workers)


def run_probe(encoder: str, manifest_dir: Path, out: Path, head: str,
              seeds: list[int], epochs: int, lrs: tuple[float, ...] | None = None,
              probes_name: str = "probes") -> dict:
    safe = encoder.replace(":", "-")
    feat_dir = out / "features"
    meta = json.load(open(feat_dir / f"{safe}.json"))
    rows, rel, mask = openmic.read_manifest(manifest_dir)
    idx = {r["id"]: i for i, r in enumerate(rows)}
    order = [idx[i] for i in meta["ids"]]
    X = torch.from_numpy(np.load(feat_dir / f"{safe}.npy"))
    from .probe import LRS
    lrs = tuple(lrs or LRS)
    runs = [probe(X, meta["splits"], rel[order], mask[order], head, s, epochs, lrs=lrs) for s in seeds]
    weights = [r.pop("_weights") for r in runs]
    res = {
        "encoder": encoder, "encoder_description": meta["encoder"], "head": head,
        "classes": list(openmic.CLASSES), "epochs": epochs,
        "n": {s: meta["splits"].count(s) for s in ("train", "val", "test")},
        **{f"test_{k}_mean": float(np.mean([r["test"][k] for r in runs])) for k in ("map", "macro_f1", "micro_f1")},
        **{f"test_{k}_ci95": t_ci95([r["test"][k] for r in runs]) for k in ("map", "macro_f1", "micro_f1")},
        "val_map_mean": float(np.mean([r["val"]["map"] for r in runs])),
        "layers": [r["layer"] for r in runs], "lrs": [r["lr"] for r in runs],
        "lr_grid": list(lrs), "runs": runs,
    }
    probe_dir = out / probes_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(probe_dir / f"{safe}.{head}.json", "w"), indent=1)
    best = int(np.argmax([r["val"]["map"] for r in runs]))
    torch.save({**weights[best], "encoder": encoder, "classes": list(openmic.CLASSES),
                "seed": runs[best]["seed"]}, probe_dir / f"{safe}.{head}.pt")
    print(f"{encoder} {head}: test mAP {res['test_map_mean']:.4f} ± {res['test_map_ci95']:.4f} "
          f"macro-F1 {res['test_macro_f1_mean']:.4f} layers {res['layers']}", flush=True)
    return res


def run_matpac_official(manifest_dir: Path, out: Path, probe_path: Path,
                        variant: str = "music") -> dict:
    """Score the MATPAC++ authors' released OpenMIC linear probe on our splits.

    The probe reads the time-pooled last layer, i.e. layer 11 of the matpac features
    already extracted; sigmoid (multi-label) scores, not the softmax of `probe_forward`.
    """
    name = f"matpac-{variant.split('_')[0]}"
    feat_dir = out / "features"
    meta = json.load(open(feat_dir / f"{name}.json"))
    rows, rel, mask = openmic.read_manifest(manifest_dir)
    idx = {r["id"]: i for i, r in enumerate(rows)}
    order = [idx[i] for i in meta["ids"]]
    X = np.load(feat_dir / f"{name}.npy", mmap_mode="r")
    state = torch.load(probe_path, map_location="cpu", weights_only=True)
    w, b = state["probe.weight"], state["probe.bias"]
    res = {"encoder": name, "head": "official-openmic-linear", "probe": str(probe_path),
           "classes": list(openmic.CLASSES)}
    for split in ("val", "test"):
        sel = [i for i, s in enumerate(meta["splits"]) if s == split]
        x = torch.from_numpy(np.asarray(X[sel, -1], dtype=np.float32))
        logits = x @ w.T + b
        y = torch.from_numpy((rel[order][sel] >= 0.5).astype(np.float32))
        m = torch.from_numpy(mask[order][sel].astype(np.float32))
        ap = average_precision(logits, y, m)
        macro, micro, per_f1 = f1_scores(logits, y, m)
        res[split] = {"map": float(ap.nanmean()), "macro_f1": float(macro),
                      "micro_f1": float(micro), "ap": ap.tolist(), "f1": per_f1.tolist()}
    (out / "probes").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "probes" / f"{name}.official.json", "w"), indent=1)
    print(f"{name} official probe: test mAP {res['test']['map']:.4f} "
          f"macro-F1 {res['test']['macro_f1']:.4f}", flush=True)
    return res


def report(out: Path, probes_name: str = "probes") -> list[dict]:
    rows = []
    for p in sorted((out / probes_name).glob("*.json")):
        r = json.load(open(p))
        if "runs" in r:
            rows.append({"encoder": r["encoder"], "head": r["head"],
                         "test_map": r["test_map_mean"], "test_map_ci95": r["test_map_ci95"],
                         "test_macro_f1": r["test_macro_f1_mean"],
                         "test_macro_f1_ci95": r["test_macro_f1_ci95"],
                         "test_micro_f1": r["test_micro_f1_mean"],
                         "val_map": r["val_map_mean"], "layers": r["layers"], "seeds": len(r["runs"])})
        else:
            rows.append({"encoder": r["encoder"], "head": r["head"], "test_map": r["test"]["map"],
                         "test_map_ci95": None, "test_macro_f1": r["test"]["macro_f1"],
                         "test_macro_f1_ci95": None, "test_micro_f1": r["test"]["micro_f1"],
                         "val_map": r["val"]["map"], "layers": [11], "seeds": 1})
    return sorted(rows, key=lambda r: -r["test_map"])


def run_export_head(encoder: str, manifest_dir: Path, out: Path, head: str, target: Path,
                    probes_name: str = "probes") -> dict:
    """Calibrate the benchmark's selected probe for `encoder` and write a head file."""
    from .heads import export_head

    safe = encoder.replace(":", "-")
    meta = out / "features" / f"{safe}.json"
    rows, rel, mask = openmic.read_manifest(manifest_dir)
    idx = {r["id"]: i for i, r in enumerate(rows)}
    order = [idx[i] for i in json.load(open(meta))["ids"]]
    return export_head(out / probes_name / f"{safe}.{head}.pt", out / "features" / f"{safe}.npy",
                       meta, rel[order], mask[order], target)
