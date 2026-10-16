"""Note-level transcription accuracy against reference annotations.

Protocol (results/guitarset05-configs-001): references and
estimates are restricted to notes intersecting [0, window) and clipped to it;
`mir_eval.transcription` one-to-one matching with 50 ms onset / 50 cent pitch
tolerance, then additionally offset ratio 0.2 (min 50 ms); micro scores sum matches
over clips, macro scores average clip F1. Drum notes are excluded from estimates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mir_eval
import numpy as np


def guitarset_reference(jams_path: Path) -> list[tuple[float, float, float]]:
    """(onset, offset, fractional MIDI pitch) from all six `note_midi` annotations."""
    d = json.load(open(jams_path))
    return [(n["time"], n["time"] + n["duration"], float(n["value"]))
            for a in d["annotations"] if a["namespace"] == "note_midi" for n in a["data"]]


def guitarset_manifest(root: Path, player: str = "05", kind: str = "comp",
                       seconds: float = 30.0) -> list[dict]:
    rows = []
    for wav in sorted((root / "audio_mono-mic").glob(f"{player}_*_{kind}_mic.wav")):
        clip = wav.name.removesuffix("_mic.wav")
        rows.append({"id": clip, "audio": str(wav), "duration": seconds,
                     "reference": str(root / "annotation" / f"{clip}.jams")})
    return rows


def _clip(notes, window):
    kept = [(s, min(e, window), p) for s, e, p in notes if s < window and e > 0]
    if not kept:
        return np.zeros((0, 2)), np.zeros(0)
    arr = np.array(kept, dtype=float)
    return arr[:, :2], mir_eval.util.midi_to_hz(arr[:, 2])


def score_clip(ref, est, window: float) -> dict:
    ri, rp = _clip(ref, window)
    ei, ep = _clip(est, window)
    out = {"reference_notes": len(rp), "estimated_notes": len(ep)}
    for name, offset_ratio in (("onset", None), ("onset_offset", 0.2)):
        if len(rp) and len(ep):
            matches = len(mir_eval.transcription.match_notes(
                ri, rp, ei, ep, onset_tolerance=0.05, pitch_tolerance=50.0,
                offset_ratio=offset_ratio, offset_min_tolerance=0.05, strict=False))
        else:
            matches = 0
        p = matches / len(ep) if len(ep) else 0.0
        r = matches / len(rp) if len(rp) else 0.0
        out.update({f"{name}_matches": matches, f"{name}_precision": p, f"{name}_recall": r,
                    f"{name}_f1": 2 * p * r / (p + r) if p + r else 0.0})
    return out


def evaluate(manifest: Path, out: Path, reference: str = "guitarset") -> dict:
    loaders = {"guitarset": guitarset_reference}
    rows, per_clip = [json.loads(line) for line in open(manifest) if line.strip()], []
    for row in rows:
        rec = json.load(open(out / "clips" / f"{row['id']}.json"))
        window = min(float(row.get("duration") or rec["duration"]), rec["duration"])
        est = [(n["start"], n["end"], n["pitch"]) for n in rec["notes"] if n["instrument"] != "drums"]
        m = score_clip(loaders[reference](Path(row["reference"])), est, window)
        per_clip.append({"id": row["id"], "audio_seconds": rec["duration"],
                         **{f"{k}_seconds": v for k, v in rec["timing"].items()}, **m})
    summary: dict = {"clips": len(per_clip),
                     "reference_notes": sum(c["reference_notes"] for c in per_clip),
                     "estimated_notes": sum(c["estimated_notes"] for c in per_clip)}
    for name in ("onset", "onset_offset"):
        tp = sum(c[f"{name}_matches"] for c in per_clip)
        p, r = tp / max(summary["estimated_notes"], 1), tp / max(summary["reference_notes"], 1)
        summary[name] = {"micro_precision": p, "micro_recall": r,
                         "micro_f1": 2 * p * r / (p + r) if p + r else 0.0,
                         "macro_f1": float(np.mean([c[f"{name}_f1"] for c in per_clip]))}
    audio = sum(c["audio_seconds"] for c in per_clip)
    summary["model_seconds_per_audio_minute"] = {
        k: round(sum(c.get(f"{k}_seconds", 0) for c in per_clip) / (audio / 60), 2)
        for k in ("separate", "tag", "transcribe")}
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_clip[0]))
        w.writeheader()
        w.writerows(per_clip)
    json.dump(summary, open(out / "metrics.json", "w"), indent=1)
    return summary
