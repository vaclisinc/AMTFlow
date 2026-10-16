"""MulTTiPop vocal relabelling: which reference tracks carry the sung melody?

Evidence per non-drum reference track:
  A (name)  the track is named as vocals (named_vocal_tracks.json: tracks named vocal / voice /
            vox / singer / singing / choir / backing or harmony vocals / bgv, excluding GM
            patch names such as "Pad", "Lead", "Oohs", "Aahs", "Synth Voice");
  B (audio) its notes coincide with the pitch of the separated vocals stem.

For B, pYIN runs on the BS-RoFormer vocals stem (10 ms hop). A note frame matches when
the stem is voiced and its pitch is within half a semitone of the note, modulo the octave
(arrangers often move the melody by an octave; chance is about 1/12). Per track:
  precision = matched frames / note frames      (how much of the track is sung)
  sung_precision = matched / note frames where the stem is voiced (ignores sections
                   where nobody sings, e.g. an instrumental melody between verses)
  coverage  = matched frames / voiced frames    (how much of the singing the track explains)

Usage: python vocal_ref.py MULTTIPOP_RAW STEM_POOL NAMED_VOCAL_TRACKS_JSON OUT [workers]
STEM_POOL is `amtflow separate --config stems` output for the MulTTiPop segments.
Writes OUT/tracks.csv (every track and its evidence) and OUT/vocal_evidence.json.
"""

from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

SR, HOP = 16000, 160                      # 10 ms frames
FMIN, FMAX = 65.0, 1100.0


def vocal_pitch(path: Path) -> np.ndarray:
    """MIDI pitch per 10 ms frame of the vocals stem; NaN where unvoiced or quiet."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP, frame_length=1024)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0][: len(f0)]
    ok = voiced & (20 * np.log10(rms + 1e-9) > -40)
    return np.where(ok, librosa.hz_to_midi(np.where(ok, f0, 1.0)), np.nan)


def segment(args) -> list[dict]:
    seg, ref, stem, named = args
    try:
        pm = pretty_midi.PrettyMIDI(str(ref))
    except Exception as e:                    # noqa: BLE001
        return [{"segment": seg, "error": type(e).__name__}]
    voc = vocal_pitch(stem)
    voiced = np.isfinite(voc)
    n = len(voc)
    out = []
    for k, ins in enumerate(pm.instruments):
        if ins.is_drum or not ins.notes:
            continue
        note_frames = matched = sung_frames = 0
        hit = np.zeros(n, bool)
        for note in ins.notes:
            a, b = int(note.start * 100), min(n, max(int(note.start * 100) + 1, int(note.end * 100)))
            if a >= n:
                continue
            note_frames += b - a
            v = voc[a:b]
            d = np.abs(((v - note.pitch) + 6) % 12 - 6)     # semitone distance modulo the octave
            m = np.isfinite(v) & (d <= 0.5)
            sung_frames += int(np.isfinite(v).sum())
            matched += int(m.sum())
            hit[a:b] |= m
        out.append({
            "segment": seg, "track_index": k, "track": (ins.name or "").strip(), "program": ins.program,
            "n_notes": len(ins.notes), "median_pitch": float(np.median([x.pitch for x in ins.notes])),
            "named_vocal": (seg, (ins.name or "").strip()) in named,
            "precision": round(matched / max(note_frames, 1), 3),
            "sung_precision": round(matched / max(sung_frames, 1), 3),
            "sung_fraction": round(sung_frames / max(note_frames, 1), 3),
            "coverage": round(int(hit.sum()) / max(int(voiced.sum()), 1), 3),
            "voiced_seconds": round(int(voiced.sum()) / 100, 2),
        })
    return out


def main(raw, pool, changes, out, workers=32):
    raw, pool, out = Path(raw), Path(pool), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    named = {(r["segment"], r["track"].strip()) for r in json.load(open(changes))}
    jobs = []
    for stem_dir in sorted((pool / "stems").iterdir()):
        ref = raw / "site" / "midi" / f"{stem_dir.name}.mid"
        if ref.exists() and (stem_dir / "vocals.flac").exists():
            jobs.append((stem_dir.name, ref, stem_dir / "vocals.flac", named))
    with Pool(int(workers)) as p:
        rows = [r for seg in p.imap_unordered(segment, jobs, chunksize=2) for r in seg]
    rows = [r for r in rows if "error" not in r]
    with open(out / "tracks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["segment"], r["track_index"])))
    print(len(jobs), "segments,", len(rows), "tracks")


if __name__ == "__main__":
    main(*sys.argv[1:])
