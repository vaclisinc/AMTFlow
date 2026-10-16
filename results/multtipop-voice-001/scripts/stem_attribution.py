"""Which separated stem does each MulTTiPop reference track sound in?

For every non-drum reference track, the CQT magnitude of each BS-RoFormer stem (bass,
guitar, piano, vocals, other) is read at the track's note pitches during its notes; the
track's share for a stem is that stem's magnitude over the sum across stems. Loud stems
(bass, vocals) take a large share at every pitch, so each share is compared with that
stem's share at the same pitch over the whole segment (lift = share / baseline). The stem
with the largest lift "explains" the track. Comparing it with the family the
reference labels the track with shows how often the label and the sound disagree.
Separation errors also count, so this is evidence, not ground truth.

usage: python stem_attribution.py MULTTIPOP_RAW STEM_POOL OUT_CSV [workers]
"""

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

STEMS = ("bass", "guitar", "piano", "vocals", "other")
SR, HOP = 22050, 512


def cqt(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    y = librosa.resample(y if y.ndim == 1 else y.mean(1), orig_sr=sr, target_sr=SR)
    return np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(21), n_bins=88))


def segment(args):
    seg, ref, stem_dir = args
    try:
        pm = pretty_midi.PrettyMIDI(str(ref))
        mags = {s: cqt(stem_dir / f"{s}.flac") for s in STEMS}
    except Exception:                           # noqa: BLE001
        return []
    n = min(m.shape[1] for m in mags.values())
    stack = np.stack([mags[s][:, :n] for s in STEMS])            # (stems, 88, frames)
    total = stack.sum(0) + 1e-9
    base = stack.sum(2) / total.sum(1)                            # (stems, 88) segment-wide share
    rows = []
    for k, ins in enumerate(pm.instruments):
        if ins.is_drum or not ins.notes:
            continue
        share, expected, weight = np.zeros(len(STEMS)), np.zeros(len(STEMS)), 0.0
        for note in ins.notes:
            b = note.pitch - 21
            if not 0 <= b < 88:
                continue
            a = int(note.start * SR / HOP)
            e = min(n, max(a + 1, int(note.end * SR / HOP)))
            if a >= n:
                continue
            w = total[b, a:e].sum()
            share += stack[:, b, a:e].sum(1)
            expected += base[:, b] * w
            weight += w
        if weight <= 0:
            continue
        lift = share / np.maximum(expected, 1e-9)
        share /= weight
        rows.append({"segment": seg, "track_index": k, "program": ins.program, "n_notes": len(ins.notes),
                     **{f"share_{s}": round(float(v), 3) for s, v in zip(STEMS, share)},
                     **{f"lift_{s}": round(float(v), 3) for s, v in zip(STEMS, lift)},
                     "best_stem": STEMS[int(lift.argmax())]})
    return rows


def main(raw, pool, out_csv, workers=48):
    raw, pool = Path(raw), Path(pool)
    jobs = [(d.name, raw / "site" / "midi" / f"{d.name}.mid", d) for d in sorted((pool / "stems").iterdir())
            if (raw / "site" / "midi" / f"{d.name}.mid").exists()]
    with Pool(int(workers)) as p:
        rows = [r for seg in p.imap_unordered(segment, jobs, chunksize=2) for r in seg]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(jobs), "segments,", len(rows), "tracks")


if __name__ == "__main__":
    main(*sys.argv[1:])
