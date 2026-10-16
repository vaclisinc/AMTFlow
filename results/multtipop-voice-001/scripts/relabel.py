"""Write MulTTiPop references with the sung melody relabelled as voice (GM program 52, the representative program of MuScriptor's voice group).

A track becomes voice when
  (A) it is named as vocals and sung_precision >= 0.3, or
  (B) it is not, but sung_precision >= 0.6, coverage >= 0.3, at least 30% of its notes
      fall where the vocals stem is voiced, and its GM class is one arrangers use for a
      sung line (pipe, reed, brass, synth lead).
Tracks already on a voice program (52-54) and drum tracks are unchanged; so are notes and
timing. Piano, guitar, bass, organ, strings and pads are never relabelled: there a match
is more likely accompaniment that shares the melody's pitches.

Usage: python relabel.py MULTTIPOP_RAW TRACKS_CSV OUT_DIR
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import pretty_midi

MELODY_CLASSES = {"pipe": range(72, 80), "reed": range(64, 72), "brass": range(56, 64),
                  "synth lead": range(80, 88)}


def gm_class(program: int) -> str | None:
    return next((k for k, r in MELODY_CLASSES.items() if program in r), None)


def decide(r: dict) -> str | None:
    p, cov, sung = float(r["sung_precision"]), float(r["coverage"]), float(r["sung_fraction"])
    prog = int(r["program"])
    if 52 <= prog <= 54:
        return None
    if r["named_vocal"] == "True" and p >= 0.3:
        return "A+B"
    if r["named_vocal"] != "True" and p >= 0.6 and cov >= 0.3 and sung >= 0.3 and gm_class(prog):
        return "B"
    return None


def main(raw, tracks_csv, out):
    raw, out = Path(raw), Path(out)
    (out / "midi").mkdir(parents=True, exist_ok=True)
    picks: dict[str, dict[int, dict]] = {}
    for r in csv.DictReader(open(tracks_csv)):
        rule = decide(r)
        if rule:
            picks.setdefault(r["segment"], {})[int(r["track_index"])] = {**r, "rule": rule}
    changes = []
    for ref in sorted((raw / "site" / "midi").glob("*.mid")):
        seg = ref.stem
        if "_" in seg[11:]:                      # published model outputs (<id>_mt3.mid, ...)
            continue
        target = out / "midi" / ref.name
        if seg not in picks:
            shutil.copyfile(ref, target)
            continue
        pm = pretty_midi.PrettyMIDI(str(ref))
        for k, r in picks[seg].items():
            ins = pm.instruments[k]
            changes.append({"segment": seg, "track_index": k, "track": ins.name, "rule": r["rule"],
                            "from_program": int(ins.program), "to_program": 52, "notes": len(ins.notes),
                            "sung_precision": float(r["sung_precision"]), "coverage": float(r["coverage"])})
            ins.program = 52
        pm.write(str(target))
    summary = {"segments_changed": len({c["segment"] for c in changes}), "tracks_changed": len(changes),
               "by_rule": {k: sum(c["rule"] == k for c in changes) for k in ("A+B", "B")},
               "notes_relabelled": sum(c["notes"] for c in changes)}
    json.dump({"rule": __doc__.split("Usage:")[0].strip(), **summary, "changes": changes},
              open(out / "changes.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:])
