"""Score MulTTiPop arms against the official and the vocal-corrected references.

usage: python score_refs.py SPLIT OUT_DIR
Paths come from $MULTTIPOP_RAW, $MULTTIPOP_DERIVED and $AMTFLOW_RUNS (run directories
named multtipop-yt2-<split>-<arm>).
Writes OUT_DIR/<split>_summary.csv and <split>_bootstrap.csv (10k paired resamples
over segments, seed 0). Arms without outputs are skipped.
"""

import csv
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from amtflow.multtipop import evaluate

RAW = Path(os.environ["MULTTIPOP_RAW"])
DER = Path(os.environ["MULTTIPOP_DERIVED"])
OUT_RUNS = Path(os.environ["AMTFLOW_RUNS"])
REFS = {"official": None, "vocal_corrected": DER / "reference-vocal-ab-v1/midi"}   # relabel.py output
ARMS = {
    "pub:mt3": "mt3",
    "pub:yourmt3": "yourmt3",
    "pub:muscriptor": "muscriptor",
    "R1 no tags": "multtipop-yt2-{s}-r1-mix-none",
    "R4 default (7 families)": "multtipop-yt2-{s}-r4-mix-paper-topn",
    "R9 default + voice": "multtipop-yt2-{s}-r9-mix-default-voice",
}
PAIRS = [("R4 default (7 families)", "pub:muscriptor"), ("R9 default + voice", "pub:muscriptor"),
         ("R9 default + voice", "R4 default (7 families)")]
SPACES = ("exact", "group", "family", "agnostic")


def score(job):
    split, arm, ref = job
    est = ARMS[arm] if arm.startswith("pub:") else str(OUT_RUNS / ARMS[arm].format(s=split))
    tmp = Path(tempfile.mkdtemp()) / "s.csv"
    evaluate(RAW, split, est, tmp, REFS[ref])
    return arm, ref, {r["id"]: r for r in csv.DictReader(open(tmp))}


def main(split, out):
    out = Path(out)
    arms = [a for a in ARMS if a.startswith("pub:") or (OUT_RUNS / ARMS[a].format(s=split) / "clips").exists()]
    with Pool(12) as p:
        res = p.map(score, [(split, a, r) for a in arms for r in REFS])
    scores = {(a, r): rows for a, r, rows in res}
    ids = sorted(set.intersection(*[set(v) for (a, r), v in scores.items() if not a.startswith("pub:")]))
    summary = []
    for (a, r), rows in scores.items():
        summary.append({"split": split, "reference": r, "arm": a, "segments": len(ids),
                        **{s: round(100 * np.mean([float(rows[i][f"{s}_f1"]) for i in ids]), 2) for s in SPACES}})
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(10000, len(ids)))
    boot = []
    for r in REFS:
        for a, b in PAIRS:
            if (a, r) not in scores or (b, r) not in scores:
                continue
            for s in ("exact", "family", "agnostic"):
                d = 100 * np.array([float(scores[a, r][i][f"{s}_f1"]) - float(scores[b, r][i][f"{s}_f1"]) for i in ids])
                bs = d[idx].mean(1)
                boot.append({"split": split, "reference": r, "arm": a, "baseline": b, "space": s,
                             "mean_diff": round(d.mean(), 2), "ci95_low": round(np.percentile(bs, 2.5), 2),
                             "ci95_high": round(np.percentile(bs, 97.5), 2)})
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in ((f"{split}_summary.csv", summary), (f"{split}_bootstrap.csv", boot)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for row in sorted(summary, key=lambda x: (x["reference"], x["arm"])):
        print(row["reference"].ljust(16), row["arm"].ljust(26), *(f"{row[s]:6.2f}" for s in SPACES))
    for b in boot:
        print(b["reference"].ljust(16), f'{b["arm"]} - {b["baseline"]}'.ljust(52), b["space"].ljust(9),
              f'{b["mean_diff"]:+.2f} [{b["ci95_low"]:+.2f}, {b["ci95_high"]:+.2f}]')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
