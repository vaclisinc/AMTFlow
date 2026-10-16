"""Summary + paired bootstrap over the MulTTiPop arms.

usage: uv run python summarize.py MULTTIPOP_ROOT RUNS_DIR OUT_DIR
RUNS_DIR holds one `amtflow run` output per arm and split, named `<split>-<arm>`
(e.g. `test-r4-mix-paper-topn`). Writes OUT_DIR/summary.csv and
OUT_DIR/paired_bootstrap.csv. Published model outputs
(site MIDIs, transcribed by the authors from their own audio) are scored on the same
segments as the arms.
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

from amtflow.multtipop import SPACES, evaluate, split_ids

ARMS = {
    "r1-mix-none": "mixture, no instrument constraint",
    "r2-mix-paper": "mixture, paper tagger",
    "r3-mix-oracle": "mixture, oracle families",
    "r4-mix-paper-topn": "mixture, paper tagger + top-N routing",
    "r4b-mix-paper-topn-batched": "R4 with batched decoding (batch 6, no prelude)",
    "r5-stems-paper": "stems, paper tagger on other",
    "r6-stems-oracle": "stems, oracle on other",
    "r7-stems-other-topn": "stems, top-N on other",
    "r8-stems-routed-top1": "stems, all stems + other top-1",
}
PUBLISHED = ("muscriptor", "mt3", "yourmt3")
PAIRS = [("r2-mix-paper", "r1-mix-none"), ("r3-mix-oracle", "r1-mix-none"),
         ("r4-mix-paper-topn", "r1-mix-none"), ("r4-mix-paper-topn", "r2-mix-paper"),
         ("r5-stems-paper", "r2-mix-paper"), ("r5-stems-paper", "r4-mix-paper-topn"),
         ("r6-stems-oracle", "r3-mix-oracle"), ("r7-stems-other-topn", "r5-stems-paper"),
         ("r8-stems-routed-top1", "r5-stems-paper"), ("r7-stems-other-topn", "r4-mix-paper-topn"),
         ("r4b-mix-paper-topn-batched", "r4-mix-paper-topn"),
         ("r1-mix-none", "pub:muscriptor"), ("r4-mix-paper-topn", "pub:muscriptor")]


def per_segment(root, split, est, ids):
    tmp = Path(tempfile.mkdtemp()) / "scores.csv"
    evaluate(root, split, est, tmp)
    rows = {r["id"]: r for r in csv.DictReader(open(tmp))}
    return {i: rows[i] for i in ids}


def main(root: Path, runs: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    summary, boot = [], []
    rng = np.random.default_rng(0)
    for split in ("dev", "dev_rich", "test", "test_rich"):
        base = split.split("_")[0]
        arms = {a: runs / f"{base}-{a}" for a in ARMS if (runs / f"{base}-{a}").exists()}
        ids = sorted({p.stem for p in next(iter(arms.values())).joinpath("clips").glob("*.json")}
                     & set(split_ids(root, split)))
        scores = {a: per_segment(root, split, str(d), ids) for a, d in arms.items()}
        scores |= {f"pub:{p}": per_segment(root, split, p, ids) for p in PUBLISHED}
        for a, rows in scores.items():
            m = lambda k: round(100 * np.mean([float(r[k]) for r in rows.values()]), 2)
            summary.append({"arm": a, "split": split, "segments": len(rows),
                            "description": ARMS.get(a, "published output (authors' audio)"),
                            **{s: m(f"{s}_f1") for s in SPACES},
                            "family_P": m("family_precision"), "agnostic_P": m("agnostic_precision"),
                            "family_R": m("family_recall"), "agnostic_R": m("agnostic_recall")})
        idx = rng.integers(0, len(ids), size=(10000, len(ids)))
        for a, b in PAIRS:
            if a not in scores or b not in scores:
                continue
            for s in ("exact", "group", "family", "agnostic"):
                d = 100 * np.array([float(scores[a][i][f"{s}_f1"]) - float(scores[b][i][f"{s}_f1"]) for i in ids])
                bs = d[idx].mean(1)
                boot.append({"split": split, "arm": a, "baseline": b, "space": s, "segments": len(ids),
                             "mean_diff": round(d.mean(), 2), "ci95_low": round(np.percentile(bs, 2.5), 2),
                             "ci95_high": round(np.percentile(bs, 97.5), 2)})
    for name, rows in (("summary.csv", summary), ("paired_bootstrap.csv", boot)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
