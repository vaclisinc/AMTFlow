"""MulTTiPop (Pruyne et al. 2026) multitrack transcription benchmark.

Data: the `--root` directory holds the HF release (`hf/`, split lists and
metadata), the project site's files (`site/midi/<id>.mid` ground truth,
`<id>_{mt3,yourmt3,muscriptor}.mid` published outputs; `site/gt-synth-audio/` is the
site's SYNTHESIZED rendering of the MIDI and must not be used as input), and the
recordings downloaded from YouTube per the metadata (`youtube/<id>.<ext>`).

Metric (paper, Table 4): note onset F1 with 50 ms onset tolerance (mir_eval, pitch
within 50 cents, offsets ignored), computed per segment and averaged over segments,
with notes matched only within the same instrument class. Instrument spaces:

    exact     GM program (drums one class, matched by drum pitch)
    hp        harmonic vs percussive (the paper's "harmonic-percussive")
    group     MuScriptor instrument group (MT3_FULL_PLUS)
    family    AMTFlow families (piano, guitar, bass, strings, winds, synthesizer,
              voice, organ, chromatic, drums)
    agnostic  pitched notes only, instrument ignored
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import mir_eval
import numpy as np

SPACES = ("exact", "hp", "group", "family", "agnostic")
DRUMS = -1

FAMILY_OF_GROUP = {
    "acoustic_piano": "piano", "electric_piano": "piano",
    "chromatic_percussion": "chromatic", "organ": "organ",
    "acoustic_guitar": "guitar", "clean_electric_guitar": "guitar",
    "distorted_electric_guitar": "guitar",
    "acoustic_bass": "bass", "electric_bass": "bass",
    **{g: "strings" for g in ("violin", "viola", "cello", "contrabass", "orchestral_harp",
                              "timpani", "string_ensemble", "synth_strings", "orchestra_hit")},
    "voice": "voice",
    **{g: "winds" for g in ("trumpet", "trombone", "tuba", "french_horn", "brass_section",
                            "soprano_and_alto_sax", "tenor_sax", "baritone_sax", "oboe",
                            "english_horn", "bassoon", "clarinet", "flutes")},
    "synth_lead": "synthesizer", "synth_pad": "synthesizer",
}


@lru_cache(maxsize=1)
def _groups() -> tuple[dict[int, str], dict[str, int]]:
    """GM program -> MuScriptor group, and group -> its representative program."""
    from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES, MT3Tokenizer
    gm = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001).group_program_map
    of_program, representative = {}, {}
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if name == "drums" or not gm.get(gid):
            continue
        representative[name] = gm[gid][0]
        for p in gm[gid]:
            of_program[p] = name
    return of_program, representative


def instrument_class(program: int, space: str) -> str | None:
    """Class of a note's program in `space`; None drops the note from that space."""
    if space == "exact":
        return "drums" if program == DRUMS else f"p{program}"
    if space == "hp":
        return "drums" if program == DRUMS else "pitched"
    if space == "agnostic":
        return None if program == DRUMS else "pitched"
    if program == DRUMS:
        return "drums"
    group = _groups()[0].get(program, f"program_{program}")
    return group if space == "group" else FAMILY_OF_GROUP.get(group, "other")


# YourMT3+ writes its singing-voice class on GM program 65 (alto sax) under this track
# name; read it as voice (Choir Aahs, 52) so it is not scored as a wind instrument.
VOICE_TRACK_NAMES = {"Singing Voice", "Singing Voice (chorus)"}


def midi_notes(path: str | Path) -> list[tuple[float, float, int, int]]:
    """(onset, offset, pitch, program or DRUMS) for every note of a MIDI file."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    return [(n.start, n.end, n.pitch,
             DRUMS if ins.is_drum else 52 if ins.name in VOICE_TRACK_NAMES else ins.program)
            for ins in pm.instruments for n in ins.notes]


def amtflow_notes(record: dict) -> list[tuple[float, float, int, int]]:
    """Notes of an `amtflow run` JSON, each MuScriptor group at its first GM program."""
    rep = _groups()[1]
    out = []
    for n in record["notes"]:
        if n["instrument"] == "drums":
            prog = DRUMS
        elif n["instrument"] in rep:
            prog = rep[n["instrument"]]
        elif n["instrument"].startswith("program_"):
            prog = int(n["instrument"].split("_")[1])
        else:
            prog = 0            # instrument-less sources (e.g. pYIN) count as piano
        out.append((n["start"], n["end"], n["pitch"], prog))
    return out


def score_segment(ref, est, space: str) -> dict:
    """Onset precision/recall/F1 with matching restricted to same-class notes."""
    def split(notes):
        by = {}
        for on, off, pitch, prog in notes:
            c = instrument_class(prog, space)
            if c is not None:
                by.setdefault(c, []).append((on, max(off, on + 1e-3), pitch))
        return by

    R, E = split(ref), split(est)
    tp = 0
    for c in set(R) & set(E):
        r, e = np.array(R[c]), np.array(E[c])
        tp += len(mir_eval.transcription.match_notes(
            r[:, :2], mir_eval.util.midi_to_hz(r[:, 2]), e[:, :2], mir_eval.util.midi_to_hz(e[:, 2]),
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None))
    n_ref, n_est = sum(map(len, R.values())), sum(map(len, E.values()))
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    return {"tp": tp, "n_ref": n_ref, "n_est": n_est, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def split_ids(root: Path, split: str) -> list[str]:
    """`dev`, `test`, `all`, or `<split>_rich`: segments whose reference has at least two
    MuScriptor groups (>= 10 notes each) outside piano, guitar, bass and voice, i.e.
    where `other`-like instruments (strings, winds, synths, organ, ...) matter."""
    if split == "all":
        return split_ids(root, "dev") + split_ids(root, "test")
    if split.endswith("_rich"):
        return [i for i in split_ids(root, split.removesuffix("_rich")) if is_rich(root, i)]
    return [r["id"] for r in json.load(open(root / "hf" / f"{split}.json"))]


def is_rich(root: Path, seg: str, min_notes: int = 10) -> bool:
    from collections import Counter
    of = _groups()[0]
    counts = Counter(of.get(p, f"program_{p}") for *_, p in midi_notes(root / "site" / "midi" / f"{seg}.mid")
                     if p != DRUMS)
    core = {"piano", "guitar", "bass", "voice"}
    return sum(1 for g, n in counts.items()
               if n >= min_notes and FAMILY_OF_GROUP.get(g, "other") not in core) >= 2


# Seconds to add to a video's metadata [start, end]: the YouTube upload we can download
# (2026-09) is offset from the audio the authors aligned the reference MIDI to. Measured
# by transcribing our cut with unconditioned MuScriptor medium and sliding it against the
# site's published MuScriptor transcription of the authors' audio (centre of the >= 95%
# agreement plateau, 10 ms grid; consistent across a video's segments). Kept when
# |offset| >= 0.05 s and agreement rose by >= 0.2 F1; clean segments sit within ±0.02 s.
# Beat phase against the metadata's `audio_beats` (beat_this on our audio) agrees for
# Y4O7ufx9D_s, fUYaosyR4bE, mpdcKmaHk_s, Urw-iutHw5E, c0A7jAVDPJU, tWjNFC-FinU.
# An m4a/webm format difference is ruled out (they differ by 0.036 s).
YOUTUBE_OFFSETS = {
    "fUYaosyR4bE": +0.97, "Y4O7ufx9D_s": +0.76, "mpdcKmaHk_s": +0.26, "Urw-iutHw5E": +0.18,
    "c0A7jAVDPJU": +0.15, "1TO48Cnl66w": +0.06, "dQw4w9WgXcQ": -0.07, "tWjNFC-FinU": -0.54,
    "p1a_4CN4onA": -0.56,
    "_CsJCa2TlXU": -0.49,   # segments at 1 s and 45 s measure -0.455 / -0.515: slight speed drift
}


AUDIO_EXTS = (".webm", ".m4a", ".opus", ".mp3", ".mp4", ".wav", ".flac", ".ogg")


def cut_recordings(root: Path, full_dir: Path, out_dir: Path | None = None,
                   overwrite: bool = False, offsets: bool = True) -> dict:
    """Cut every segment out of its full YouTube audio (`<full_dir>/<ytid>.<ext>`) into
    `<out_dir>/<id>.flac` (default `root/youtube`), decoding first so the cut is
    sample-accurate at the metadata's [start, end] (yt-dlp's --download-sections cuts at
    packet/keyframe boundaries and started up to ~10 s early). Native sample rate and
    channels kept. A negative start means the authors' segment begins before the video:
    it is padded with that much leading silence, so the length is the metadata's
    `audio_length`. `offsets` applies `YOUTUBE_OFFSETS`."""
    import subprocess
    out_dir = out_dir or root / "youtube"
    out_dir.mkdir(parents=True, exist_ok=True)
    done, missing, adjusted = 0, [], {}
    for meta in sorted((root / "hf").glob("*/*/meta.json")):
        m = json.load(open(meta))
        src = [p for p in full_dir.glob(f"{m['youtube']['ytid']}.*")
               if p.suffix in AUDIO_EXTS and p.stem == m["youtube"]["ytid"]]   # ids may start with "_"
        target = out_dir / f"{m['id']}.flac"
        if not src:
            missing.append(m["id"])
            continue
        y = m["youtube"]
        shift = YOUTUBE_OFFSETS.get(y["ytid"], 0.0) if offsets else 0.0
        start, end = y["start"] + shift, y["end"] + shift
        if shift or start < 0:
            adjusted[m["id"]] = {"offset": shift, "lead_silence": max(0.0, -start)}
        if target.exists() and not overwrite:
            done += 1
            continue
        if start < 0:
            sr = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                     "stream=sample_rate", "-of", "csv=p=0", str(src[0])],
                                    capture_output=True, text=True, check=True).stdout.strip())
            cut = ["-af", f"atrim=end={end:.6f},asetpts=PTS-STARTPTS,"
                          f"adelay=delays={round(-start * sr)}S:all=1"]
        else:
            cut = ["-ss", f"{start:.6f}", "-to", f"{end:.6f}"]
        tmp = target.with_suffix(".tmp.flac")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src[0]), *cut, "-c:a", "flac", str(tmp)],
                       check=True)
        tmp.replace(target)
        done += 1
    return {"segments": done, "missing": missing, "adjusted": adjusted}


def manifest(root: Path, split: str, out: Path, audio_dir: Path | None = None) -> int:
    ids = split_ids(root, split)
    audio_dir = audio_dir or root / "youtube"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w") as f:
        for i in ids:
            audio = sorted(audio_dir.glob(f"{i}.*"))
            if not audio:
                continue                      # recording unavailable
            f.write(json.dumps({"id": i, "audio": str(audio[0]),
                                "reference": str(root / "site" / "midi" / f"{i}.mid")}) + "\n")
            written += 1
    return written


def keep_families(notes, families: set[str] | None):
    """Only notes whose AMTFlow family (via the MuScriptor group of their program) is in
    `families`; None keeps everything."""
    if families is None:
        return notes
    return [n for n in notes if instrument_class(n[3], "family") in families]


def evaluate(root: Path, split: str, estimates, out_csv: Path | None = None,
             reference_dir: Path | None = None, families: set[str] | None = None) -> dict:
    """`estimates`: a published model name (mt3 | yourmt3 | muscriptor) or an `amtflow
    run` output directory. Returns per-space means over segments (the paper's
    averaging) and micro totals."""
    rows = []
    est_dir = None if isinstance(estimates, str) and not Path(estimates).exists() else Path(estimates)
    for i in split_ids(root, split):
        if est_dir is not None and not (est_dir / "clips" / f"{i}.json").exists():
            continue                      # no recording for this segment
        ref = keep_families(midi_notes((Path(reference_dir) if reference_dir else root / "site" / "midi")
                                       / f"{i}.mid"), families)
        if families is not None and not ref:
            continue                      # none of the scored families in this segment
        if isinstance(estimates, str) and not Path(estimates).exists():
            est = midi_notes(root / "site" / "midi" / f"{i}_{estimates}.mid")
        else:
            est = amtflow_notes(json.load(open(Path(estimates) / "clips" / f"{i}.json")))
        est = keep_families(est, families)
        row = {"id": i}
        for space in SPACES:
            m = score_segment(ref, est, space)
            row.update({f"{space}_{k}": v for k, v in m.items()})
        rows.append(row)
    summary = {"split": split, "segments": len(rows)}
    for space in SPACES:
        tp = sum(r[f"{space}_tp"] for r in rows)
        ne = sum(r[f"{space}_n_est"] for r in rows)
        nr = sum(r[f"{space}_n_ref"] for r in rows)
        summary[space] = {
            "precision": float(np.mean([r[f"{space}_precision"] for r in rows])),
            "recall": float(np.mean([r[f"{space}_recall"] for r in rows])),
            "f1": float(np.mean([r[f"{space}_f1"] for r in rows])),
            "micro_f1": 2 * tp / (ne + nr) if ne + nr else 0.0}
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return summary
