"""Note lists -> piano-roll tensors and MIDI files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

LOWEST_PIANO_PITCH = 21     # A0; 88 keys up to C8 (108)


def cube(notes: list[dict], families: list[str], duration: float, fps: float) -> np.ndarray:
    """(F, T, 88) uint8 activity; T = ceil(duration * fps). Drums and out-of-range pitches are
    kept only when they fall inside the 88-key range, like any other note."""
    frames = max(1, math.ceil(duration * fps))
    out = np.zeros((len(families), frames, 88), dtype=np.uint8)
    index = {f: i for i, f in enumerate(families)}
    for n in notes:
        f, key = index.get(n.get("family")), n["pitch"] - LOWEST_PIANO_PITCH
        if f is None or not 0 <= key < 88:
            continue
        a = min(frames - 1, int(n["start"] * fps))
        b = max(a + 1, min(frames, math.ceil(n["end"] * fps)))
        out[f, a:b, key] = 1
    return out


def merged_roll(c: np.ndarray) -> np.ndarray:
    """(T, 88): the cube with its family channels merged."""
    return c.max(axis=0)


def to_midi(notes: list[dict], path: str | Path) -> None:
    """One MIDI track per (stem, instrument), at each MuScriptor group's representative GM
    program; drums on the drum channel. Velocity is constant (MuScriptor emits none)."""
    import pretty_midi

    from .multtipop import _groups

    program = _groups()[1]
    pm = pretty_midi.PrettyMIDI()
    tracks: dict[tuple[str, str], pretty_midi.Instrument] = {}
    for n in notes:
        key = (n.get("stem", "mix"), n["instrument"])
        if key not in tracks:
            drums = n["instrument"] == "drums"
            label = n["instrument"].replace("_", " ").title()        # "Acoustic Piano"
            name = label if key[0] == "mix" else f"{key[0]}: {label}"
            tracks[key] = pretty_midi.Instrument(program=int(program.get(n["instrument"], 0)),
                                                 is_drum=drums, name=name)
        end = max(n["end"], n["start"] + 0.01)
        tracks[key].notes.append(pretty_midi.Note(velocity=80, pitch=int(n["pitch"]),
                                                  start=n["start"], end=end))
    pm.instruments.extend(tracks.values())
    pm.write(str(path))
