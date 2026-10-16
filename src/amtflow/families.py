"""Instrument families: tagger labels -> families -> AMT-model instrument constraints."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def normalize_label(label: str) -> str:
    label = label.strip().lower().removeprefix("instrument---")
    return re.sub(r"[^a-z0-9]+", "", label)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_dir() -> Path:
    """The built-in configs: `configs/` of a source checkout, or `amtflow/configs` in a wheel."""
    here = Path(__file__).resolve().parent
    return here / "configs" if (here / "configs").is_dir() else here.parents[1] / "configs"


def config_path(name: str | Path) -> Path:
    """A YAML path, or the built-in config of that name (`default`, `paper.yaml`, ...)."""
    path = Path(name)
    if path.exists():
        return path
    builtin = config_dir() / (path.name if path.suffix in (".yaml", ".yml") else f"{name}.yaml")
    if not builtin.exists():
        known = sorted(p.stem for p in config_dir().glob("*.yaml"))
        raise FileNotFoundError(f"no config {name!r}: not a file and not a built-in config {known}")
    return builtin


def load_config(path: str | Path) -> dict:
    """YAML with ${ENV} expansion; `extends: other.yaml` deep-merges onto that file
    (mappings merge, every other value replaces). `path` and `extends` may name a built-in
    config; `extends` is otherwise relative to the file."""
    path = config_path(path)
    text = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), path.read_text())
    cfg = yaml.safe_load(text)
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg
    local = path.parent / parent
    return _merge(load_config(local if local.exists() else parent), cfg)


class Taxonomy:
    """Families, their default instruments, and fine routing inside a family.

    `families`: family -> default MuScriptor instruments (the paper: one per family).
    `family_groups`: family -> every MuScriptor group fine routing may choose from.
    `label_groups`: tagger label -> the specific groups it names (generic labels such as
    "guitar" name none and only vote for the family).
    `routing`: mode `family` (defaults), `top1` (best-scoring group of each detected
    family), or `topn` (groups within `ratio` of the family's best score and above
    `min_score`, at most `max_per_family`). A family without group evidence falls back
    to its defaults (`fallback: default`), or with `fallback: all` to all its groups when
    it has at most `fallback_all_max` of them (MuScriptor then chooses among a few, e.g.
    synth lead vs pad; wide sets such as every wind instrument degrade it) and to its
    defaults otherwise.
    """

    def __init__(self, config: dict):
        self.families: dict[str, list[str]] = {f: list(v) for f, v in config["families"].items()}
        self.label_families = {normalize_label(k): v for k, v in config["label_families"].items()}
        self.fallback = list(config.get("fallback_families", []))
        self.family_groups: dict[str, list[str]] = {
            f: list(config.get("family_groups", {}).get(f, ins)) for f, ins in self.families.items()}
        self.label_groups = {normalize_label(k): list(v)
                             for k, v in config.get("label_groups", {}).items()}
        self.routing = {"mode": "family", "ratio": 0.5, "min_score": 0.0, "max_per_family": 3,
                        "fallback": "default", "fallback_all_max": 3, **config.get("routing", {})}
        self.family_of_instrument = {i: f for f, ins in self.families.items() for i in ins}
        for f, groups in self.family_groups.items():
            for g in groups:
                self.family_of_instrument.setdefault(g, f)

    def groups_of_label(self, label: str) -> list[str]:
        return self.label_groups.get(normalize_label(label), [])

    def route(self, families, group_scores: dict[str, float] | None,
              routing: dict | None = None) -> list[str]:
        """MuScriptor instruments for detected `families` under the routing mode."""
        r = {**self.routing, **(routing or {})}
        if r["mode"] == "family":
            return self.instruments(families)
        scores = group_scores or {}
        out = []
        for f in self.order(families):
            cands = sorted(((scores[g], g) for g in self.family_groups[f]
                            if scores.get(g, 0.0) > r["min_score"]), reverse=True)
            if not cands:
                small = len(self.family_groups[f]) <= int(r["fallback_all_max"])
                out += self.family_groups[f] if r["fallback"] == "all" and small else self.families[f]
                continue
            top = cands[0][0]
            if r["mode"] == "top1":
                keep = [g for s, g in cands if s >= top]          # ties stay together
            else:
                keep = [g for s, g in cands if s >= r["ratio"] * top][: int(r["max_per_family"])]
            out += keep
        return out

    def order(self, families) -> list[str]:
        wanted = set(families)
        return [f for f in self.families if f in wanted]

    def family_of_label(self, label: str) -> str | None:
        """Routing family of a tagger label; None for known non-routing labels."""
        key = normalize_label(label)
        if key not in self.label_families:
            raise KeyError(f"label {label!r} has no family mapping")
        return self.label_families[key]

    def instruments(self, families) -> list[str]:
        return [i for f in self.order(families) for i in self.families[f]]
