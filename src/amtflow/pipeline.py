"""The AMTFlow pipeline: separator -> per-stem instrument constraint -> AMT model.

Per stem the constraint comes from a rule in `stems:` (fixed families, explicit
instruments, `skip`) or, by default, from the tagger. With the `none` separator the
single stem is `mix`, which is the paper's method.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .audio import load_mono
from .families import Taxonomy
from .separators import build_separator
from .taggers import build_tagger
from .transcribers import build_transcriber


class Pipeline:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.taxonomy = Taxonomy(cfg)
        self.separator = build_separator(cfg.get("separator", {"name": "none"}), device)
        self.stem_rules: dict[str, dict] = cfg.get("stems", {})
        self.min_coverage = float(cfg.get("min_stem_coverage", 0.0))
        self.device = device
        self._taggers: dict = {}             # built on first use: stem rules may make them unused
        self._row: dict | None = None
        self._windows = None
        self.transcriber = build_transcriber(cfg["transcriber"], device)
        self.input_sr = cfg.get("input_sample_rate", 44100)
        # `stem_context`: mix rhythmic context into each pitched stem before transcription
        # (tagging still sees the clean stem): `source: drums` (the drums stem) or `click`
        # (noise clicks on beats tracked on the mixture), at `gain_db` relative to the
        # context's own level; drums are removed from such a stem's constraint.
        self.context = cfg.get("stem_context") or {}

    def _tag(self, stem, which: str):
        """Run the main tagger (`tagger`) or the routing source (`routing.source`)."""
        if which not in self._taggers:
            cfg = self.cfg["tagger"] if which == "tagger" else self.taxonomy.routing["source"]
            self._taggers[which] = build_tagger(cfg, self.taxonomy, self.device)
        return self._taggers[which](stem.wav, stem.sr, row=self._row, stem=stem.name)

    def _group_evidence(self, stem, tag) -> tuple[dict[str, float], dict | None]:
        """Group scores for fine routing: the routing source's, else the tagger's."""
        if self.taxonomy.routing.get("source"):
            src = self._tag(stem, "source")
            return src.groups, src.to_json()
        return (tag.groups if tag else {}), None

    def _constraint(self, stem) -> tuple[list[str], list[str], dict | None]:
        self._windows = None                 # per-window instruments, when the tagger gives windows
        rule = self.stem_rules.get(stem.name, {})
        if "families" in rule and rule.get("route"):
            # the stem's family is fixed; which instruments inside it is routed
            fams = self.taxonomy.order(rule["families"])
            tag = None if self.taxonomy.routing.get("source") else self._tag(stem, "tagger")
            groups, source = self._group_evidence(stem, tag)
            routing = {"mode": "top1", **{k: v for k, v in self.taxonomy.routing.items()
                                          if k != "mode" or v != "family"}, **rule.get("routing", {})}
            info = {"routed": True, **({"tagger": tag.to_json()} if tag else {}),
                    **({"routing_source": source} if source else {})}
            return fams, self.taxonomy.route(fams, groups, routing), info
        if "instruments" in rule:
            return list(rule.get("families", [])), list(rule["instruments"]), None
        if "families" in rule:
            fams = self.taxonomy.order(rule["families"])
            return fams, self.taxonomy.instruments(fams), None
        tag = self._tag(stem, "tagger")
        if self.taxonomy.routing["mode"] == "family":
            if tag.windows is not None:
                self._windows = (tag.window_seconds, [self.taxonomy.instruments(w) for w in tag.windows])
            return tag.families, self.taxonomy.instruments(tag.families), tag.to_json()
        groups, source = self._group_evidence(stem, tag)
        info = {**tag.to_json(), **({"routing_source": source} if source else {})}
        if tag.windows is not None:
            self._windows = (tag.window_seconds, [self.taxonomy.route(w, groups) for w in tag.windows])
        return tag.families, self.taxonomy.route(tag.families, groups), info

    def tag_clip(self, wav: np.ndarray, sr: int, row: dict | None = None) -> dict:
        """Separation and instrument constraints only (the `tag` stage); a later
        transcription stage reads them back through the `precomputed` tagger."""
        self._row = row
        out = []
        for stem in self.separator(wav, sr, row):
            rule = self.stem_rules.get(stem.name, {})
            rec = {"stem": stem.name, "coverage": round(stem.coverage, 2)}
            if stem.missing or rule.get("skip") or stem.coverage <= 0 or stem.coverage < self.min_coverage:
                out.append({**rec, "skipped": "missing" if stem.missing else
                            "rule" if rule.get("skip") else "silent"})
                continue
            fams, instruments, tag = self._constraint(stem)
            out.append({**rec, "families": fams, "instruments": instruments,
                        **({"tagger": tag} if tag and "routed" not in tag else {}),
                        **({k: v for k, v in tag.items() if k != "routed"}
                           if tag and "routed" in tag else {})})
        return {"stems": out}

    def __call__(self, wav: np.ndarray, sr: int, row: dict | None = None) -> dict:
        timing: dict[str, float] = {}
        t = time.perf_counter()
        self._row = row
        stems = self.separator(wav, sr, row)
        timing["separate"] = time.perf_counter() - t
        timing["tag"] = timing["transcribe"] = 0.0
        notes, stem_records = [], []
        for stem in stems:
            rule = self.stem_rules.get(stem.name, {})
            rec = {"stem": stem.name, "coverage": round(stem.coverage, 2)}
            if stem.missing:
                stem_records.append({**rec, "skipped": "missing"})
                continue
            if rule.get("skip") or stem.coverage <= 0 or stem.coverage < self.min_coverage:
                stem_records.append({**rec, "skipped": "rule" if rule.get("skip") else "silent"})
                continue
            t = time.perf_counter()
            fams, instruments, tag = self._constraint(stem)
            timing["tag"] += time.perf_counter() - t
            if not instruments and self.cfg.get("skip_untagged"):
                # an empty constraint would otherwise mean "unconstrained"
                stem_records.append({**rec, "skipped": "untagged",
                                     **({"tagger": tag} if tag and "routed" not in tag else {})})
                continue
            t = time.perf_counter()
            stem_wav = stem.wav
            if self.context and stem.name in self.context.get("stems", ()):
                stem_wav = stem.wav + self._context_wav(stems, wav, sr, stem)
                instruments = [i for i in instruments if i != "drums"]
            extra = {}
            if self._windows is not None:
                extra["window_instruments"] = self._windows
            stem_notes = self.transcriber(stem_wav, stem.sr, instruments, stem=stem.name, **extra)
            timing["transcribe"] += time.perf_counter() - t
            fixed_family = fams[0] if "instruments" in rule and len(fams) == 1 else None
            for n in stem_notes:
                notes.append({**n.to_json(), "stem": stem.name,
                              "family": fixed_family or self.taxonomy.family_of_instrument.get(n.instrument)})
            stem_records.append({**rec, "families": fams, "instruments": instruments,
                                 "n_notes": len(stem_notes),
                                 **({"tagger": tag} if tag and "routed" not in tag else {}),
                                 **({k: v for k, v in tag.items() if k != "routed"}
                                    if tag and "routed" in tag else {})})
        notes.sort(key=lambda n: (n["start"], n["pitch"], n["stem"]))
        missing = [s["stem"] for s in stem_records if s.get("skipped") == "missing"]
        return {"duration": round(len(wav) / sr, 4), "complete": not missing,
                **({"missing_stems": missing} if missing else {}), "stems": stem_records, "notes": notes,
                "timing": {k: round(v, 3) for k, v in timing.items()}}

    def _context_wav(self, stems, mix: np.ndarray, sr: int, stem) -> np.ndarray:
        from .audio import resample

        gain = 10 ** (float(self.context.get("gain_db", 0.0)) / 20)
        if self.context["source"] == "drums":
            drums = next((s for s in stems if s.name == "drums" and len(s.wav)), None)
            if drums is None:
                return np.zeros_like(stem.wav)
            ctx = resample(drums.wav, drums.sr, stem.sr) if drums.sr != stem.sr else drums.wav
        elif self.context["source"] == "click":
            if getattr(self, "_click_for", None) is not mix:
                from beat_this.inference import Audio2Beats
                if not hasattr(self, "_beats"):
                    self._beats = Audio2Beats(checkpoint_path="final0", device=self.device, dbn=False)
                beats, downbeats = self._beats(mix, sr)
                self._click_for, self._click_beats = mix, (np.asarray(beats), np.asarray(downbeats))
            beats, downbeats = self._click_beats
            ctx = np.zeros_like(stem.wav)
            n = int(0.02 * stem.sr)
            env = np.exp(-np.linspace(0, 8, n)).astype(np.float32)
            rng = np.random.default_rng(0)
            for b in beats:
                a = int(b * stem.sr)
                if a >= len(ctx):
                    break
                amp = 1.0 if np.any(np.abs(downbeats - b) < 0.02) else 0.6
                seg = min(n, len(ctx) - a)
                ctx[a:a + seg] += amp * env[:seg] * rng.uniform(-1, 1, seg).astype(np.float32)
        else:
            raise ValueError(f"unknown stem_context source {self.context['source']!r}")
        ctx = ctx[: len(stem.wav)]
        ctx = np.pad(ctx, (0, len(stem.wav) - len(ctx)))
        # level the context to the stem: equal RMS at 0 dB
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        return (gain * rms(stem.wav) / rms(ctx) * ctx).astype(np.float32)

    def describe(self) -> dict:
        return {"config_sha256": hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest(),
                "transcriber": getattr(self.transcriber, "describe", None)}


def run_manifest(cfg: dict, manifest: Path, out: Path, audio_root: Path | None = None,
                 shard: int = 0, num_shards: int = 1, device: str = "cuda",
                 cube_fps: float | None = None, stage: str = "all",
                 follow_tags: bool = False, poll_seconds: float = 60.0, midi: bool = False,
                 overwrite: bool = False) -> dict:
    """One JSON per clip under `out/clips/` (or `out/tags/` for stage `tag`), plus a MIDI
    file with `midi`; existing
    outputs are skipped (resume). Resuming into a directory written with a different
    config is refused unless `overwrite`, which then redoes every clip. With `follow_tags` and a `precomputed` tagger, clips
    whose tags are not written yet are deferred and revisited until all are done, so the
    transcription stage can run alongside the tag stage."""
    tags_dir = (Path(cfg["tagger"]["dir"]) / "tags"
                if follow_tags and cfg.get("tagger", {}).get("name") == "precomputed" else None)
    total = {"done": 0, "audio_seconds": 0.0, "passes": 0}
    started = time.time()
    while True:
        s = _run_pass(cfg, manifest, out, audio_root, shard, num_shards, device, cube_fps, stage,
                      tags_dir, midi, overwrite)
        total["done"] += s["done"]
        total["audio_seconds"] += s["audio_seconds"]
        total["passes"] += 1
        if not s["deferred"]:
            break
        time.sleep(poll_seconds)
    el = time.time() - started
    summary = {"shard": shard, **total, "seconds": round(el, 1),
               "seconds_per_audio_minute": round(el / (total["audio_seconds"] / 60), 2)
               if total["audio_seconds"] else None}
    print(json.dumps(summary), flush=True)
    return summary


_PIPELINES: dict = {}


def _run_pass(cfg, manifest, out, audio_root, shard, num_shards, device, cube_fps, stage,
              tags_dir, midi=False, overwrite=False) -> dict:
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    rows = [r for i, r in enumerate(rows) if i % num_shards == shard]
    sub = "tags" if stage == "tag" else "clips"
    (out / sub).mkdir(parents=True, exist_ok=True)
    import yaml
    resolved = out / f"resolved_config{'.tag' if stage == 'tag' else ''}.yaml"
    redo = resolved.exists() and yaml.safe_load(resolved.read_text()) != cfg
    if redo and not overwrite:
        raise SystemExit(f"{out} holds outputs of a different config ({resolved}); use another "
                         "--out, or --overwrite to redo every clip with this config")
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    key = (id(cfg), device)
    if key not in _PIPELINES:
        _PIPELINES[key] = Pipeline(cfg, device)
    pipe = _PIPELINES[key]
    started, done, audio_seconds, deferred = time.time(), 0, 0.0, 0
    for row in rows:
        clip_id = row.get("id") or Path(row["audio"]).stem + (f"@{row['offset']:g}" if row.get("offset") else "")
        target = out / sub / f"{clip_id}.json"
        if target.exists() and not redo:
            continue
        if tags_dir is not None and not (tags_dir / f"{clip_id}.json").exists():
            deferred += 1
            continue
        path = Path(row["audio"])
        if audio_root is not None and not path.is_absolute():
            path = audio_root / path
        wav = load_mono(path, pipe.input_sr, row.get("offset"), row.get("duration"))
        head = {"id": clip_id, "audio": str(path), "offset": row.get("offset"),
                "duration_requested": row.get("duration"), "duration": round(len(wav) / pipe.input_sr, 4)}
        if stage == "tag":
            record = {**head, **pipe.tag_clip(wav, pipe.input_sr, {**row, "id": clip_id})}
        else:
            record = {**head, **pipe(wav, pipe.input_sr, {**row, "id": clip_id}),
                      "pipeline": pipe.describe()}
        if cube_fps and stage != "tag":
            from .render import cube
            fams = list(pipe.taxonomy.families)
            np.savez_compressed(out / "clips" / f"{clip_id}.cube.npz", families=np.array(fams),
                                cube=cube(record["notes"], fams, record["duration"], cube_fps))
        if midi and stage != "tag":
            from .render import to_midi
            to_midi(record["notes"], out / "clips" / f"{clip_id}.mid")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(target)
        done += 1
        audio_seconds += record["duration"]
        if done % 20 == 0:
            el = time.time() - started
            print(json.dumps({"shard": shard, "done": done, "of": len(rows), "seconds": round(el, 1),
                              "seconds_per_audio_minute": round(el / (audio_seconds / 60), 2)}), flush=True)
    el = time.time() - started
    print(json.dumps({"shard": shard, "pass_done": done, "deferred": deferred, "of": len(rows),
                      "seconds": round(el, 1)}), flush=True)
    return {"done": done, "audio_seconds": audio_seconds, "deferred": deferred}
