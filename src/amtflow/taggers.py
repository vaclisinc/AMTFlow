"""Instrument taggers: audio -> routing families.

A tagger returns a `TagResult`; `abstained` means it could not vote (e.g. an ALM whose
endpoint is unreachable), which differs from voting for no family. Taggers are
registered by name and combined by `vote` (union, majority or at-least-k over any
members, ALMs included). MuScriptor masks every instrument outside the constraint,
so recall matters more than precision.

    matpac  MATPAC++ with the authors' released probes (paper)
    probe   any registered encoder + an AMTFlow OpenMIC head
    alm     Music Flamingo or Qwen3-Omni, asked an open or a closed question
    fixed   a constant family set
    vote    combine members (`union` is `vote` with policy union)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .families import Taxonomy, normalize_label


@dataclass
class TagResult:
    source: str
    families: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    abstained: bool = False
    members: list[dict] = field(default_factory=list)
    policy: str | None = None
    groups: dict[str, float] = field(default_factory=dict)    # MuScriptor group -> evidence
    # optional per-window families (each `window_seconds` long, from 0), a subset of
    # `families`, for window-level constraints
    windows: list[list[str]] | None = None
    window_seconds: float | None = None

    def to_json(self) -> dict:
        return {"source": self.source, "families": self.families, "abstained": self.abstained,
                **({"policy": self.policy} if self.policy else {}),
                **({"scores": {k: round(v, 4) for k, v in self.scores.items()}} if self.scores else {}),
                **({"groups": {k: round(v, 4) for k, v in self.groups.items()}} if self.groups else {}),
                **({"labels": self.labels} if self.labels else {}),
                **({"members": self.members} if self.members else {})}


def window_starts(total: int, window: int, hop: int) -> list[int]:
    """Starts covering the whole clip, including its final sample."""
    if total <= window:
        return [0]
    last = total - window
    starts = list(range(0, last + 1, hop))
    if starts[-1] != last:
        starts.append(last)
    return starts


def top_fraction_mean(values: list[float], fraction: float, minimum: int) -> float:
    k = min(len(values), max(minimum, math.ceil(len(values) * fraction)))
    return float(np.mean(sorted(values, reverse=True)[:k]))


def load_thresholds(cfg: dict, key: str) -> dict[str, float]:
    """`thresholds`: a {name: value} mapping or a JSON file. A file may be flat or
    nested ({encoder: {family: {"threshold": v}}}), selected by
    `thresholds_key` (default: this tagger's name). `threshold_factor` scales every
    value (capped at 1)."""
    t = cfg["thresholds"]
    if isinstance(t, (str, Path)):
        t = json.load(open(t))
        t = t.get(cfg.get("thresholds_key", key), t)
    t = {k: float(v["threshold"] if isinstance(v, dict) else v) for k, v in t.items()}
    factor = float(cfg.get("threshold_factor", 1.0))
    return {k: min(1.0, v * factor) for k, v in t.items()}


def label_group_scores(label_scores: dict[str, float], taxonomy: Taxonomy) -> dict[str, float]:
    """Group evidence from label scores: each specific label votes its score for the
    groups it names (generic labels name none); a group keeps its best vote."""
    out: dict[str, float] = {}
    for label, score in label_scores.items():
        for g in taxonomy.groups_of_label(label):
            out[g] = max(out.get(g, 0.0), float(score))
    return out


def _windows(y: np.ndarray, sr: int, window_s: float, hop_s: float):
    w, h = round(window_s * sr), round(hop_s * sr)
    for s in window_starts(len(y), w, h):
        chunk = y[s:s + w]
        yield np.pad(chunk, (0, w - len(chunk))) if len(chunk) < w else chunk


class MatpacTagger:
    """MATPAC++ with the authors' released linear probes (e.g. OpenMIC, MTG instrument).

    Each window's family score is the max sigmoid over that family's labels across
    heads; a clip score averages the strongest `top_fraction` of windows; a family is
    detected when its score reaches its threshold.
    """

    name = "matpac"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        from .assets import resolve
        from .encoders.matpac import Matpac

        self.enc = Matpac(cfg.get("variant", "music")).load(device)
        self.window, self.hop = cfg.get("window_seconds", 6.0), cfg.get("hop_seconds", 6.0)
        self.fraction, self.minimum = cfg.get("top_fraction", 0.1), cfg.get("minimum_windows", 2)
        self.thresholds = load_thresholds(cfg, self.name)
        root = resolve(cfg.get("probe_dir"), "matpac-probes")
        if (root / "Weights_probes_matpac").is_dir():
            root = root / "Weights_probes_matpac"
        probe_dir = root / f"matpac++_{cfg.get('variant', 'music')}"
        mapping_dir = Path(cfg.get("mapping_dir", Path(__file__).parent / "class_mappings"))
        self.heads = []
        for head in cfg["heads"]:
            state = torch.load(probe_dir / f"{head}.pth", map_location=device, weights_only=True)
            labels = json.load(open(mapping_dir / f"{head}_mapping.json"))
            by_family: dict[str, list[int]] = {}
            for label, idx in labels.items():
                fam = taxonomy.label_families.get(normalize_label(label))
                if fam in self.thresholds:
                    by_family.setdefault(fam, []).append(int(idx))
            names = [None] * len(labels)
            for label, idx in labels.items():
                names[int(idx)] = normalize_label(label)
            self.heads.append((state["probe.weight"], state["probe.bias"], by_family, names))
        missing = set(self.thresholds) - {f for _, _, bf, _ in self.heads for f in bf}
        self.taxonomy = taxonomy
        if missing:
            raise ValueError(f"no MATPAC labels for thresholded families {sorted(missing)}")

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        from .audio import resample
        y = resample(wav, sr, self.enc.sample_rate)
        per_window: dict[str, list[float]] = {f: [] for f in self.thresholds}
        per_label: dict[str, list[float]] = {}
        for chunk in _windows(y, self.enc.sample_rate, self.window, self.hop):
            emb = self.enc.layer_features(chunk)[-1].float()
            probs = [torch.sigmoid(emb @ wt.T + b) for wt, b, _, _ in self.heads]
            for fam in per_window:
                per_window[fam].append(max(float(p[idx].max()) for p, (_, _, bf, _) in
                                           zip(probs, self.heads) for idx in [bf.get(fam)] if idx))
            for p, (_, _, _, names) in zip(probs, self.heads):
                for i, name in enumerate(names):
                    per_label.setdefault(name, []).append(float(p[i]))
        scores = {f: top_fraction_mean(v, self.fraction, self.minimum) for f, v in per_window.items()}
        # A label present in both heads (e.g. "guitar") pools both heads' windows.
        labels = {n: top_fraction_mean(v, self.fraction, self.minimum) for n, v in per_label.items()}
        return TagResult(self.name, [f for f, s in scores.items() if s >= self.thresholds[f]], scores,
                         groups=label_group_scores(labels, self.taxonomy))


class ProbeTagger:
    """A registered encoder with an AMTFlow OpenMIC head (from `amtflow openmic export-head`).

    The head file carries its layer, feature normalization, classes and per-class
    thresholds calibrated on OpenMIC validation. Per window the head gives class
    probabilities; a class score averages the strongest `top_fraction` of windows; a
    family is detected when any of its classes reaches its class threshold.
    """

    name = "probe"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        from .encoders import build
        from .tagging.heads import load_head

        self.head = load_head(cfg["head"], device)
        self.name = f"probe:{self.head.encoder}"
        self.enc = build(cfg.get("encoder", self.head.encoder)).load(device)
        self.window = cfg.get("window_seconds", self.head.clip_seconds)
        self.hop = cfg.get("hop_seconds", self.window)
        self.fraction, self.minimum = cfg.get("top_fraction", 0.1), cfg.get("minimum_windows", 2)
        self.thresholds = (load_thresholds(cfg, self.name) if "thresholds" in cfg
                           else dict(zip(self.head.classes, self.head.thresholds)))
        self.family_of = {c: taxonomy.label_families.get(normalize_label(c)) for c in self.head.classes}
        self.taxonomy = taxonomy

    @torch.inference_mode()
    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        from .audio import resample
        y = resample(wav, sr, self.enc.sample_rate)
        probs = [self.head(self.enc.layer_features(chunk))
                 for chunk in _windows(y, self.enc.sample_rate, self.window, self.hop)]
        P = torch.stack(probs).cpu().numpy()                      # (windows, classes)
        cls = {c: top_fraction_mean(P[:, i].tolist(), self.fraction, self.minimum)
               for i, c in enumerate(self.head.classes)}
        hits = [c for c, s in cls.items() if s >= self.thresholds[c]]
        fams = [self.family_of[c] for c in hits if self.family_of[c]]
        fam_scores: dict[str, float] = {}
        for c, s in cls.items():
            f = self.family_of[c]
            if f:
                fam_scores[f] = max(fam_scores.get(f, 0.0), s)
        return TagResult(self.name, self.taxonomy.order(set(fams)), fam_scores, labels=hits,
                         groups=label_group_scores(cls, self.taxonomy))


# ----------------------------------------------------------------------------- ALMs
OPEN_PROMPT = ("List every instrument that is clearly audible in this audio. "
               "Return only a comma-separated list of instrument names, or none.")
CLOSED_PROMPT = ("Which broad instrument families are clearly audible in this audio? "
                 "Choose only from: piano, guitar, bass, drums, percussion, strings, winds, "
                 "organ, synthesizer, vocals. Return only a comma-separated list using "
                 "exactly these names, or none.")

ALM_MODELS = {
    "music-flamingo": ("nvidia/music-flamingo-hf", "35a2c9071753ee075b0f7fc2fd81151c21389530"),
    "qwen3-omni": ("Qwen/Qwen3-Omni-30B-A3B-Instruct", None),
}
_LOADED_ALMS: dict[tuple, object] = {}      # one copy per (model, device) per process


class LocalALM:
    """An ALM loaded in-process with transformers (downloaded on first use)."""

    def __init__(self, model: str, device: str, max_new_tokens: int = 64):
        self.kind = model
        self.repo, self.revision = ALM_MODELS[model]
        self.device, self.max_new_tokens = device, max_new_tokens
        if model == "music-flamingo":
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.repo, revision=self.revision)
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                self.repo, revision=self.revision, dtype=torch.bfloat16).to(device).eval()
        else:
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.repo, revision=self.revision)
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.repo, revision=self.revision, dtype=torch.bfloat16).to(device).eval()
            self.model.disable_talker()

    @torch.inference_mode()
    def ask(self, wav16k: np.ndarray, prompt: str) -> str:
        if self.kind == "music-flamingo":
            conv = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                 {"type": "audio", "audio": wav16k}]}]
            inputs = self.processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True).to(self.device)
            inputs = {k: (v.to(self.model.dtype) if torch.is_floating_point(v) else v)
                      for k, v in inputs.items()}
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        else:
            conv = [{"role": "user", "content": [{"type": "audio", "audio": wav16k},
                                                 {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, audio=[wav16k], return_tensors="pt", padding=True,
                                    use_audio_in_video=False).to(self.device)
            inputs = {k: (v.to(self.model.dtype) if torch.is_floating_point(v) else v)
                      for k, v in inputs.items()}
            # The full model's generate (talker off) sets the thinker's EOS and multimodal
            # positions; thinker.generate alone decodes garbage after the first token.
            out = self.model.generate(**inputs, return_audio=False, use_audio_in_video=False,
                                      thinker_max_new_tokens=self.max_new_tokens,
                                      thinker_do_sample=False)
            out = out[0] if isinstance(out, tuple) else out
            out = getattr(out, "sequences", out)
        n = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()


class HttpALM:
    """A served ALM: multipart POST `{base_url}{path}` with an
    `audio` file and `prompt`/`max_new_tokens`/`repetition_penalty`; JSON `text`/`model`."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def ask(self, wav16k: np.ndarray, prompt: str) -> str:
        import io
        import os
        import requests
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wav16k, 16000, format="WAV")
        buf.seek(0)
        c = self.cfg
        key = os.environ.get(c.get("api_key_env", "VLLM_API_KEY"), "")
        r = requests.post(f"{c['base_url'].rstrip('/')}{c.get('path', '/generate')}",
                          headers={"Authorization": f"Bearer {key}"} if key else {},
                          files={"audio": ("clip.wav", buf, "audio/wav")},
                          data={"prompt": prompt, "max_new_tokens": c.get("max_new_tokens", 64),
                                "repetition_penalty": c.get("repetition_penalty", 1.2)},
                          timeout=c.get("timeout_seconds", 240))
        r.raise_for_status()
        payload = r.json()
        if c.get("expected_model") and payload.get("model") != c["expected_model"]:
            raise RuntimeError(f"endpoint serves {payload.get('model')!r}, expected {c['expected_model']!r}")
        return str(payload["text"])


class ALMTagger:
    """An audio-language model answering an instrument question.

    `model`: music-flamingo | qwen3-omni; `backend`: local (default, in-process,
    downloaded on first use) or http (`base_url`; abstains when unset or failing);
    `prompt`: open | closed | both (both answers pooled). Answers are split on commas
    and mapped to families through the taxonomy plus `label_overrides`.
    """

    name = "alm"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.cfg, self.taxonomy = cfg, taxonomy
        self.model = cfg.get("model", "music-flamingo")
        prompt = cfg.get("prompt", "both")
        self.prompts = {"open": [OPEN_PROMPT], "closed": [CLOSED_PROMPT],
                        "both": [OPEN_PROMPT, CLOSED_PROMPT]}[prompt]
        self.name = f"alm:{self.model}:{prompt}"
        self.label_families = {**taxonomy.label_families,
                               **{normalize_label(k): v for k, v in cfg.get("label_overrides", {}).items()}}
        self.backend = cfg.get("backend", "local")
        self.client = None
        if self.backend == "http":
            self.client = HttpALM(cfg) if cfg.get("base_url") else None
        else:
            key = (self.model, device)
            if key not in _LOADED_ALMS:
                _LOADED_ALMS[key] = LocalALM(self.model, device, cfg.get("max_new_tokens", 64))
            self.client = _LOADED_ALMS[key]
        self.max_seconds = cfg.get("max_audio_seconds", 600)

    def _ask(self, wav16k: np.ndarray, prompt: str) -> str:
        return self.client.ask(wav16k, prompt)

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        if self.client is None and self.backend == "http":
            return TagResult(self.name, [], abstained=True)
        from .audio import resample
        y = resample(wav, sr, 16000)[: int(self.max_seconds * 16000)]
        try:
            answers = [self._ask(y, p) for p in self.prompts]
        except OSError:
            return TagResult(self.name, [], abstained=True)
        except Exception as e:        # HTTP errors abstain; local failures are bugs
            if self.backend == "http":
                return TagResult(self.name, [], abstained=True, labels=[f"error: {type(e).__name__}"])
            raise
        labels = [p.strip().strip(".") for a in answers for p in a.replace("/", ",").split(",")
                  if p.strip() and normalize_label(p) not in {"none", "noinstruments"}]
        fams = [f for x in labels for f in self.families_of(x)]
        return TagResult(self.name, self.taxonomy.order(fams), labels=labels,
                         groups={g: 1.0 for x in labels for g in self.groups_of(x)})

    def groups_of(self, label: str) -> set[str]:
        """Specific groups a free-text answer names: exact label first, then its words
        and adjacent word pairs ("distorted electric guitar" -> distorted_electric_guitar)."""
        if self.taxonomy.groups_of_label(label):
            return set(self.taxonomy.groups_of_label(label))
        words = [normalize_label(w) for w in re.findall(r"[A-Za-z0-9]+", label)]
        grams = words + [a + b for a, b in zip(words, words[1:])]
        by_family: dict[str, set[str]] = {}
        for x in grams:
            hit = set(self.taxonomy.groups_of_label(x))
            if not hit:
                continue
            f = self.taxonomy.family_of_instrument.get(next(iter(hit)))
            # words that narrow the same family intersect ("distorted" ∩ "electric guitar")
            by_family[f] = by_family[f] & hit if f in by_family and by_family[f] & hit else \
                by_family.get(f, set()) | hit
        return set().union(*by_family.values()) if by_family else set()

    def families_of(self, label: str) -> set[str]:
        """Exact label match first; otherwise every family named by one of the label's
        words or adjacent word pairs ("synth bass" -> synthesizer, bass). Free-text
        answers are open-ended, and a missed family is worse than an extra one."""
        key = normalize_label(label)
        if key in self.label_families:
            return {self.label_families[key]} - {None}
        words = [normalize_label(w) for w in re.findall(r"[A-Za-z0-9]+", label)]
        grams = words + [a + b for a, b in zip(words, words[1:])]
        return {self.label_families[g] for g in grams if self.label_families.get(g)}


class FixedTagger:
    name = "fixed"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.families = taxonomy.order(cfg["families"])

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        return TagResult(self.name, list(self.families))


class VoteTagger:
    """Combine member taggers; abstaining members do not count.

    policy: union (any member), majority (more than half of the members that voted),
    at_least (`k` members). When nothing is selected: `fallback: plurality` takes the
    families with the most votes (if any), then the taxonomy's fallback families;
    `fallback: default` goes straight to them; `fallback: none` returns no family.
    """

    name = "vote"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.members = [build_tagger(m, taxonomy, device) for m in cfg["members"]]
        self.taxonomy = taxonomy
        self.policy = cfg.get("policy", "union")
        self.k = int(cfg.get("k", 1))
        self.fallback = cfg.get("fallback", "plurality" if self.policy != "union" else "default")
        if self.policy not in {"union", "majority", "at_least"}:
            raise ValueError(f"unknown vote policy {self.policy!r}")

    def combine(self, votes: list[TagResult]) -> TagResult:
        voted = [v for v in votes if not v.abstained]
        counts = Counter(f for v in voted for f in set(v.families))
        need = {"union": 1, "majority": len(voted) // 2 + 1, "at_least": self.k}[self.policy]
        fams = self.taxonomy.order(f for f, c in counts.items() if c >= need) if voted else []
        how = f"{self.policy}>={need}"
        if not fams and self.fallback == "plurality" and counts:
            top = max(counts.values())
            fams, how = self.taxonomy.order(f for f, c in counts.items() if c == top), "plurality"
        if not fams and self.fallback in {"plurality", "default"}:
            fams, how = list(self.taxonomy.fallback), "fallback"
        groups: dict[str, float] = {}
        for v in voted:
            for g, sc in v.groups.items():
                groups[g] = max(groups.get(g, 0.0), sc)
        return TagResult(self.name, fams, {f: float(c) for f, c in counts.items()}, policy=how,
                         groups=groups,
                         members=[v.to_json() for v in votes])

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        return self.combine([m(wav, sr, **ctx) for m in self.members])


class OracleTagger:
    """Ground-truth families from the clip's reference MIDI (manifest field `reference`):
    every GM program present, mapped through its MuScriptor group to a routing family.
    For benchmarking the rest of the pipeline under perfect instrument recognition.
    `families` (optional) keeps only those families, e.g. the ones a separator has stems for."""

    name = "oracle"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.taxonomy = taxonomy
        self.group_family = cfg.get("group_families", {
            **{g: f for g, f in (("acoustic_piano", "piano"), ("electric_piano", "piano"),
                                 ("acoustic_guitar", "guitar"), ("clean_electric_guitar", "guitar"),
                                 ("distorted_electric_guitar", "guitar"), ("acoustic_bass", "bass"),
                                 ("electric_bass", "bass"), ("synth_lead", "synthesizer"),
                                 ("synth_pad", "synthesizer"))},
            **{g: "strings" for g in ("violin", "viola", "cello", "contrabass", "orchestral_harp",
                                      "string_ensemble", "synth_strings")},
            **{g: "winds" for g in ("trumpet", "trombone", "tuba", "french_horn", "brass_section",
                                    "soprano_and_alto_sax", "tenor_sax", "baritone_sax", "oboe",
                                    "english_horn", "bassoon", "clarinet", "flutes")}})
        self.only = set(cfg["families"]) if cfg.get("families") else None
        # `window_seconds`: also report the families sounding in each window (oracle
        # upper bound for window-level constraints)
        self.window = float(cfg["window_seconds"]) if cfg.get("window_seconds") else None

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        from .multtipop import DRUMS, _groups, midi_notes
        ref = (ctx.get("row") or {}).get("reference")
        if not ref:
            return TagResult(self.name, [], abstained=True)
        of_program = _groups()[0]
        notes = midi_notes(ref)
        fams = {"drums" if p == DRUMS else self.group_family.get(of_program.get(p, ""))
                for *_, p in notes}
        if self.only is not None:
            fams &= self.only
        counts = Counter("drums" if p == DRUMS else of_program.get(p) for *_, p in notes)
        total = max(sum(counts.values()), 1)
        windows = None
        if self.window:
            fam_of = lambda p: "drums" if p == DRUMS else self.group_family.get(of_program.get(p, ""))
            n_win = int(np.ceil(len(wav) / sr / self.window))
            windows = [set() for _ in range(max(n_win, 1))]
            for on, off, _, p in notes:
                f = fam_of(p)
                if not f or (self.only is not None and f not in self.only):
                    continue
                for w in range(int(on // self.window), min(int(max(off, on) // self.window), n_win - 1) + 1):
                    windows[w].add(f)
            windows = [self.taxonomy.order(w) for w in windows]
        return TagResult(self.name, self.taxonomy.order(f for f in fams if f),
                         groups={g: n / total for g, n in counts.items() if g},
                         windows=windows, window_seconds=self.window)


class PrecomputedTagger:
    """Families written by an earlier `amtflow run --stage tag` (`<dir>/tags/<id>.json`),
    looked up by clip id and stem, so large taggers need not live in every worker."""

    name = "precomputed"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.dir = Path(cfg["dir"]) / "tags"
        self.taxonomy = taxonomy

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        clip = (ctx.get("row") or {}).get("id")
        rec = json.load(open(self.dir / f"{clip}.json"))
        for s in rec["stems"]:
            if s["stem"] == ctx.get("stem"):
                if "families" not in s:
                    raise KeyError(f"{clip}/{s['stem']} was not tagged ({s.get('skipped')})")
                t = s.get("tagger") or {}
                return TagResult(self.name, self.taxonomy.order(s["families"]),
                                 members=[t] if t else [], groups=t.get("groups", {}))
        raise KeyError(f"no tags for {clip}/{ctx.get('stem')}")


class MuScriptorSelfTagger:
    """MuScriptor's own instrument choice: an unconstrained pass over the audio, each
    group's evidence its share of the pitched notes. Families are those groups' families
    (set `vote_families: false` to use it only as a routing source)."""

    name = "muscriptor_self"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        from .transcribers import MuScriptor
        self.tr = MuScriptor({"weights": cfg.get("weights", "medium"),
                              "batch_size": cfg.get("batch_size", 1),
                              "prelude_forcing": cfg.get("prelude_forcing", True)}, device)
        self.taxonomy = taxonomy
        self.min_share = float(cfg.get("min_share", 0.02))
        self.vote_families = cfg.get("vote_families", True)

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        notes = self.tr(wav, sr, None)
        counts = Counter(n.instrument for n in notes)
        total = max(sum(counts.values()), 1)
        groups = {g: c / total for g, c in counts.items() if c / total >= self.min_share}
        fams = {self.taxonomy.family_of_instrument.get(g) for g in groups} if self.vote_families else set()
        return TagResult(self.name, self.taxonomy.order(f for f in fams if f), groups=groups)


def _union(cfg, taxonomy, device):
    return VoteTagger({**cfg, "policy": "union"}, taxonomy, device)


# MVSep Mega 53-stem BS-RoFormer (ZFTurbo MSST release v1.0.21) stem -> (family, groups).
# Vocals and unpitched/chromatic percussion route nowhere.
MVSEP53_STEMS: dict[str, tuple[str | None, list[str]]] = {
    "accordion": ("winds", []), "acoustic-guitar": ("guitar", ["acoustic_guitar"]),
    "back-vocal": (None, []), "banjo": ("guitar", ["acoustic_guitar"]),
    "bass": ("bass", ["electric_bass"]), "bassoon": ("winds", ["bassoon"]), "bells": (None, []),
    "bowed_strings": ("strings", ["string_ensemble"]), "brass": ("winds", ["brass_section"]),
    "cello": ("strings", ["cello"]), "clarinet": ("winds", ["clarinet"]), "congas": (None, []),
    "digital-piano": ("piano", ["electric_piano"]), "dobro": ("guitar", ["acoustic_guitar"]),
    "double-bass": ("bass", ["acoustic_bass"]), "drums": ("drums", ["drums"]),
    "electric-guitar": ("guitar", ["clean_electric_guitar", "distorted_electric_guitar"]),
    "flute": ("winds", ["flutes"]), "french-horn": ("winds", ["french_horn"]),
    "glockenspiel": (None, []), "guitar": ("guitar", []), "harmonica": ("winds", []),
    "harp": ("strings", ["orchestral_harp"]), "harpsichord": ("piano", ["acoustic_piano"]),
    "hh": ("drums", ["drums"]), "keys": ("piano", []), "kick": ("drums", ["drums"]),
    "lead-vocal": (None, []), "mandolin": ("guitar", ["acoustic_guitar"]), "marimba": (None, []),
    "oboe": ("winds", ["oboe"]), "organ": (None, []), "percussion": (None, []),
    "piano": ("piano", ["acoustic_piano"]), "saxophone": ("winds", ["soprano_and_alto_sax", "tenor_sax"]),
    "sitar": ("guitar", ["acoustic_guitar"]), "snare": ("drums", ["drums"]),
    "strings": ("strings", ["string_ensemble"]), "synth": ("synthesizer", []),
    "tambourine": (None, []), "timpani": (None, []), "toms": ("drums", ["drums"]),
    "triangle": (None, []), "trombone": ("winds", ["trombone"]), "trumpet": ("winds", ["trumpet"]),
    "tuba": ("winds", ["tuba"]), "ukulele": ("guitar", ["acoustic_guitar"]),
    "viola": ("strings", ["viola"]), "violin": ("strings", ["violin"]), "vocal": (None, []),
    "wind": ("winds", []), "wind-chimes": (None, []), "woodwind": ("winds", []),
}


class StemEnergyTagger:
    """Instruments from a many-stem separator: a stem is present when its energy relative
    to the mixture exceeds `threshold_db` (per-stem overrides in `thresholds_db`) and at
    least `min_coverage` percent of its 100 ms blocks exceed -40 dBFS. Group evidence is
    the stem's margin over its threshold, scaled to [0, 1] over `scale_db`.

    Default model: MVSep Mega 53-stem BS-RoFormer (`checkpoint`, `config` paths). With
    `cache_dir`, per-clip stem statistics are stored and reused, so thresholds can be
    changed without separating again."""

    name = "stems53"

    def __init__(self, cfg: dict, taxonomy: Taxonomy, device: str):
        self.cfg, self.taxonomy, self.device = cfg, taxonomy, device
        self.threshold = float(cfg.get("threshold_db", -20.0))
        self.thresholds = {k: float(v) for k, v in (cfg.get("thresholds_db") or {}).items()}
        self.min_coverage = float(cfg.get("min_coverage", 5.0))
        self.scale = float(cfg.get("scale_db", 15.0))
        self.cache = Path(cfg["cache_dir"]) if cfg.get("cache_dir") else None
        self.model = None

    def _load(self):
        import torch
        import yaml
        from bs_roformer.inference import SafeLoaderWithTuple
        from bs_roformer.utils import get_model_from_config, load_checkpoint_state
        from ml_collections import ConfigDict

        self.config = ConfigDict(yaml.load(Path(self.cfg["config"]).read_text(), Loader=SafeLoaderWithTuple))
        # bs_roformer's iSTFT returns whole hops: keep chunks a multiple of the hop length
        hop = int(self.config.model.stft_hop_length)
        for sec in ("audio", "inference"):
            if sec in self.config and "chunk_size" in self.config[sec]:
                self.config[sec].chunk_size = -(-int(self.config[sec].chunk_size) // hop) * hop
        model = get_model_from_config("bs_roformer", self.config)
        model.load_state_dict(load_checkpoint_state(self.cfg["checkpoint"], map_location=torch.device("cpu")))
        self.model = model.to(self.device).eval()
        self.first_chunk_time = None

    def stats(self, wav: np.ndarray, sr: int) -> dict[str, list[float]]:
        """stem -> [energy relative to the mixture in dB, coverage percent]."""
        import torch
        from bs_roformer.utils import demix_track

        from .audio import resample
        from .separators import coverage

        if self.model is None:
            self._load()
        y = resample(wav, sr, 44100) if sr != 44100 else wav
        stereo = torch.from_numpy(np.stack([y, y]).astype(np.float32))
        with torch.no_grad():
            res, self.first_chunk_time = demix_track(self.config, self.model, stereo, torch.device(self.device),
                                                     self.first_chunk_time)
        e_mix = float((y.astype(np.float64) ** 2).sum()) + 1e-12
        out = {}
        for name, value in res.items():
            mono = np.asarray(value).mean(axis=0).astype(np.float32)
            rel = 10 * np.log10(float((mono.astype(np.float64) ** 2).sum()) / e_mix + 1e-12)
            out[name] = [round(rel, 2), round(coverage(mono, 44100), 2)]
        return out

    def __call__(self, wav: np.ndarray, sr: int, **ctx) -> TagResult:
        clip = (ctx.get("row") or {}).get("id")
        key = f"{clip}.{ctx.get('stem') or 'mix'}.json" if clip else None
        if self.cache is not None and key and (self.cache / key).exists():
            st = json.load(open(self.cache / key))
        else:
            st = self.stats(wav, sr)
            if self.cache is not None and key:
                self.cache.mkdir(parents=True, exist_ok=True)
                (self.cache / key).write_text(json.dumps(st))
        fams, groups, scores, labels = set(), {}, {}, []
        for stem, (rel, cov) in st.items():
            fam, gs = MVSEP53_STEMS.get(stem, (None, []))
            margin = rel - self.thresholds.get(stem, self.threshold)
            if margin <= 0 or cov < self.min_coverage:
                continue
            labels.append(stem)
            ev = min(1.0, margin / self.scale)
            if fam:
                fams.add(fam)
                scores[fam] = max(scores.get(fam, 0.0), ev)
            for g in gs:
                groups[g] = max(groups.get(g, 0.0), ev)
        return TagResult(self.name, self.taxonomy.order(fams), scores=scores, labels=labels,
                         groups=groups)


REGISTRY: dict[str, Callable] = {
    "matpac": MatpacTagger,
    "probe": ProbeTagger,
    "alm": ALMTagger,
    "fixed": FixedTagger,
    "oracle": OracleTagger,
    "precomputed": PrecomputedTagger,
    "stems53": StemEnergyTagger,
    "muscriptor_self": MuScriptorSelfTagger,
    "vote": VoteTagger,
    "union": _union,
}


def build_tagger(cfg: dict, taxonomy: Taxonomy, device: str):
    """`device` in a tagger's config overrides the pipeline device (e.g. put a large ALM
    on its own GPU)."""
    if cfg["name"] not in REGISTRY:
        raise KeyError(f"unknown tagger {cfg['name']!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[cfg["name"]](cfg, taxonomy, cfg.get("device", device))
