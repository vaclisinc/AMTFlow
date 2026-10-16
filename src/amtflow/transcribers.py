"""AMT models: (audio, instrument constraint) -> notes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    instrument: str

    def to_json(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = round(self.start, 4), round(self.end, 4)
        return d


class MuScriptor:
    """MuScriptor (Kyutai x Mirelo). `instruments` is a hard constraint on its decoder."""

    def __init__(self, cfg: dict, device: str):
        from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
        from muscriptor.transcription_model import TranscriptionModel

        self.known = set(MT3_FULL_PLUS_GROUP_NAMES)
        self.model = TranscriptionModel.load_model(cfg.get("weights", "large"), device=device,
                                                   dtype=cfg.get("dtype"))
        self.batch_size = cfg.get("batch_size", 1)
        self.prelude_forcing = cfg.get("prelude_forcing", True)
        # Classifier-free guidance on the conditions (audio + instrument group). MuScriptor's
        # API and CLI default to 1.0 (off); its paper decodes with 2.0. Doubles the batch.
        self.cfg_coef = float(cfg.get("cfg_coef", 1.0))
        # `logit_bias`: added to the decoder's logits by event type (pitch, shift, velocity,
        # program, drum, tie) or `eos`, e.g. {pitch: 1.0} favors emitting more notes.
        self.logit_bias = {k: float(v) for k, v in (cfg.get("logit_bias") or {}).items()}
        if self.logit_bias:
            self._install_logit_bias()
        # MuScriptor decodes 5 s chunks; a final chunk of a few near-silent frames makes
        # it emit hundreds of spurious notes, some timed past the end of the audio. Such
        # a tail is trimmed, and notes starting after the audio are dropped.
        self.min_tail = float(cfg.get("min_tail_seconds", 1.0))
        # `boundaries: quiet` starts each 5 s window at the quietest point (onset strength +
        # RMS) in [previous + min_window, previous + 5] s instead of every 5 s. Each window
        # still decodes 5 s of real audio; events past the next window's start are dropped
        # by MuScriptor's decoder and prelude forcing carries sustained notes across.
        self.boundaries = cfg.get("boundaries", "fixed")
        self.min_window = float(cfg.get("min_window_seconds", 3.5))
        self.describe = {"name": "muscriptor", "min_tail_seconds": self.min_tail,
                         **{k: cfg.get(k) for k in ("weights", "batch_size", "prelude_forcing", "dtype")},
                         **({"cfg_coef": self.cfg_coef} if self.cfg_coef != 1.0 else {}),
                         **({"logit_bias": self.logit_bias} if self.logit_bias else {}),
                         **({"boundaries": self.boundaries, "min_window_seconds": self.min_window}
                            if self.boundaries != "fixed" else {})}

    def _install_logit_bias(self):
        import torch

        tok = self.model._tokenizer
        bias = torch.zeros(tok.num_tokens, dtype=torch.float32)
        for i, e in enumerate(tok._vocab):
            t = getattr(e, "type", None)
            if t in self.logit_bias:
                bias[i] = self.logit_bias[t]
        if "eos" in self.logit_bias:
            bias[tok.eos_id] = self.logit_bias["eos"]
        lm = self.model._model
        original = lm._compute_logits

        def biased(*args, **kwargs):
            logits = original(*args, **kwargs)
            b = bias.to(logits.device)
            n = min(logits.shape[-1], b.shape[0])
            logits[:, :n] += b[:n]
            return logits

        lm._compute_logits = biased

    def __call__(self, wav: np.ndarray, sr: int, instruments: list[str] | None,
                 stem: str | None = None, window_instruments=None) -> list[Note]:
        """`window_instruments`: optional (window_seconds, [instruments per window]) from
        the tagger; each 5 s window then decodes under its own constraint (an empty window
        falls back to `instruments`)."""
        import torch
        from muscriptor.events import NoteEndEvent

        unknown = set(instruments or ()) - self.known
        if unknown:
            raise ValueError(f"MuScriptor does not know instruments {sorted(unknown)}")
        if window_instruments is not None:
            if float(window_instruments[0]) != 5.0:
                raise ValueError("window-level constraints need 5 s windows (MuScriptor's chunk)")
            events = self._transcribe_windows(wav, sr, instruments, per_window=window_instruments[1])
        elif self.boundaries == "fixed":
            chunk = int(5.0 * sr)
            tail = len(wav) % chunk
            if len(wav) > chunk and 0 < tail < self.min_tail * sr:
                wav = wav[: len(wav) - tail]
            events = self.model.transcribe((torch.from_numpy(np.ascontiguousarray(wav))[None], sr),
                                           instruments=instruments or None,
                                           batch_size=self.batch_size,
                                           prelude_forcing=self.prelude_forcing,
                                           cfg_coef=self.cfg_coef)
        else:
            events = self._transcribe_windows(wav, sr, instruments)
        duration = len(wav) / sr
        notes = []
        for ev in events:
            if isinstance(ev, NoteEndEvent) and ev.start_event.start_time < duration:
                s = ev.start_event
                notes.append(Note(s.start_time, min(ev.end_time, duration), s.pitch, s.instrument))
        return sorted(notes, key=lambda n: (n.start, n.pitch))

    def window_starts(self, wav16: np.ndarray, sr: int = 16000) -> list[float]:
        """Window start times (s) for `boundaries`: `fixed` every 5 s, `quiet` at the
        lowest onset-strength + RMS point (10 ms frames) in [prev + min_window, prev + 5]."""
        total = len(wav16) / sr
        if self.boundaries == "fixed":
            return [5.0 * i for i in range(max(1, int(np.ceil(total / 5.0 - 1e-9))))]
        import librosa
        hop = sr // 100
        env = librosa.onset.onset_strength(y=wav16, sr=sr, hop_length=hop)
        rms = librosa.feature.rms(y=wav16, frame_length=4 * hop, hop_length=hop)[0]
        n = min(len(env), len(rms))
        score = env[:n] / (env.max() + 1e-9) + rms[:n] / (rms.max() + 1e-9)
        score = np.convolve(score, np.ones(5) / 5, "same")
        starts = [0.0]
        while total - starts[-1] > 5.0:
            # the last window must reach the end: start it no earlier than total - 5 s
            last = total - starts[-1] <= 5.0 + self.min_window
            lo = int(np.ceil((total - 5.0 if last else starts[-1] + self.min_window) * 100))
            hi = min(int(round((starts[-1] + 5.0) * 100)), n - 1)
            starts.append((lo + int(np.argmin(score[lo:hi + 1]))) / 100 if hi >= lo else lo / 100)
        return starts

    def _transcribe_windows(self, wav: np.ndarray, sr: int, instruments: list[str] | None,
                            per_window: list[list[str]] | None = None):
        """MuScriptor's `transcribe` with window starts from `window_starts`, or, with
        `per_window`, fixed 5 s windows each decoded under its own instrument constraint."""
        import torch
        import torch.nn.functional as F
        from muscriptor.events import decode_model_tokens
        from muscriptor.tokenizer.mt3 import instrument_group_from_names

        m = self.model
        wav16 = m._load_wav(torch.from_numpy(np.ascontiguousarray(wav))[None], sr)
        seg = 5 * 16000
        if per_window is not None:
            starts = [5.0 * i for i in range(max(1, int(np.ceil(wav16.shape[-1] / seg - 1e-9))))]
        else:
            starts = self.window_starts(wav16[0].float().cpu().numpy(), 16000)
        chunks = []
        for t in starts:
            a = int(round(t * 16000))
            chunk = wav16[:, a: a + seg]
            if chunk.shape[-1] < seg:
                chunk = F.pad(chunk, (0, seg - chunk.shape[-1]))
            chunks.append(chunk)

        def constraint(inst):
            group = instrument_group_from_names(inst) if inst else None
            forb = (torch.tensor(m._tokenizer.forbidden_token_ids(inst), device=m._device,
                                 dtype=torch.long) if inst else None)
            return group, forb

        if per_window is None:
            group, forbidden = constraint(instruments)
            conditions = [m._build_conditions(c, group)[0] for c in chunks]
            batch = m._resolve_batch_size(self.batch_size, self.prelude_forcing)
            stream = m._generate_token_stream(conditions, starts, batch, 2000, False, 1.0, self.cfg_coef,
                                              True, self.prelude_forcing, 1, forbidden)
        else:
            stream = self._per_window_stream(chunks, starts, [
                (per_window[i] if i < len(per_window) and per_window[i] else instruments)
                for i in range(len(starts))], constraint)
        yield from decode_model_tokens(stream, m._tokenizer._vocab, m._instrument_for_program,
                                       frame_rate=m._tokenizer.frame_rate)

    def _per_window_stream(self, chunks, starts, insts, constraint):
        """MuScriptor's `_generate_token_stream` for batch 1, one constraint per chunk; the
        tie prologue is still teacher-forced from the previous chunk's open notes."""
        import torch
        from muscriptor.events import ChunkBoundary, OpenNoteTracker, ProgressEvent

        m = self.model
        tok = m._tokenizer
        tracker = OpenNoteTracker(tok._vocab, tok.frame_rate) if self.prelude_forcing else None
        n = len(starts)
        for c in range(n):
            bnd = ChunkBoundary(starts[c], starts[c + 1] if c + 1 < n else None)
            prompt = None
            if tracker is not None:
                tracker.feed(bnd)
                if c > 0:
                    prompt = torch.tensor([tok.tie_section_token_ids(tracker.open_keys())],
                                          device=m._device, dtype=torch.long)
            yield bnd
            group, forbidden = constraint(insts[c])
            cond = m._build_conditions(chunks[c], group)[0]
            for step in m._model.generate(prompt=prompt, conditions=[cond], max_gen_len=2000,
                                          use_sampling=False, temp=1.0, top_k=0, top_p=0.0,
                                          cfg_coef=self.cfg_coef, early_stop_on_token=tok.eos_id,
                                          beam_size=1, forbidden_tokens=forbidden):
                t = step.tolist()[0]
                if t == tok.eos_id:
                    break
                if tracker is not None:
                    tracker.feed(t)
                yield t
            yield ProgressEvent(completed=c + 1, total=n)


class PYIN:
    """Monophonic pYIN notes per stem: a rule-based baseline for comparison.

    A note is a run of pYIN-voiced frames above -45 dBFS whose rounded MIDI pitch is
    constant after a 5-frame median filter, lasting >= 0.12 s; the pitch search range
    depends on the stem. Ignores the instrument constraint; stems outside `stems`
    produce no notes.
    """

    SR, HOP = 22050, 256
    RANGES = {"bass": (35., 400.), "vocals": (75., 1100.), "guitar": (75., 1400.),
              "piano": (40., 2100.), "other": (60., 2100.), "mix": (35., 2100.)}

    def __init__(self, cfg: dict, device: str):
        self.stems = set(cfg.get("stems", ["bass", "vocals"]))
        self.max_seconds = cfg.get("max_seconds", 30.0)
        self.describe = {"name": "pyin", "stems": sorted(self.stems), "max_seconds": self.max_seconds}

    def __call__(self, wav: np.ndarray, sr: int, instruments: list[str] | None,
                 stem: str | None = None) -> list[Note]:
        import librosa
        from scipy.signal import medfilt

        from .audio import resample

        if stem not in self.stems:
            return []
        y = resample(wav, sr, self.SR)[: int(self.max_seconds * self.SR)]
        fmin, fmax = self.RANGES[stem]
        f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=self.SR, hop_length=self.HOP,
                                     frame_length=2048)
        rms = librosa.feature.rms(y=y, hop_length=self.HOP)[0][: len(f0)]
        ok = voiced & (20 * np.log10(rms + 1e-8) > -45)
        midi = np.where(ok, librosa.hz_to_midi(np.where(ok, f0, 1.0)), np.nan)
        rounded = np.where(ok, medfilt(np.nan_to_num(np.round(midi), nan=-1), 5), -1)
        t = librosa.frames_to_time(np.arange(len(f0)), sr=self.SR, hop_length=self.HOP)
        notes, i = [], 0
        while i < len(rounded):
            if rounded[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < len(rounded) and rounded[j + 1] == rounded[i]:
                j += 1
            start, end = float(t[i]), float(t[j] + self.HOP / self.SR)
            if end - start >= 0.12:
                notes.append(Note(start, end, int(rounded[i]), f"pyin:{stem}"))
            i = j + 1
        return notes


REGISTRY: dict[str, Callable] = {"muscriptor": MuScriptor, "pyin": PYIN}


def build_transcriber(cfg: dict, device: str):
    if cfg["name"] not in REGISTRY:
        raise KeyError(f"unknown transcriber {cfg['name']!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[cfg["name"]](cfg, device)
