"""Frozen music/audio encoders reduced to one contract: waveform -> (L, T, D) hidden states.

These are the wrappers whose features trained AMTFlow's released OpenMIC heads; keep
layer indexing, chunking and preprocessing unchanged so the heads apply exactly.
Every layer is the full stack (input embedding plus each transformer block). Clips
longer than an encoder's chunk are encoded in consecutive non-overlapping chunks.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

CHUNK_SECONDS = 30.0


def hub_path(repo: str, filename: str, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, revision=revision)


def load_prefixed(path: str, prefix: str) -> dict:
    """The tensors under `prefix` from one safetensors file, prefix stripped."""
    from safetensors import safe_open

    out = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(prefix):
                out[key[len(prefix):]] = f.get_tensor(key)
    return out


def load_sharded_prefixed(repo: str, prefix: str) -> dict:
    index = json.load(open(hub_path(repo, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
        out.update(load_prefixed(hub_path(repo, shard), prefix))
    return out


class HiddenStateEncoder:
    name: str
    repo: str
    sample_rate: int
    frame_rate: float
    n_layers: int
    dim: int
    chunk_seconds: float = CHUNK_SECONDS

    def load(self, device: str) -> "HiddenStateEncoder":
        raise NotImplementedError

    def _chunk(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def hidden_states(self, wav: np.ndarray) -> torch.Tensor:
        """(L, T, D) float16 on the device."""
        n = int(round(self.chunk_seconds * self.sample_rate))
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        parts = []
        # A tail shorter than two frame periods after a full chunk is dropped (some
        # front-ends cannot encode it; it holds at most a fraction of one frame).
        min_tail = int(2 * self.sample_rate / self.frame_rate)
        for start in range(0, max(len(x), 1), n):
            piece = x[start:start + n]
            if start and len(piece) < min_tail:
                break
            parts.append(self._chunk(piece))
        return torch.cat(parts, dim=1)

    def layer_features(self, wav: np.ndarray) -> torch.Tensor:
        """(L, D): every layer averaged over time."""
        return self.hidden_states(wav).float().mean(1)

    def describe(self) -> dict:
        return {"name": self.name, "repo": self.repo, "sample_rate": self.sample_rate,
                "frame_rate": self.frame_rate, "n_layers": self.n_layers, "dim": self.dim,
                "chunk_seconds": self.chunk_seconds, "pooling": "mean over time per layer"}


class Mert(HiddenStateEncoder):
    """m-a-p/MERT-v1-330M, 24 kHz, 75 Hz; hand-rolled layer loop (its remote code
    defeats transformers 5's hidden-state capture)."""
    name, repo = "mert", "m-a-p/MERT-v1-330M"
    sample_rate, frame_rate, n_layers, dim = 24000, 75.0, 25, 1024

    def load(self, device):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(self.repo, trust_remote_code=True).to(device).eval().float()
        self.device = device
        self.stable = bool(getattr(self.m.config, "do_stable_layer_norm", False))
        return self

    def _chunk(self, wav):
        m = self.m
        x = wav.to(self.device)[None]
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)       # MERT's processor normalization
        x = m.feature_extractor(x)
        x = m.feature_projection(x.transpose(1, 2))
        if isinstance(x, tuple):
            x = x[0]
        enc = m.encoder
        x = x + enc.pos_conv_embed(x)
        if not self.stable:
            x = enc.layer_norm(x)
        states = [x]
        for layer in enc.layers:
            out = layer(x, attention_mask=None)
            x = out[0] if isinstance(out, tuple) else out
            states.append(x)
        if self.stable:
            states[-1] = enc.layer_norm(states[-1])
        return torch.stack([s[0] for s in states]).half()


class MuQ(HiddenStateEncoder):
    """OpenMuQ/MuQ-large-msd-iter (weights CC-BY-NC 4.0), 24 kHz, 25 Hz; vendored MuQ code."""
    name, repo = "muq", "OpenMuQ/MuQ-large-msd-iter"
    sample_rate, frame_rate, n_layers, dim = 24000, 25.0, 13, 1024

    def load(self, device):
        from ..third_party.muq import MuQ as MuQModel
        self.m = MuQModel.from_pretrained(self.repo).to(device).eval().float()
        self.device = device
        return self

    def _chunk(self, wav):
        out = self.m(wav.to(self.device)[None], output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]).half()


class PupuM2D(HiddenStateEncoder):
    """spellbrush/PupuM2D large, 24 kHz mel, 4x16 patches -> 25 Hz x 8 bands; a frame is
    the 8 frequency patches of one time step concatenated (8192). EMA target encoder."""
    name, repo = "pupum2d", "spellbrush/PupuM2D"
    sample_rate, frame_rate, n_layers, dim = 24000, 25.0, 25, 8192
    chunk_seconds = 10.24
    SUB = "pupum2dV2_25hz_large"
    MEL_MEAN, MEL_STD = -4.089994845986366, 2.0242277159094813

    def load(self, device):
        import json5
        from easydict import EasyDict
        from huggingface_hub import snapshot_download
        from librosa.filters import mel as mel_fn
        from safetensors.torch import load_file

        from ..third_party.pupum2d.pupujepa import PupuM2D as Model

        root = Path(snapshot_download(self.repo, allow_patterns=[f"{self.SUB}/*"])) / self.SUB
        cfg = EasyDict(json5.loads((root / "args.json").read_text()))
        cfg.train.grad_checkpointing_step = None
        self.cfg = cfg
        model = Model(cfg)
        ckpt = sorted((root / "checkpoint").glob("*/model.safetensors"))
        if len(ckpt) != 1:
            raise RuntimeError(f"expected one PupuM2D checkpoint, found {ckpt}")
        state = {k.removeprefix("module."): v for k, v in load_file(ckpt[0]).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # args.json says qk_norm=true but the checkpoint has no q/k norm parameters, and
        # the released weights reproduce their training loss only without QK-Norm.
        other = [k for k in missing if not k.endswith(("q_norm.weight", "q_norm.bias",
                                                        "k_norm.weight", "k_norm.bias"))]
        if other or unexpected:
            raise RuntimeError(f"PupuM2D: missing {other[:5]} unexpected {unexpected[:5]}")
        for module in model.modules():
            if hasattr(module, "q_norm"):
                module.q_norm = torch.nn.Identity()
                module.k_norm = torch.nn.Identity()
        self.m = model.to(device).eval().float()
        self.device = device
        p = cfg.preprocess
        self.mel_basis = torch.from_numpy(mel_fn(sr=p.sample_rate, n_fft=p.n_fft, n_mels=p.n_mels,
                                                 fmin=p.fmin, fmax=p.fmax)).float().to(device)
        self.window = torch.hann_window(p.win_size).to(device)
        return self

    def _mel(self, y):
        p = self.cfg.preprocess
        pad = int((p.n_fft - p.hop_size) / 2)
        y = torch.nn.functional.pad(y[None, None], (pad, pad), mode="reflect")[0]
        spec = torch.stft(y, p.n_fft, hop_length=p.hop_size, win_length=p.win_size,
                          window=self.window, center=False, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
        spec = torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))
        spec = (spec - self.MEL_MEAN) / (self.MEL_STD + 1e-8)
        return spec.transpose(-1, -2)[:, None]

    def _chunk(self, wav):
        mel = self._mel(wav.to(self.device))
        t_patch = self.cfg.model.patch_size[0]
        n_time = mel.shape[2] // t_patch
        mel = mel[:, :, : n_time * t_patch]
        grid = (n_time, mel.shape[3] // self.cfg.model.patch_size[1])
        x = self.m.patch_embed(mel)
        rope = self.m.rope_encoder.get_embed(list(grid))[None, None]
        enc = self.m.teacher
        states = [x]
        for blk in enc.blocks:
            x = blk(x, rope=rope)
            states.append(x)
        states[-1] = enc.norm(states[-1])
        h = torch.stack([s[0] for s in states])
        return h.reshape(h.shape[0], grid[0], grid[1] * h.shape[-1]).half()


class AFWhisper(HiddenStateEncoder):
    """The AF-Whisper audio tower of an Audio Flamingo checkpoint, 16 kHz, 50 Hz, read
    before the final 2x pool; inputs padded to 30 s with the real-frame mask."""
    sample_rate, frame_rate, n_layers, dim = 16000, 50.0, 33, 1280

    def __init__(self, name, repo, sharded):
        self.name, self.repo, self.sharded = name, repo, sharded

    def load(self, device):
        from transformers import WhisperFeatureExtractor
        from transformers.models.audioflamingo3.configuration_audioflamingo3 import (
            AudioFlamingo3EncoderConfig,
        )
        from transformers.models.audioflamingo3.modeling_audioflamingo3 import (
            AudioFlamingo3Encoder,
        )

        raw = json.load(open(hub_path(self.repo, "config.json")))
        audio_cfg = AudioFlamingo3EncoderConfig(**{k: v for k, v in raw["audio_config"].items()
                                                   if k != "model_type"})
        self.m = AudioFlamingo3Encoder(audio_cfg)

        def load(prefix):
            return (load_sharded_prefixed(self.repo, prefix) if self.sharded
                    else load_prefixed(hub_path(self.repo, "model.safetensors"), prefix))

        state = load("audio_tower.") or load("model.audio_tower.")
        missing, unexpected = self.m.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"{self.repo} audio tower: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.m = self.m.to(device).eval().float()
        self.fe = WhisperFeatureExtractor(feature_size=audio_cfg.num_mel_bins, sampling_rate=16000)
        self.device = device
        self.captured = []
        for layer in self.m.layers:
            layer.register_forward_hook(
                lambda _m, _i, o: self.captured.append(o[0] if isinstance(o, tuple) else o))
        self.m.layers[0].register_forward_pre_hook(lambda _m, args: self.captured.append(args[0]))
        return self

    def _chunk(self, wav):
        out = self.fe(wav.numpy(), sampling_rate=16000, return_tensors="pt",
                      padding="max_length", return_attention_mask=True)
        self.captured = []
        self.m(out.input_features.to(self.device), input_features_mask=out.attention_mask.to(self.device))
        n = math.ceil(len(wav) / 160 / 2)
        return torch.stack([h[0, :n] for h in self.captured]).half()


class QwenAuT(HiddenStateEncoder):
    """Qwen3-Omni-30B-A3B-Instruct's AuT audio encoder, 16 kHz mel, ~13 tokens/s; 32
    layers read with hooks, layer 0 the conv front-end output."""
    name, repo = "qwen3-omni-aut", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    sample_rate, frame_rate, n_layers, dim = 16000, 13.0, 33, 1280

    def load(self, device):
        from transformers import WhisperFeatureExtractor
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        raw = json.load(open(hub_path(self.repo, "config.json")))
        cfg = Qwen3OmniMoeAudioEncoderConfig(**{
            k: v for k, v in raw["thinker_config"]["audio_config"].items()
            if k in Qwen3OmniMoeAudioEncoderConfig().to_dict() and k != "model_type"})
        cfg._attn_implementation = "sdpa"
        self.m = Qwen3OmniMoeAudioEncoder(cfg)
        missing, unexpected = self.m.load_state_dict(
            load_sharded_prefixed(self.repo, "thinker.audio_tower."), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"AuT: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.m = self.m.to(device).eval().float()
        pre = json.load(open(hub_path(self.repo, "preprocessor_config.json")))
        self.fe = WhisperFeatureExtractor(feature_size=pre.get("feature_size", 128), sampling_rate=16000,
                                          hop_length=pre.get("hop_length", 160), n_fft=pre.get("n_fft", 400))
        self.device = device
        self.captured = []
        for layer in self.m.layers:
            layer.register_forward_hook(
                lambda _m, _i, o: self.captured.append(o[0] if isinstance(o, tuple) else o))
        self.m.layers[0].register_forward_pre_hook(lambda _m, args: self.captured.append(args[0]))
        return self

    def _chunk(self, wav):
        out = self.fe(wav.numpy(), sampling_rate=16000, return_tensors="pt",
                      padding=False, return_attention_mask=True)
        feats = out.input_features[0].to(self.device)
        self.captured = []
        self.m(feats, feature_lens=torch.tensor([feats.shape[1]], device=self.device))
        return torch.stack(self.captured).half()


ENCODERS = {
    "mert": Mert,
    "muq": MuQ,
    "pupum2d": PupuM2D,
    "af3-whisper": lambda: AFWhisper("af3-whisper", "nvidia/audio-flamingo-3-hf", True),
    "afnext-whisper": lambda: AFWhisper("afnext-whisper", "nvidia/audio-flamingo-next-hf", False),
    "qwen3-omni-aut": QwenAuT,
}
