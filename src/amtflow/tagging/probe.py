"""Masked multi-label probes on pooled encoder features: every layer x lr, several seeds.

Follows the MARBLE encoder-probing protocol (MLP 512 / dropout 0.2 or a linear head; features z-scored per (layer, dim)
on train; Adam at constant lr; best-val epoch kept), adapted to partial labels:
unobserved (clip, class) entries are excluded from the loss and every metric.
Per seed the (layer, lr) with the best val mAP is selected and scored on test.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

LRS = (1e-4, 5e-4, 1e-3, 5e-3)


class BatchedHead(torch.nn.Module):
    """G independent heads; member g reads layer g of a (G, B, D) input."""

    def __init__(self, g: int, d: int, c: int, kind: str, hidden: int = 512):
        super().__init__()

        def init(fan_in, *shape):
            bound = fan_in ** -0.5
            return torch.nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

        self.kind = kind
        if kind == "mlp":
            self.w1, self.b1 = init(d, g, d, hidden), init(d, g, 1, hidden)
            self.w2, self.b2 = init(hidden, g, hidden, c), init(hidden, g, 1, c)
        elif kind == "linear":
            self.w2, self.b2 = init(d, g, d, c), init(d, g, 1, c)
        else:
            raise ValueError(f"unknown head {kind!r}")

    def forward(self, x):
        if self.kind == "mlp":
            x = F.dropout(F.relu(torch.baddbmm(self.b1, x, self.w1)), 0.2, self.training)
        return torch.baddbmm(self.b2, x, self.w2)             # (G, B, C)

    def member(self, g: int) -> dict:
        names = ("w1", "b1", "w2", "b2") if self.kind == "mlp" else ("w2", "b2")
        return {n: getattr(self, n)[g].detach().cpu() for n in names}


def masked_bce(logits, y, m):
    loss = F.binary_cross_entropy_with_logits(logits, y.expand_as(logits), reduction="none")
    return (loss * m).sum() / m.sum()


@torch.no_grad()
def average_precision(scores, y, m):
    """(..., C) AP per class over observed entries; scores (..., N, C), y/m (N, C).

    A class without an observed positive has no defined AP and is NaN; average with nanmean.
    """
    order = scores.argsort(dim=-2, descending=True)
    ys = torch.gather(y.expand_as(scores), -2, order)
    ms = torch.gather(m.expand_as(scores), -2, order)
    pos = ys * ms
    precision = pos.cumsum(-2) / ms.cumsum(-2).clamp_min(1)
    n_pos = pos.sum(-2)
    ap = (precision * pos).sum(-2) / n_pos.clamp_min(1)
    return torch.where(n_pos > 0, ap, torch.full_like(ap, float("nan")))


@torch.no_grad()
def f1_scores(logits, y, m):
    """(macro_f1 (...,), micro_f1 (...,), per-class f1 (..., C)) at probability 0.5."""
    p = (logits > 0).float() * m
    t = y * m
    tp, fp, fn = (p * t).sum(-2), (p * (1 - t)).sum(-2), ((1 - p) * t).sum(-2)
    per_class = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-12)
    micro = 2 * tp.sum(-1) / (2 * tp.sum(-1) + fp.sum(-1) + fn.sum(-1)).clamp_min(1e-12)
    return per_class.mean(-1), micro, per_class


def probe(X: torch.Tensor, splits: list[str], relevance: np.ndarray, mask: np.ndarray,
          head: str, seed: int, epochs: int = 50, batch: int = 64,
          device: str = "cuda", lrs: tuple[float, ...] = LRS) -> dict:
    """X (N, L, D) float16. Returns the selected probe's scores and weights."""
    L = X.shape[1]
    sel = {k: torch.tensor([s == k for s in splits]) for k in ("train", "val", "test")}
    Xtr = X[sel["train"]].float()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5                    # (L, D)
    del Xtr

    def prep(k):
        return ((X[sel[k]].float() - mu) / sd).transpose(0, 1).contiguous().to(device)

    x = {k: prep(k) for k in sel}
    y_all = torch.from_numpy((relevance >= 0.5).astype(np.float32))
    m_all = torch.from_numpy(mask.astype(np.float32))
    y = {k: y_all[sel[k]].to(device) for k in sel}
    m = {k: m_all[sel[k]].to(device) for k in sel}
    C = y_all.shape[1]

    torch.manual_seed(seed)
    models = [BatchedHead(L, X.shape[2], C, head).to(device) for _ in lrs]
    opts = [torch.optim.Adam(md.parameters(), lr=lr) for md, lr in zip(models, lrs)]
    best_val = torch.full((len(lrs), L), -1.0, device=device)
    best_epoch = torch.zeros(len(lrs), L, dtype=torch.long)
    best_state: dict[tuple[int, int], dict] = {}
    gen = torch.Generator().manual_seed(seed)
    ntr = x["train"].shape[1]

    def logits_of(md, k):
        md.eval()
        with torch.no_grad():
            out = torch.cat([md(x[k][:, i:i + 4096]) for i in range(0, x[k].shape[1], 4096)], 1)
        md.train()
        return out

    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(ntr, generator=gen).to(device)
        for i in range(0, ntr, batch):
            b = perm[i:i + batch]
            for md, o in zip(models, opts):
                o.zero_grad(set_to_none=True)
                masked_bce(md(x["train"][:, b]), y["train"][b], m["train"][b]).backward()
                o.step()
        for k, md in enumerate(models):
            v = average_precision(logits_of(md, "val"), y["val"], m["val"]).nanmean(-1)
            better = v > best_val[k]
            best_val[k] = torch.where(better, v, best_val[k])
            for layer in better.nonzero().flatten().tolist():
                best_epoch[k, layer] = epoch
                best_state[(k, layer)] = models[k].member(layer)
    k, layer = divmod(int(best_val.argmax()), L)

    # Re-score the selected member (and every layer's best-lr member) from saved weights.
    def score(state, lyr, split):
        md = BatchedHead(1, X.shape[2], C, head).to(device).eval()
        with torch.no_grad():
            for n, v in state.items():
                getattr(md, n).copy_(v[None].to(device))
            out = md(x[split][lyr:lyr + 1])[0]
        ap = average_precision(out, y[split], m[split])
        macro, micro, per_f1 = f1_scores(out, y[split], m[split])
        return {"map": float(ap.nanmean()), "macro_f1": float(macro), "micro_f1": float(micro),
                "ap": ap.tolist(), "f1": per_f1.tolist()}

    best_lr = best_val.argmax(0).tolist()
    chosen = best_state[(k, layer)]
    return {
        "seed": seed, "head": head, "layer": layer, "lr": lrs[k],
        "epoch": int(best_epoch[k, layer]), "val": score(chosen, layer, "val"),
        "test": score(chosen, layer, "test"),
        "layer_val_map": best_val.max(0).values.tolist(),
        "layer_test_map": [score(best_state[(best_lr[l], l)], l, "test")["map"] for l in range(L)],
        "seconds": round(time.time() - t0, 1),
        "_weights": {"state": chosen, "mu": mu[layer].clone(), "sd": sd[layer].clone(),
                     "layer": layer, "head": head},
    }
