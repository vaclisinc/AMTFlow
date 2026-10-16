import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from amtflow.tagging.probe import average_precision, f1_scores, probe


def _data(n=400, c=5, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, c)) > 0.6).astype(np.float32)
    m = (rng.random((n, c)) > 0.3).astype(np.float32)
    s = y * 1.5 + rng.normal(size=(n, c)).astype(np.float32)
    return s, y, m


def test_masked_ap_matches_sklearn():
    s, y, m = _data()
    ours = average_precision(torch.from_numpy(s), torch.from_numpy(y), torch.from_numpy(m))
    ref = [average_precision_score(y[m[:, j] > 0, j], s[m[:, j] > 0, j]) for j in range(y.shape[1])]
    np.testing.assert_allclose(ours.numpy(), ref, atol=1e-5)


def test_masked_f1_matches_sklearn():
    s, y, m = _data(seed=1)
    macro, micro, per = f1_scores(torch.from_numpy(s), torch.from_numpy(y), torch.from_numpy(m))
    ref = [f1_score(y[m[:, j] > 0, j], s[m[:, j] > 0, j] > 0) for j in range(y.shape[1])]
    np.testing.assert_allclose(per.numpy(), ref, atol=1e-5)
    assert abs(float(macro) - np.mean(ref)) < 1e-5


def test_probe_learns_informative_layer():
    rng = np.random.default_rng(0)
    n, c, d = 600, 4, 16
    y = (rng.random((n, c)) > 0.5).astype(np.float32)
    noise = rng.normal(size=(n, 2, d)).astype(np.float32)
    noise[:, 1, :c] += 3 * y                           # layer 1 carries the labels
    splits = ["train"] * 400 + ["val"] * 100 + ["test"] * 100
    res = probe(torch.from_numpy(noise).half(), splits, y, np.ones_like(y, dtype=bool),
                "linear", seed=0, epochs=30, device="cpu")
    assert res["layer"] == 1 and res["test"]["map"] > 0.9


def test_ap_is_nan_without_positives():
    s, y, m = _data(seed=2)
    y[:, 0] = 0
    ap = average_precision(torch.from_numpy(s), torch.from_numpy(y), torch.from_numpy(m))
    assert torch.isnan(ap[0]) and not torch.isnan(ap[1:]).any()
