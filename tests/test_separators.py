import json

import numpy as np
import soundfile as sf

from amtflow.separators import Precomputed


def test_precomputed_uses_actual_audio_for_silence(tmp_path):
    (tmp_path / "clips").mkdir()
    path = tmp_path / "other.wav"
    sf.write(path, np.zeros(16000, np.float32), 16000)
    (tmp_path / "clips" / "c1.json").write_text(json.dumps({
        "stem_presence": {"other": {"coverage_above_minus_40_percent": 100}},
        "stem_paths": {"other": str(path)},
    }))
    stem, = Precomputed({"root": tmp_path}, "cpu")(np.zeros(1), 16000, {"id": "c1"})
    assert stem.coverage == 0 and not stem.missing


def test_precomputed_distinguishes_missing_from_silent(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "stems").mkdir()
    sf.write(tmp_path / "stems" / "c1_bass.flac", np.full(22050, 0.1, np.float32), 22050)
    pres = lambda c: {"coverage_above_minus_40_percent": c}
    json.dump({"stem_presence": {"bass": pres(80.0), "other": pres(90.0), "piano": pres(1.0)},
               "stem_paths": {"bass": str(tmp_path / "stems" / "c1_bass.flac")}},
              open(tmp_path / "clips" / "c1.json", "w"))
    stems = {s.name: s for s in Precomputed({"root": tmp_path}, "cpu")(np.zeros(1), 44100, {"id": "c1"})}
    assert stems["bass"].sr == 22050 and len(stems["bass"].wav) == 22050 and not stems["bass"].missing
    assert stems["other"].missing                       # audible but not saved
    assert not stems["piano"].missing and len(stems["piano"].wav) == 0   # silent


def test_fetch_is_safe_under_concurrency(tmp_path, monkeypatch):
    from multiprocessing import Pool

    from amtflow import assets
    src = tmp_path / "blob.bin"
    src.write_bytes(b"x" * 1_000_000)
    monkeypatch.setenv("AMTFLOW_CACHE", str(tmp_path / "cache"))
    monkeypatch.setitem(assets.ASSETS, "blob", {"url": src.as_uri(), "sha256": None})
    with Pool(4) as pool:
        paths = pool.map(_fetch_blob, [str(src)] * 4)
    assert len(set(paths)) == 1 and (tmp_path / "cache" / "assets" / "blob").stat().st_size == 1_000_000


def _fetch_blob(src):
    from amtflow import assets
    assets.ASSETS["blob"] = {"url": "file://" + src, "sha256": None}
    return str(assets.fetch("blob"))
