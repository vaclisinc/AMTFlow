"""Model assets that are not on the Hugging Face Hub, fetched on first use.

Hub-hosted weights (MuScriptor, MATPAC++, the encoders, the ALMs) download through
`huggingface_hub` the first time they are loaded. Everything else is listed here with
a URL and SHA-256, downloaded once into `$AMTFLOW_CACHE` (default
`~/.cache/amtflow`) and verified; zip archives are extracted next to the download.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ASSETS = {
    # MATPAC++ authors' downstream linear probes (OpenMIC, MTG instrument, ...).
    "matpac-probes": {
        "url": "https://github.com/aurianworld/matpac/releases/download/Probes_weights/Weights_probes_matpac.zip",
        "sha256": "0bbdcae21cb7de61aa2a705303941c3b34cabf202256d22ea8827171cbdd97f2",
        "extract": True,
    },
    # BS-RoFormer SW (jarredou, fixed). Weights CC-BY-NC-SA 4.0.
    "bs-roformer-sw-fixed.ckpt": {
        "url": "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/a443a2985534b3bc815ef54a5d446c6a0390f974/BS-Rofo-SW-Fixed.ckpt",
        "sha256": "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
    },
    "bs-roformer-sw-fixed.yaml": {
        "url": "https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed/resolve/a443a2985534b3bc815ef54a5d446c6a0390f974/BS-Rofo-SW-Fixed.yaml",
        "sha256": "f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b",
    },
}


def cache_dir() -> Path:
    return Path(os.environ.get("AMTFLOW_CACHE", Path.home() / ".cache" / "amtflow"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str) -> Path:
    """Local path of asset `name` (a directory for extracted archives). Safe when several
    processes ask at once: one downloads under a file lock, the others wait for it."""
    import fcntl

    spec = ASSETS[name]
    root = cache_dir() / "assets"
    target, done = root / name, root / f"{name}.ok"
    if done.exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if done.exists():                       # another process finished meanwhile
            return target
        download = root / (f"{name}.zip" if spec.get("extract") else name)
        tmp = download.with_name(f"{download.name}.{os.getpid()}.part")
        print(f"[amtflow] downloading {name} from {spec['url']}", flush=True)
        try:
            with urllib.request.urlopen(spec["url"]) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if spec.get("sha256") and _sha256(tmp) != spec["sha256"]:
                raise RuntimeError(f"{name}: SHA-256 mismatch for {spec['url']}")
            tmp.replace(download)
        finally:
            tmp.unlink(missing_ok=True)
        if spec.get("extract"):
            with zipfile.ZipFile(download) as z:
                z.extractall(target)
            download.unlink()
        done.touch()
    return target


def resolve(value: str | None, asset: str) -> Path:
    """A config path, or the fetched asset when the value is `auto`/unset."""
    if value in (None, "auto"):
        return fetch(asset)
    return Path(value)
