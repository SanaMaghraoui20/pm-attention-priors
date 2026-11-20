"""
On-disk cache for loaded datasets
=================================
Large logs (OULAD: 453 MB CSV, EdNet: 5,000 files) take minutes and over 2 GB
of memory to parse, and every job of a campaign parses them again. The first
job saves the loaded data dict to cache/<key>.pkl; the following jobs read it
in seconds.

The key holds the dataset name and every loader argument, so a change of
arguments uses a new file. After changing a loader's CODE, delete the cache
folder (rm -rf cache/) so that the data are parsed again.

Writes go to a temporary file renamed atomically, so concurrent jobs never read
a half-written cache.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
# Bump when the content of the loaded data dict changes (2: timestamps).
CACHE_VERSION = 2


def cached(cache_name: str, loader, **kwargs) -> dict:
    """Return loader(**kwargs), reading or writing cache/<name>-<hash>.pkl."""
    sig = repr(sorted((k, str(v)) for k, v in kwargs.items()))
    key = f"{cache_name}-v{CACHE_VERSION}-{hashlib.sha1(sig.encode()).hexdigest()[:10]}"
    path = CACHE_DIR / f"{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    data = loader(**kwargs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return data