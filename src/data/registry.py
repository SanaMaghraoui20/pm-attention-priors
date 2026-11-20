"""
Dataset registry
================
load_dataset(name, cfg, root) reads the entry `name` of the `datasets` section
of experiments/hyperparameters.yaml and returns the data dict shared by every
loader:

    {"sequences": {case_id: [item_id, ...]}, "item2id", "id2item",
     "n_items", "n_users", "source", ...}

"""

from __future__ import annotations

from pathlib import Path

from .cache import cached

LOADERS = {}


def _register(name):
    def deco(fn):
        LOADERS[name] = fn
        return fn
    return deco


@_register("dev1")
def _dev1(**kw):
    from .dev1 import load_dev1
    return load_dev1(verbose=False, **kw)


@_register("oulad")
def _oulad(**kw):
    from .oulad import load_oulad
    return load_oulad(verbose=False, **kw)


@_register("ednet")
def _ednet(**kw):
    from .ednet import load_ednet_kt3
    return load_ednet_kt3(verbose=False, **kw)


@_register("eventlog")
def _eventlog(**kw):
    from .eventlog import load_event_log
    return load_event_log(verbose=False, **kw)


PATH_ARGS = {"csv_path", "assets_path", "oulad_dir", "kt3_dir", "path"}


def load_dataset(name: str, cfg: dict, root: str | Path, use_cache: bool = True) -> dict:
    entry = cfg["datasets"][name]
    kw = dict(entry.get("args", {}))
    for k in PATH_ARGS & kw.keys():
        kw[k] = Path(root) / kw[k]
    fn = LOADERS[entry["loader"]]
    if not use_cache:
        return fn(**kw)
    return cached(f"{name}", fn, **kw)
