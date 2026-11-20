"""
EdNet-KT3 loader
================
One CSV file per student (u1.csv, u2.csv, ...), with timestamp and item_id.

  item     : content type given by the item_id prefix
             (b = bundle, q = question, e = explanation, l = lecture)
  order    : by timestamp within each student
  students : the first `max_users` files by numeric id (deterministic sample)

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"
PREFIX_MAP = {"q": "question", "e": "explanation", "l": "lecture", "b": "bundle"}


def _semantic(item_id) -> str:
    if not isinstance(item_id, str) or not item_id:
        return "unknown"
    return PREFIX_MAP.get(item_id[0].lower(), "unknown")


def load_ednet_kt3(kt3_dir: str | Path, max_users: int = 5000, min_seq_len: int = 5,
                   max_seq_len: int | None = 200, verbose: bool = True) -> dict:
    kt3_dir = Path(kt3_dir)
    assert kt3_dir.exists(), f"Directory not found: {kt3_dir}"
    files = sorted(kt3_dir.glob("u*.csv"), key=lambda f: int(f.stem[1:]))[:max_users]
    if not files:
        raise ValueError(f"No u*.csv files in {kt3_dir}")

    raw: dict[str, list[str]] = {}
    raw_t: dict[str, list[float]] = {}
    for fpath in files:
        try:
            df = pd.read_csv(fpath, usecols=["timestamp", "item_id"])
        except Exception:
            continue
        df = df.dropna(subset=["timestamp", "item_id"])
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        pairs = [(x, t / 1000.0) for x, t in                     # ms -> seconds
                 zip((_semantic(str(i)) for i in df["item_id"]), df["timestamp"])
                 if x != "unknown"]
        if len(pairs) >= min_seq_len:
            raw[fpath.stem] = [x for x, _ in pairs]
            raw_t[fpath.stem] = [t for _, t in pairs]
    if not raw:
        raise ValueError(f"No student left with min_seq_len={min_seq_len}")

    items = sorted({x for s in raw.values() for x in s})
    item2id = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    item2id.update({x: k + 2 for k, x in enumerate(items)})
    id2item = {v: k for k, v in item2id.items()}
    user2id = {u: k for k, u in enumerate(sorted(raw))}
    sequences, timestamps = {}, {}
    for u, s in raw.items():
        enc = [item2id[x] for x in s]
        sequences[user2id[u]] = enc[-max_seq_len:] if max_seq_len else enc
        timestamps[user2id[u]] = raw_t[u][-max_seq_len:] if max_seq_len else raw_t[u]

    if verbose:
        print(f"[ednet-kt3] students={len(raw)} activities={len(items)}")
    return {"sequences": sequences, "timestamps": timestamps,
            "item2id": item2id, "id2item": id2item,
            "user2id": user2id, "n_users": len(user2id), "n_items": len(item2id),
            "source": "EdNet-KT3", "max_seq_len": max_seq_len}