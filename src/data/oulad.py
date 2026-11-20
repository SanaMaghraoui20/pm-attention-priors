"""
OULAD loader (module FFF)
=========================
Open University Learning Analytics Dataset, module FFF.

  item     : activity_type of the VLE resource (resource, oucontent, ...)
  order    : by student, then by `date` (whole days relative to the course
             start); clicks of the same day keep the file order
  clicks   : a row with sum_click = n becomes min(n, 5) identical events
  case     : one sequence per student; students enrolled in several FFF
             presentations keep a single case (presentations interleaved)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"


def load_oulad(oulad_dir: str | Path, module: str = "FFF", min_seq_len: int = 5,
               max_seq_len: int | None = 200, click_cap: int = 5,
               verbose: bool = True) -> dict:
    oulad_dir = Path(oulad_dir)
    vle_path, svle_path = oulad_dir / "vle.csv", oulad_dir / "studentVle.csv"
    assert vle_path.exists() and svle_path.exists(), f"OULAD files not found in {oulad_dir}"

    vle = pd.read_csv(vle_path, usecols=["id_site", "activity_type"]).dropna()
    vle["activity_type"] = vle["activity_type"].astype(str).str.strip()
    site2activity = dict(zip(vle["id_site"], vle["activity_type"]))

    df = pd.read_csv(svle_path, usecols=["id_student", "id_site", "date", "sum_click", "code_module"])
    df = df[df["code_module"] == module].copy()
    df["activity_type"] = df["id_site"].map(site2activity)
    df = df.dropna(subset=["activity_type", "date"])
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["sum_click"] = df["sum_click"].clip(upper=click_cap).astype(int)
    df = df.loc[df.index.repeat(df["sum_click"])].reset_index(drop=True)
    df = df.sort_values(["id_student", "date"]).reset_index(drop=True)

    raw = {int(u): list(g) for u, g in df.groupby("id_student", sort=True)["activity_type"]}
    raw_t = {int(u): [float(d) * 86400.0 for d in g]          # whole days -> seconds
             for u, g in df.groupby("id_student", sort=True)["date"]}
    raw = {u: s for u, s in raw.items() if len(s) >= min_seq_len}
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
        print(f"[oulad-{module}] students={len(raw)} activities={len(items)} "
              f"events={sum(len(s) for s in raw.values())}")
    return {"sequences": sequences, "timestamps": timestamps,
            "item2id": item2id, "id2item": id2item,
            "user2id": user2id, "n_users": len(user2id), "n_items": len(item2id),
            "source": f"OULAD/{module}", "max_seq_len": max_seq_len}