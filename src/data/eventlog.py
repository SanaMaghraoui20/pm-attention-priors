"""
Generic event-log loader for public PPM benchmarks
======================================================================
Turns a standard event log (Helpdesk, Sepsis, BPIC 2013/2020) into the
data dict shared by every loader of this repository:

    {"sequences": {case_id(int): [item_id, ...]}, "item2id", "id2item",
     "user2id", "n_users", "n_items", "source", ...}

Events are ordered by timestamp within each case (stable sort, so events that
share a timestamp keep their file order).
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd

PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"


def _seconds(t) -> float:
    """POSIX seconds of a pandas Timestamp; naive timestamps are read as UTC."""
    return (t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")).timestamp()


def _iter_xes(path: Path):
    """Stream (case_id, activity, timestamp, lifecycle) from an XES file."""
    from xml.etree.ElementTree import iterparse

    def local(tag):
        return tag.rsplit("}", 1)[-1] if "}" in tag else tag

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        case_id, event, depth_event = None, None, 0
        for evt, elem in iterparse(fh, events=("start", "end")):
            name = local(elem.tag)
            if evt == "start":
                if name == "trace":
                    case_id = None
                elif name == "event":
                    event = {}
                continue
            if name in ("string", "date", "int", "float", "boolean", "id"):
                k, v = elem.attrib.get("key", ""), elem.attrib.get("value", "")
                if event is not None:
                    event[k] = v
                elif k == "concept:name":
                    case_id = v
            elif name == "event":
                if event is not None and case_id is not None:
                    act = event.get("concept:name", "").strip()
                    ts = event.get("time:timestamp", "").strip()
                    if act and ts:
                        yield case_id, act, ts, event.get("lifecycle:transition", "").strip()
                event = None
            elif name == "trace":
                case_id = None
            elem.clear()


def _read_csv(path: Path, case_col, activity_col, time_col, lifecycle_col):
    df = pd.read_csv(path, sep=None, engine="python")
    for c in (case_col, activity_col, time_col):
        assert c in df.columns, f"column {c!r} not in {list(df.columns)}"
    lc = df[lifecycle_col] if lifecycle_col and lifecycle_col in df.columns else ""
    return zip(df[case_col].astype(str), df[activity_col].astype(str).str.strip(),
               df[time_col].astype(str), (lc.astype(str) if isinstance(lc, pd.Series)
                                          else [""] * len(df)))


def load_event_log(
    path: str | Path,
    name: str | None = None,
    min_seq_len: int = 2,
    max_seq_len: int | None = None,
    include_lifecycle: bool | None = None,
    keep_lifecycle: set[str] | None = None,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    time_col: str = "time:timestamp",
    lifecycle_col: str = "lifecycle:transition",
    verbose: bool = True,
) -> dict:
    """
    Parameters
    ----------
    path              : .xes, .xes.gz or .csv event log.
    min_seq_len       : drop cases with fewer events (2 = at least one prefix).
    max_seq_len       : keep the last N events of each case.
    keep_lifecycle    : keep only events whose transition is in this set
                        (case-insensitive), e.g. {"complete"}.
    """
    path = Path(path)
    assert path.exists(), f"File not found: {path}"
    if path.suffix == ".csv":
        rows = _read_csv(path, case_col, activity_col, time_col, lifecycle_col)
    else:
        rows = _iter_xes(path)

    keep = {k.lower() for k in keep_lifecycle} if keep_lifecycle else None
    events: dict[str, list] = defaultdict(list)
    lifecycles = set()
    n_events = 0
    for i, (case, act, ts, lc) in enumerate(rows):
        lc = lc if lc not in ("nan", "None") else ""
        if keep is not None and lc.lower() not in keep:
            continue
        lifecycles.add(lc.lower())
        events[case].append((pd.Timestamp(ts), i, act, lc))
        n_events += 1

    use_lc = (len(lifecycles - {""}) > 1) if include_lifecycle is None else include_lifecycle

    raw: dict[str, list[str]] = {}
    raw_t: dict[str, list[float]] = {}
    for case, evs in events.items():
        evs.sort(key=lambda e: (e[0], e[1]))      # timestamp, then file order
        seq = [f"{a}+{l}" if use_lc and l else a for _, _, a, l in evs]
        if len(seq) >= min_seq_len:
            raw[case] = seq
            raw_t[case] = [_seconds(t) for t, _, _, _ in evs]
    if not raw:
        raise ValueError(f"No case left with min_seq_len={min_seq_len}")

    all_items = sorted({x for s in raw.values() for x in s})
    item2id = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    item2id.update({x: k + 2 for k, x in enumerate(all_items)})
    id2item = {v: k for k, v in item2id.items()}
    user2id = {c: k for k, c in enumerate(sorted(raw))}
    sequences, timestamps = {}, {}
    for case, seq in raw.items():
        enc = [item2id[x] for x in seq]
        sequences[user2id[case]] = enc[-max_seq_len:] if max_seq_len else enc
        ts = raw_t[case]
        timestamps[user2id[case]] = ts[-max_seq_len:] if max_seq_len else ts

    lens = sorted(len(s) for s in raw.values())
    stats = {"events": sum(lens), "events_in_file": n_events, "cases": len(raw), "activities": len(all_items),
             "len_min": lens[0], "len_median": lens[len(lens) // 2],
             "len_max": lens[-1], "len_mean": sum(lens) / len(lens),
             "variant_ratio": len({tuple(s) for s in raw.values()}) / len(raw),
             "lifecycle_in_item": use_lc}
    if verbose:
        print(f"[{name or path.name}] " + ", ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))

    return {"sequences": sequences, "timestamps": timestamps, "item2id": item2id, "id2item": id2item,
            "user2id": user2id, "n_users": len(user2id), "n_items": len(item2id),
            "source": path.name, "name": name or path.stem, "stats": stats,
            "max_seq_len": max_seq_len}