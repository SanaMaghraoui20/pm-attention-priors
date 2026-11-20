"""
DEV1 loader (Bloom-only export of Moodle2EventLog, three granularities)
=======================================================================
Source: the Bloom-only export of the Moodle2EventLog enrichment tool, which
keeps the semantic learning activities of Bloom's revised taxonomy and drops
navigation, feedback and interaction events.

Three granularities, all anchored on the Bloom activity
-------------------------------------------------------
  - "bloom"    : item = SemanticActivity                      
                 "what kind of learning act comes next?"
  - "section"  : item = SemanticActivity @ course section     
                 "what kind of act, at which point in the course?"
  - "resource" : item = SemanticActivity @ resource name      
                 "what kind of act, on which specific resource?"

The activity is always the anchor; only the complement changes. The three
datasets therefore differ by granularity alone.

Pedagogical filtering
---------------------
moodleAssets.csv restricts events to modules that belong to the course's
teaching path:
  - sections 3-7 and 9-15 (ten weekly sessions plus two assessments),
  - types fichier, test, devoir, dossier.
This excludes optional and administrative material (holiday exercises,
consent forms, external reference links).

"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_COL = "MoodleUserId"
TIME_COL = "Time"
ACTIVITY_COL = "SemanticActivity"
DESCRIPTION_COL = "Description"
CONTEXT_COL = "Event context"

PAD_TOKEN = "<PAD>"
MASK_TOKEN = "<MASK>"

# Course sections that belong to the teaching path
PEDAGOGICAL_SECTIONS = {3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}
PEDAGOGICAL_TYPES = {"fichier", "test", "devoir", "dossier"}

# The six Bloom learning activities the enrichment tool produces.
# Anything else means the source file is not the Bloom-only export.
BLOOM_ACTIVITIES = {
    "Study_P", "Study_A",
    "Exercise_P", "Exercise_A",
    "Assess_P", "Assess_A",
    "Synthesize",
}

ITEM_LEVELS = ("bloom", "section", "resource")

_CMID_REGEX = re.compile(r"course module id ['\"](\d+)['\"]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_lower(s: str) -> str:
    import unicodedata
    s = s.strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def load_pedagogical_modules(assets_path: str | Path) -> dict[int, dict]:
    """Parse moodleAssets.csv -> {cmid: {section, type, name}} for modules
    that belong to a pedagogical section and have a pedagogical type."""
    assets_path = Path(assets_path)
    assert assets_path.exists(), f"File not found: {assets_path}"

    modules: dict[int, dict] = {}
    with open(assets_path, "r", encoding="utf-8") as f:
        in_module_section = False
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if not parts or all(p == "" for p in parts):
                continue
            header = parts[0].lower()
            if header in ("numero de la section", "numéro de la section"):
                if len(parts) > 2 and "module" in parts[2].lower():
                    in_module_section = True
                continue
            if not in_module_section or len(parts) < 4:
                continue
            try:
                section_id = int(parts[0])
                module_id = int(parts[1])
            except ValueError:
                break
            module_type = _strip_lower(parts[2])
            if section_id in PEDAGOGICAL_SECTIONS and module_type in PEDAGOGICAL_TYPES:
                modules[module_id] = {"section": section_id,
                                      "type": module_type,
                                      "name": parts[3]}
    return modules


def extract_cmid(description: str) -> int | None:
    """Pull the Moodle course module id out of an event description."""
    if not isinstance(description, str):
        return None
    m = _CMID_REGEX.search(description)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Item construction
# ---------------------------------------------------------------------------

def _build_items(df: pd.DataFrame, item_level: str) -> pd.Series:
    """Return the item label of every event, for the requested granularity.
    The Bloom activity is always the anchor."""
    activity = df[ACTIVITY_COL].astype(str).str.strip()

    if item_level == "bloom":
        return activity
    if item_level == "section":
        return activity + " @ S" + df["_section"].astype(int).astype(str)
    if item_level == "resource":
        return activity + " @ " + df[CONTEXT_COL].astype(str).str.strip()
    raise ValueError(f"item_level must be one of {ITEM_LEVELS}, got '{item_level}'")


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dev1(
    csv_path: str | Path,
    item_level: Literal["bloom", "section", "resource"] = "bloom",
    min_seq_len: int = 5,
    max_seq_len: int | None = None,
    assets_path: str | Path | None = None,
    filter_pedagogical: bool = True,
    verbose: bool = True,
) -> dict:

    if item_level not in ITEM_LEVELS:
        raise ValueError(f"item_level must be one of {ITEM_LEVELS}, got '{item_level}'")

    csv_path = Path(csv_path)
    assert csv_path.exists(), f"File not found: {csv_path}"
    if assets_path is None:
        assets_path = csv_path.parent / "moodleAssets.csv"

    # ---- 1. read -----------------------------------------------------------
    df = pd.read_csv(csv_path, sep=None, engine="python")
    n_raw = len(df)
    if verbose:
        print(f"Loaded {n_raw} events from {csv_path.name}")

    # ---- 2. sanity check: is this really the Bloom-only export? ------------
    found = set(df[ACTIVITY_COL].dropna().unique())
    unexpected = found - BLOOM_ACTIVITIES
    if unexpected:
        warnings.warn(
            f"{len(unexpected)} non-Bloom activity labels found: "
            f"{sorted(unexpected)[:6]}. This looks like the full enriched log "
            f"rather than the Bloom-only export.",
            RuntimeWarning,
        )
    if verbose:
        print(f"Bloom activities present: {sorted(found & BLOOM_ACTIVITIES)}")

    # ---- 3. resource id ----------------------------------------------------
    df["_cmid"] = df[DESCRIPTION_COL].apply(extract_cmid)
    n_with_cmid = df["_cmid"].notna().sum()
    if verbose:
        print(f"Events with a resource id: {n_with_cmid} / {n_raw} "
              f"({100 * n_with_cmid / n_raw:.1f}%)")

    # ---- 4. pedagogical filter, and the metadata it carries ---------------
    if filter_pedagogical:
        peda = load_pedagogical_modules(assets_path)
        before = len(df)
        df = df[df["_cmid"].isin(peda.keys())].copy()
        # here we finally use the VALUES of moodleAssets, not only its keys
        df["_section"] = df["_cmid"].map(lambda c: peda[c]["section"])
        df["_mtype"] = df["_cmid"].map(lambda c: peda[c]["type"])
        if verbose:
            print(f"Pedagogical filter: {len(df)} / {before} events kept "
                  f"({100 * len(df) / before:.1f}%), "
                  f"{df['_cmid'].nunique()} resources, "
                  f"{df['_section'].nunique()} sections")
    elif item_level == "section":
        raise ValueError("item_level='section' needs the section metadata; "
                         "set filter_pedagogical=True")

    # ---- 5. build the item label ------------------------------------------
    df["_item"] = _build_items(df, item_level)

    # ---- 6. clean and order -----------------------------------------------
    df = df[[USER_COL, TIME_COL, "_item"]].rename(
        columns={USER_COL: "user", TIME_COL: "timestamp", "_item": "item"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user", "timestamp", "item"])
    df["item"] = df["item"].astype(str).str.strip()
    df = df[(df["item"] != "") & (df["item"] != "<NA>") & (df["item"] != "nan")]
    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    # ---- 7. per-learner sequences -----------------------------------------
    raw_sequences = df.groupby("user")["item"].apply(list).to_dict()
    raw_times = (df["timestamp"].astype("int64") / 1e9).groupby(df["user"]).apply(list).to_dict()
    raw_sequences = {u: s for u, s in raw_sequences.items() if len(s) >= min_seq_len}
    if not raw_sequences:
        raise ValueError(f"No learner left with min_seq_len={min_seq_len}")

    if verbose:
        lens = [len(s) for s in raw_sequences.values()]
        print(f"Learners with >= {min_seq_len} events: {len(raw_sequences)}")
        print(f"Sequence length before truncation: min={min(lens)} "
              f"median={sorted(lens)[len(lens)//2]} max={max(lens)} "
              f"mean={sum(lens)/len(lens):.1f}")

    # ---- 8. vocabulary (0 = PAD, 1 = MASK) --------------------------------
    all_items = sorted({i for s in raw_sequences.values() for i in s})
    item2id = {PAD_TOKEN: 0, MASK_TOKEN: 1}
    item2id.update({label: idx + 2 for idx, label in enumerate(all_items)})
    id2item = {v: k for k, v in item2id.items()}
    user2id = {u: idx for idx, u in enumerate(sorted(raw_sequences))}

    # ---- 9. encode and truncate -------------------------------------------
    sequences: dict[int, list[int]] = {}
    timestamps: dict[int, list[float]] = {}
    for user, seq in raw_sequences.items():
        encoded = [item2id[i] for i in seq]
        sequences[user2id[user]] = encoded[-max_seq_len:] if max_seq_len else encoded
        ts = raw_times[user]
        timestamps[user2id[user]] = ts[-max_seq_len:] if max_seq_len else ts

    if verbose:
        variants = len({tuple(s) for s in sequences.values()})
        print(f"Vocabulary: {len(item2id)} (incl. PAD and MASK) | "
              f"variant ratio: {variants / len(sequences):.3f}")

    return {
        "user2id": user2id,
        "item2id": item2id,
        "id2item": id2item,
        "sequences": sequences,
        "timestamps": timestamps,           # seconds, aligned with sequences
        "n_users": len(user2id),
        "n_items": len(item2id),
        "item_level": item_level,
        "max_seq_len": max_seq_len,
        "filtered": filter_pedagogical,
        "source": csv_path.name,
    }