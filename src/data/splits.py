"""
Case-level folds
================
Whole cases (learners) are assigned to folds: a test case is never seen in
training, and every one of its events can be predicted.

"""

from __future__ import annotations

import random


def subset(sequences: dict, ids: list) -> dict:
    return {u: sequences[u] for u in ids}


def benchmark_split(
    sequences: dict,
    n_folds: int = 5,
    fold: int = 0,
    val_frac: float = 0.2,
    train_frac: float = 1.0,
    seed: int = 42,
) -> dict:
    from sklearn.model_selection import KFold

    assert 0 <= fold < n_folds and 0.0 < train_frac <= 1.0
    ids = sorted(sequences)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, test_idx = list(kf.split(ids))[fold]
    if 0 < val_frac < 1:
        cut = round(len(train_idx) * (1 - val_frac))
        val_idx, train_idx = train_idx[cut:], train_idx[:cut]
    else:
        val_idx = []
    train = [ids[i] for i in train_idx]
    if train_frac < 1.0:
        perm = train[:]
        random.Random(seed * 7919 + fold + 1).shuffle(perm)
        train = perm[:max(1, round(train_frac * len(perm)))]
    out = {"train": sorted(train), "val": sorted(ids[i] for i in val_idx),
           "test": sorted(ids[i] for i in test_idx)}
    assert not (set(out["train"]) & set(out["val"])) and \
        not (set(out["train"]) & set(out["test"])) and \
        not (set(out["val"]) & set(out["test"])), "case split overlaps"
    return out
