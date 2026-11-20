"""
Training windows and prefix evaluation samples
==============================================
Training uses EVERY event of the training cases: each sequence is cut into
windows of the model's context length.

Evaluation predicts everyevent of the held-out cases from its prefix: one
sample per (case, position).
"""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_ID = 0
MASK_ID = 1


def _windows(seq: list, size: int, stride: int) -> list:
    """Windows of length `size` covering `seq`; the last one ends at the end."""
    if len(seq) <= size:
        return [seq]
    starts = list(range(0, len(seq) - size + 1, stride))
    if starts[-1] != len(seq) - size:
        starts.append(len(seq) - size)
    return [seq[s:s + size] for s in starts]


def _left_pad(x: list, length: int) -> list:
    return [PAD_ID] * (length - len(x)) + x


class CausalWindowDataset(Dataset):


    def __init__(self, sequences: dict, max_len: int = 200):
        self.max_len = max_len
        self.samples = []
        for uid in sorted(sequences):
            s = sequences[uid]
            if len(s) < 2:
                continue
            for w in _windows(s, max_len + 1, max_len):
                if len(w) >= 2:
                    self.samples.append((uid, w))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, w = self.samples[idx]
        return {
            "user_id": uid,
            "input":  torch.tensor(_left_pad(w[:-1], self.max_len), dtype=torch.long),
            "target": torch.tensor(_left_pad(w[1:], self.max_len), dtype=torch.long),
        }


class RejectWindowDataset(CausalWindowDataset):

    def __init__(self, sequences: dict, max_len: int, constraints: dict, n_items: int):
        from pmpriors.declare import reject_matrix
        super().__init__(sequences, max_len)
        self.n_items = n_items
        self.packed = {uid: np.packbits(reject_matrix(sequences[uid], constraints, n_items), axis=1)
                       for uid in {u for u, _ in self.samples}}
        self.starts, seen = [], {}
        for uid, w in self.samples:                     # offset of each window in its case
            s = sequences[uid]
            k = seen.get(uid, 0)
            starts = ([0] if len(s) <= max_len + 1 else
                      sorted(set(list(range(0, len(s) - max_len, max_len)) + [len(s) - max_len - 1])))
            self.starts.append(starts[k])
            seen[uid] = k + 1

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        uid, w = self.samples[idx]
        st = self.starts[idx]
        rows = np.unpackbits(self.packed[uid][st:st + len(w) - 1], axis=1,
                             count=self.n_items).astype(bool)
        rej = np.zeros((self.max_len, self.n_items), dtype=bool)
        rej[self.max_len - len(rows):] = rows
        item["reject"] = torch.from_numpy(rej)
        return item


class AllowWindowDataset(CausalWindowDataset):
    """CausalWindowDataset plus, for every target position, the output mask of
    the reachability graph (models/rgmask.py) after the WHOLE case prefix.
    PAD positions get the "no mask" row."""

    def __init__(self, sequences: dict, max_len: int, rg):
        super().__init__(sequences, max_len)
        self.rg = rg
        self.states = {uid: rg.state_ids(sequences[uid]) for uid in {u for u, _ in self.samples}}
        self.starts, seen = [], {}
        for uid, w in self.samples:
            s = sequences[uid]
            k = seen.get(uid, 0)
            starts = ([0] if len(s) <= max_len + 1 else
                      sorted(set(list(range(0, len(s) - max_len, max_len)) + [len(s) - max_len - 1])))
            self.starts.append(starts[k])
            seen[uid] = k + 1

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        uid, w = self.samples[idx]
        st = self.starts[idx]
        ids = np.full(self.max_len, self.rg.unknown, dtype=np.int64)
        n = len(w) - 1
        ids[self.max_len - n:] = self.states[uid][st:st + n]
        item["allow"] = torch.from_numpy(self.rg.table[ids])
        return item


class MaskedWindowDataset(Dataset):
    """Masked-item training for BERT4Rec over all windows of max_len events.
    Masks are redrawn at every access (dynamic masking), at least one per sample.

    last_item_prob: probability that a sample masks ONLY its last event, as in
    the original BERT4Rec."""

    def __init__(self, sequences: dict, max_len: int = 200,
                 mask_prob: float = 0.15, seed: int = 42,
                 last_item_prob: float = 0.0):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.last_item_prob = last_item_prob
        self.rng = random.Random(seed)
        self.samples = []
        for uid in sorted(sequences):
            s = sequences[uid]
            if not s:
                continue
            for w in _windows(s, max_len, max_len):
                self.samples.append((uid, w))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, w = self.samples[idx]
        if self.last_item_prob > 0 and self.rng.random() < self.last_item_prob:
            tokens = list(w[:-1]) + [MASK_ID]
            labels = [PAD_ID] * (len(w) - 1) + [w[-1]]
            return {
                "user_id": uid,
                "input":  torch.tensor(_left_pad(tokens, self.max_len), dtype=torch.long),
                "target": torch.tensor(_left_pad(labels, self.max_len), dtype=torch.long),
            }
        tokens, labels = [], []
        for item in w:
            if self.rng.random() < self.mask_prob:
                tokens.append(MASK_ID)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(PAD_ID)
        if all(l == PAD_ID for l in labels):
            tokens[-1], labels[-1] = MASK_ID, w[-1]
        return {
            "user_id": uid,
            "input":  torch.tensor(_left_pad(tokens, self.max_len), dtype=torch.long),
            "target": torch.tensor(_left_pad(labels, self.max_len), dtype=torch.long),
        }


class PrefixEvalDataset(Dataset):
    """One sample per (learner, position t >= min_prefix): predict seq[t] from
    seq[:t], using its last max_len events (max_len - 1 + MASK for BERT4Rec).
"""

    def __init__(self, sequences: dict, max_len: int = 200,
                 mask_for_bert: bool = False, min_prefix: int = 1,
                 max_per_learner: int | None = None, seed: int = 0):
        self.sequences = sequences
        self.max_len = max_len
        self.mask_for_bert = mask_for_bert
        self.items = []
        for uid in sorted(sequences):
            n = len(sequences[uid])
            ts = list(range(min_prefix, n))
            if max_per_learner is not None and len(ts) > max_per_learner:
                key = uid if isinstance(uid, int) else zlib.crc32(str(uid).encode())
                ts = sorted(random.Random(seed * 1_000_003 + key)
                            .sample(ts, max_per_learner))
            self.items.extend((uid, t) for t in ts)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        uid, t = self.items[idx]
        s = self.sequences[uid]
        prefix, target = s[:t], s[t]
        if self.mask_for_bert:
            inp = prefix[-(self.max_len - 1):] + [MASK_ID]
        else:
            inp = prefix[-self.max_len:]
        return {
            "user_id": uid,
            "pos":     t,
            "input":   torch.tensor(_left_pad(inp, self.max_len), dtype=torch.long),
            "target":  torch.tensor(target, dtype=torch.long),
        }