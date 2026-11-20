"""
Reachability-graph masking on SASRec (controlled version of Theis & Darabi 2023)
================================================================================
The output mask of J. Theis, H. Darabi, "Improving Predictive Process
Monitoring Through Reachability Graph-Based Masking of Neural Networks", applied to the SASRec baseline:

    o = MSK * softmax(x) / sum(MSK * softmax(x))

in training and at prediction, with the same Petri net, replay and mask as
models/dream.py (Inductive Miner on the training cases; mask = activities
enabled in the reached marking if that marking was reached when replaying the
training cases, 1e-10 for the others; no mask for an unknown marking). Only
the injection point differs from our priors: the network and its
hyperparameters are those of the SASRec baseline.

Training masks come from the replay of the whole case prefix. At prediction,
the visible prefix (at most the context window) is replayed; for prefixes
longer than the window, a marking reached from a truncated prefix is usually
unknown, and the prediction is then left unmasked.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_FLOOR = 1e-10


class ReachabilityMask:

    def __init__(self, train: dict, id2item: dict, noise_threshold: float = 0.20):
        from pmpriors.petri import Replayer, discover_net
        net, im, _ = discover_net(train, id2item, noise_threshold)
        self.rp = Replayer(net, im, {v: k for k, v in id2item.items()})
        self.n_items = len(id2item)
        self.states: dict = {}
        self.table: list = []
        for seq in train.values():
            self.state_ids(seq, explore=True)
        unknown = np.ones(self.n_items, dtype=np.float32)
        unknown[:2] = 0.0
        self.table.append(unknown)             
        self.table = np.stack(self.table)
        self.unknown = len(self.table) - 1
        self._prefix_cache: dict = {}

    def _state(self, m, explore: bool) -> int:
        if m not in self.states:
            if not explore:
                return -1
            self.states[m] = len(self.table)
            allowed = self.rp.enabled(m)
            row = np.full(self.n_items, MASK_FLOOR, dtype=np.float32)
            if allowed:
                row[list(allowed)] = 1.0
            else:
                row[:] = 1.0
            row[:2] = 0.0
            self.table.append(row)
        return self.states[m]

    def state_ids(self, seq: list, explore: bool = False) -> np.ndarray:
        m = self.rp.start()
        out = np.empty(len(seq), dtype=np.int64)
        for i, x in enumerate(seq):
            m, _, _ = self.rp.fire(m, x)
            s = self._state(m, explore)
            out[i] = s if s >= 0 else getattr(self, "unknown", -1)
        return out

    def for_prefix(self, prefix: tuple) -> np.ndarray:
        if prefix not in self._prefix_cache:
            s = self.state_ids(list(prefix))[-1] if prefix else self.unknown
            self._prefix_cache[prefix] = s
        return self.table[self._prefix_cache[prefix]]


class MaskedSASRec(nn.Module):
    """SASRec output distribution is masked by the reachability graph."""

    def __init__(self, sasrec: nn.Module, rg: ReachabilityMask):
        super().__init__()
        self.sasrec = sasrec
        self.rg = rg

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.sasrec(input_ids)                       # logits; masked in the loss

    @staticmethod
    def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(mask == 0, float("-inf"))    # PAD, MASK
        p = torch.softmax(logits, -1) * mask
        p = p / p.sum(-1, keepdim=True)
        return torch.log(p.clamp_min(1e-7))

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.sasrec(input_ids)[:, -1, :]
        rows = [self.rg.for_prefix(tuple(i for i in r if i > 1)) for r in input_ids.tolist()]
        mask = torch.from_numpy(np.stack(rows)).to(logits.device)
        out = self.masked_log_probs(logits, mask)
        out[:, 0] = float("-inf")
        out[:, 1] = float("-inf")
        return out


class MaskedLoss:

    def __call__(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        logp = MaskedSASRec.masked_log_probs(logits, batch["allow"])
        return F.nll_loss(logp.transpose(1, 2), batch["target"], ignore_index=0)