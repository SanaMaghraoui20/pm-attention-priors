"""
Count-based baselines
=====================
PopRec : global activity frequency in the training cases, the same score
         vector for every prefix (floor of reference).
PMOnly : first-order directly-follows model. For a prefix ending with
         activity i, activities are ranked by the number of times they
         directly follow i in the training cases (raw counts, no smoothing);
         if i never occurs as a source, global frequency is used. It uses the
         same counts as the DFG prior, without a neural network.

Both implement predict(input_ids (B, L)) -> scores (B, V) with PAD and MASK
at -inf, like the neural models, and are fitted on {case: [item ids]}.
"""

import numpy as np
import torch


class PopRec:
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.item_counts = np.zeros(n_items, dtype=np.float32)

    def fit(self, train: dict[int, list[int]]) -> None:
        for seq in train.values():
            for item in seq:
                self.item_counts[item] += 1

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        scores = self.item_counts.copy()
        scores[0] = -np.inf
        scores[1] = -np.inf
        return torch.from_numpy(scores).unsqueeze(0).expand(input_ids.shape[0], -1).clone()


class PMOnly:
    def __init__(self, n_items: int):
        self.n_items = n_items
        self.dfg = np.zeros((n_items, n_items), dtype=np.float32)

    def fit(self, train: dict[int, list[int]]) -> None:
        for seq in train.values():
            for a, b in zip(seq[:-1], seq[1:]):
                self.dfg[a][b] += 1

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        last = input_ids[:, -1].cpu().numpy()
        scores = self.dfg[last].copy()
        zero_rows = scores.sum(axis=1) == 0
        if zero_rows.any():                       
            scores[zero_rows] = self.dfg.sum(axis=0)
        out = torch.from_numpy(scores)
        out[:, 0] = float("-inf")
        out[:, 1] = float("-inf")
        return out
