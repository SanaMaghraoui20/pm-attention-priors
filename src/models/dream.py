"""
Decay replay mining for next-activity prediction (Theis & Darabi)
=================================================================
Two competitors of the paper:

  dream_nap     J. Theis, H. Darabi. "Decay Replay Mining to Predict Next
                Process Events." IEEE Access 7, 2019 (input injection).
  dream_nap_rg  J. Theis, H. Darabi. "Improving Predictive Process Monitoring
                Through Reachability Graph-Based Masking of Neural Networks."
                IEEE Trans. Computational Social Systems 10(4), 2023
                (input + output injection).

Reimplemented from the papers

Pipeline, fitted on the TRAINING cases of the fold:
  1. Petri net: Inductive Miner, same noise threshold as our priors
     (the authors use given PNML models).
  2. Enhancement: replay the training cases; for each place, the time deltas
     between consecutive activations; linear decay
         f_p(dt) = max(0, beta - alpha_p * dt),  beta = 10,
         alpha_p = 1 / mean(deltas_p)   (1 / longest case duration if < 2 deltas).
     PyDREAM appends the deltas of every activated place to the list of the
     last place of its loop (variable reuse); we follow the paper (per place).
  3. Timed state sample after each event: decay values, token counts per
     place, marking (S = [F, C, M]), min-max scaled on the training samples.
  4. Reduced reachability graph from the training replay: for each marking
     reached, the visible activities that can occur next. Mask of a prefix =
     that set for its marking (1 for allowed, 1e-10 otherwise, as in
     PyDREAM); a marking never seen in training gives no mask (all ones).
Network (sizes relative to the number of features F, as in PyDREAM):
  dream_nap:    F -> 1.2F -> 0.6F -> 0.3F -> softmax, ReLU, dropout 0.2.
  dream_nap_rg: [F (S) || V (mask)] -> 1.2F -> 0.6F -> 0.3F -> softmax,
                Swish, dropout 0.2, then o = mask * p / sum(mask * p) (Eq. 5),
                in training and at prediction.
Training: Adam lr 1e-3, batch 64, at most 100 epochs; model selection on the
validation cases (NDCG@5 with the regime's patience, like every other model;
the authors keep the epoch of lowest validation loss).

Adaptations: the first prediction is made after ONE event (PyDREAM
starts after two), so every model is evaluated on the same prefixes;
resource counts are not used (no model of the comparison sees resources).
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BETA = 10.0
MASK_FLOOR = 1e-10


class DecayReplay:
    def __init__(self, train: dict, times: dict, id2item: dict, noise_threshold: float = 0.20):
        from pmpriors.petri import Replayer, discover_net
        net, im, _ = discover_net(train, id2item, noise_threshold)
        item2id = {v: k for k, v in id2item.items()}
        self.rp = Replayer(net, im, item2id)
        self.n_items = len(id2item)
        P = self.rp.n_places
        # --- enhancement: decay parameters per place ---
        deltas = [[] for _ in range(P)]
        longest = 1.0
        for c, seq in train.items():
            ts = times[c]
            longest = max(longest, ts[-1] - ts[0])
            last = [None] * P
            m = self.rp.start()
            for p, n in enumerate(m):
                if n:
                    last[p] = ts[0]
            for x, t in zip(seq, ts):
                m, act, _ = self.rp.fire(m, x)
                for p in act:
                    if last[p] is not None and t - last[p] > 0:
                        deltas[p].append(t - last[p])
                    last[p] = t
        self.alpha = np.array([1.0 / np.mean(d) if len(d) > 1 else 1.0 / longest
                               for d in deltas])
        # --- reduced reachability graph from the training replay ---
        self.states: dict = {}
        self.state_masks: list = []
        feats = []
        for c, seq in train.items():
            f, s = self._replay(seq, times[c], explore=True)
            feats.append(f)
        allf = np.concatenate(feats)
        self.fmin, self.fmax = allf.min(0), allf.max(0)
        self.n_features = allf.shape[1]

    def _state(self, m: tuple, explore: bool) -> int:
        k = m
        if k not in self.states:
            if not explore:
                return -1
            self.states[k] = len(self.state_masks)
            allowed = self.rp.enabled(m)
            mask = np.full(self.n_items, MASK_FLOOR, dtype=np.float32)
            if allowed:
                mask[list(allowed)] = 1.0
            else:                                       
                mask[:] = 1.0
            mask[:2] = 0.0                              # PAD, MASK
            self.state_masks.append(mask)
        return self.states[k]

    def _replay(self, seq: list, ts: list, explore: bool = False):
        """Features (len(seq), 3P) and state id after each event."""
        P = self.rp.n_places
        last = np.full(P, np.nan)
        counts = np.zeros(P)
        m = self.rp.start()
        for p, n in enumerate(m):
            if n:
                last[p] = ts[0]
                counts[p] += 1
        out = np.zeros((len(seq), 3 * P), dtype=np.float32)
        states = np.zeros(len(seq), dtype=np.int64)
        for i, (x, t) in enumerate(zip(seq, ts)):
            m, act, _ = self.rp.fire(m, x)
            for p in act:
                last[p] = t
                counts[p] += 1
            dt = t - last
            decay = np.where(np.isnan(last), 0.0, np.maximum(0.0, BETA - self.alpha * np.nan_to_num(dt)))
            marking = np.asarray(m, dtype=float)
            out[i] = np.concatenate([decay, counts, marking])
            states[i] = self._state(m, explore)
        return out, states

    def transform(self, seq: list, ts: list):
        f, s = self._replay(seq, ts, explore=False)
        rng = np.where(self.fmax > self.fmin, self.fmax - self.fmin, 1.0)
        return ((f - self.fmin) / rng).astype(np.float32), s

    def masks(self, states: np.ndarray) -> np.ndarray:
        table = np.stack(self.state_masks + [np.r_[np.zeros(2), np.ones(self.n_items - 2)]
                                             .astype(np.float32)])
        return table[np.where(states < 0, len(self.state_masks), states)]


class DreamNAP(nn.Module):
    def __init__(self, n_features: int, n_items: int, masked: bool, dropout: float = 0.2):
        super().__init__()
        F = n_features
        act = nn.SiLU if masked else nn.ReLU            # 2023 version
        h = lambda r: max(1, int(F * r))
        self.masked = masked
        self.s_in = nn.Sequential(nn.Linear(F, F), act(), nn.Dropout(dropout))
        self.m_in = nn.Sequential(nn.Linear(n_items, n_items), act(), nn.Dropout(dropout)) if masked else None
        width = F + (n_items if masked else 0)
        self.body = nn.Sequential(
            nn.Linear(width, h(1.2)), act(), nn.Dropout(dropout),
            nn.Linear(h(1.2), h(0.6)), act(), nn.Dropout(dropout),
            nn.Linear(h(0.6), h(0.3)), act(), nn.Dropout(dropout),
            nn.Linear(h(0.3), n_items))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log-probabilities (B, n_items); PAD and MASK get probability 0."""
        z = self.s_in(x)
        if self.masked:
            z = torch.cat([z, self.m_in(mask)], 1)
        logits = self.body(z)
        pad = torch.zeros_like(logits, dtype=torch.bool)
        pad[:, :2] = True
        logits = logits.masked_fill(pad, float("-inf"))
        p = torch.softmax(logits, 1)
        if self.masked:                                  
            p = p * mask
            p = p / p.sum(1, keepdim=True)
        return torch.log(p.clamp_min(1e-7))              


def _samples(rep: DecayReplay, seqs: dict, times: dict, items=None):
    """(features, masks, targets, case ids, positions) for every prefix t >= 1
    of `seqs`, or only for the (case, t) pairs of `items`."""
    X, S, Y, U, T = [], [], [], [], []
    want = None
    if items is not None:
        want = {}
        for u, t in items:
            want.setdefault(u, []).append(t)
    for c in sorted(seqs):
        seq = seqs[c]
        if len(seq) < 2:
            continue
        f, s = rep.transform(seq, times[c])
        ts = want.get(c, []) if want is not None else range(1, len(seq))
        for t in ts:
            X.append(f[t - 1]); S.append(s[t - 1]); Y.append(seq[t]); U.append(c); T.append(t)
    return (np.array(X, dtype=np.float32), np.array(S, dtype=np.int64),
            np.array(Y, dtype=np.int64), U, T)


@torch.no_grad()
def _predict(model, rep, X, S, Y, U, T, device, bs=1024) -> pd.DataFrame:
    model.eval()
    M = rep.masks(S)
    cols = {"user_id": U, "pos": T, "target": Y.tolist(), "pred": [], "rank": []}
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        m = torch.from_numpy(M[i:i + bs]).to(device)
        y = torch.from_numpy(Y[i:i + bs]).to(device)
        scores = model(x, m)
        tv = scores.gather(1, y.unsqueeze(1))
        rank = (scores > tv).sum(1).float() + 1.0 + ((scores == tv).sum(1) - 1).float() / 2.0
        cols["pred"].extend(scores.argmax(1).cpu().tolist())
        cols["rank"].extend(rank.cpu().tolist())
    return pd.DataFrame(cols)


def run_dream(variant: str, train, val, test, times, id2item, val_items, test_items,
              regime: dict, noise_threshold: float, seed: int, device: str = "cpu") -> dict:
    from training.evaluation import metrics_from_predictions
    t0 = time.time()
    torch.manual_seed(seed)
    rep = DecayReplay(train, times, id2item, noise_threshold)
    Xtr, Str, Ytr, _, _ = _samples(rep, train, times)
    val_s = _samples(rep, val, times, val_items)
    test_s = _samples(rep, test, times, test_items)
    Mtr = rep.masks(Str)
    model = DreamNAP(rep.n_features, rep.n_items, masked=(variant == "dream_nap_rg")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs, patience, bs = min(100, regime["n_epochs"]), regime["patience"], 64
    best, best_state, best_epoch, wait = -1.0, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for epoch in range(1, n_epochs + 1):
        model.train()
        for idx in torch.randperm(len(Xtr), generator=g).split(bs):
            idx = idx.numpy()          
            x = torch.from_numpy(Xtr[idx]).to(device)
            m = torch.from_numpy(Mtr[idx]).to(device)
            y = torch.from_numpy(Ytr[idx]).to(device)
            loss = nn.functional.nll_loss(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = metrics_from_predictions(_predict(model, rep, *val_s, device))["NDCG@5"]
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    preds = _predict(model, rep, *test_s, device)
    in_rg = float(np.mean(test_s[1] >= 0)) if len(test_s[1]) else float("nan")
    return {"preds": preds, "best_epoch": best_epoch, "val_NDCG@5": best,
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_places": rep.rp.n_places, "n_states": len(rep.state_masks),
            "test_marking_known": in_rg, "seconds": round(time.time() - t0, 1)}