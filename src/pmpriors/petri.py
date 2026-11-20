"""
Petri net discovered from the training cases, and token-game replay
===================================================================
Used by the Theis & Darabi competitors (models/dream.py).

discover_net()    Inductive Miner (pm4py) on the training cases, with the same
                  noise threshold as the behavioural-profile prior.
Replayer          token game with silent transitions; a marking is a tuple of
                  token counts, one per place.
    fire(m, item) -> (new marking, activated places, fits)
                  An event whose transition is not enabled is reached through
                  the shortest sequence of silent transitions (breadth-first,
                  at most `max_states` markings explored); if none enables it,
                  the missing tokens are added and the event is fired anyway
                  (fits = False), as in PyDREAM's decay replay. An activity
                  without a transition leaves the marking unchanged.
    enabled(m) -> visible activities that can occur next (directly or after
                  silent transitions) = the out-edges of the marking's node in
                  the reduced reachability graph RG_r of Theis & Darabi (2023,
                  Sec. IV-A).
Both are memoised per marking.
"""

from __future__ import annotations

import warnings
from collections import deque


def discover_net(train: dict, id2item: dict, noise_threshold: float = 0.20):
    import pandas as pd
    import pm4py

    rows = [(str(c), id2item[i], pos) for c, s in train.items() for pos, i in enumerate(s)]
    df = pd.DataFrame(rows, columns=["case:concept:name", "concept:name", "pos"])
    df["time:timestamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(df.pop("pos"), unit="s")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm4py.discover_petri_net_inductive(df, noise_threshold=noise_threshold)


class Replayer:
    def __init__(self, net, im, item2id: dict, max_states: int = 64):
        places = sorted(net.places, key=lambda p: str(p.name))
        pidx = {p: k for k, p in enumerate(places)}
        self.n_places = len(places)
        self.trans = []                                  # (pre, post, item id or None)
        for t in sorted(net.transitions, key=lambda t: str(t.name)):
            pre = tuple(sorted((pidx[a.source], a.weight) for a in t.in_arcs))
            post = tuple(sorted((pidx[a.target], a.weight) for a in t.out_arcs))
            self.trans.append((pre, post, item2id.get(t.label) if t.label is not None else None))
        self.silent = [k for k, t in enumerate(self.trans) if t[2] is None]
        self.by_item: dict[int, list[int]] = {}
        for k, t in enumerate(self.trans):
            if t[2] is not None:
                self.by_item.setdefault(t[2], []).append(k)
        m0 = [0] * self.n_places
        for p, n in im.items():
            m0[pidx[p]] = n
        self.initial = tuple(m0)
        self.max_states = max_states
        self._fire_cache: dict = {}
        self._enabled_cache: dict = {}

    
    def _is_enabled(self, k: int, m: tuple) -> bool:
        return all(m[p] >= w for p, w in self.trans[k][0])

    def _fire(self, k: int, m: tuple) -> tuple:
        x = list(m)
        for p, w in self.trans[k][0]:
            x[p] -= w
        for p, w in self.trans[k][1]:
            x[p] += w
        return tuple(x)

    def _silent_paths(self, m: tuple):
        """Breadth-first over silent firings: (marking, fired silent
        transitions), shortest paths first, at most max_states markings."""
        seen = {m}
        queue = deque([(m, ())])
        while queue:
            cur, path = queue.popleft()
            yield cur, path
            for k in self.silent:
                if len(seen) >= self.max_states:
                    break
                if self._is_enabled(k, cur):
                    nxt = self._fire(k, cur)
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, path + (k,)))

    def fire(self, m: tuple, item: int):
        
        key = (m, item)
        hit = self._fire_cache.get(key)
        if hit is None:
            hit = self._fire_cache[key] = self._fire_uncached(m, item)
        return hit

    def _fire_uncached(self, m: tuple, item: int):
        cands = self.by_item.get(item)
        if not cands:
            return m, (), False
        for k in cands:                                  
            if self._is_enabled(k, m):
                return self._fire(k, m), tuple(p for p, _ in self.trans[k][1]), True
        for cur, path in self._silent_paths(m):
            for k in cands:
                if self._is_enabled(k, cur):
                    act = [p for s in path for p, _ in self.trans[s][1]]
                    act += [p for p, _ in self.trans[k][1]]
                    return self._fire(k, cur), tuple(act), True
        k = cands[0]                                     # forced: add missing tokens
        x = list(m)
        for p, w in self.trans[k][0]:
            x[p] = max(x[p], w)
        return self._fire(k, tuple(x)), tuple(p for p, _ in self.trans[k][1]), False

    def enabled(self, m: tuple) -> frozenset:
        hit = self._enabled_cache.get(m)
        if hit is None:
            out = set()
            for cur, _ in self._silent_paths(m):
                for item, ks in self.by_item.items():
                    if item not in out and any(self._is_enabled(k, cur) for k in ks):
                        out.add(item)
            hit = self._enabled_cache[m] = frozenset(out)
        return hit

    def start(self) -> tuple:
        return self.initial