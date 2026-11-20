"""
Declare constraints mined from the training cases, and step-level violations
============================================================================
Knowledge used by the Mezini et al. (2026) competitor (models/logic_loss.py):

  A. Mezini, E. Umili, I. Donadello, F. M. Maggi, M. Mancanelli, F. Patrizi.
  "Neuro-Symbolic Predictive Process Monitoring." Information Systems 141
  (2026) 102762.

mine_constraints()
    Declare constraints satisfied by at least `min_support` of the training
    cases (85% in the paper). Mined from the TRAINING cases of the fold only;
    the paper mines them from its test set. Support uses vacuity detection
    (see mine_constraints). pm4py's discover_declare is not
    used: its negative templates are the negation of the positive ones
    (nonchainsuccession = trace does not satisfy chainsuccession), not the
    Declare semantics (a never directly followed by b).

reject_matrix()
    For every position t of a case, the activities whose occurrence at t+1
    would permanently violate a mined constraint (the transitions into a
    failing state of the constraint's automaton, Eq. (12) of the paper).


Only templates with failing states reachable without an end-of-trace symbol
give a step-level signal, because the models do not predict the end of the
case: precedence, altprecedence, chainprecedence,
altresponse, chainresponse, noncoexistence, nonsuccession, nonchainsuccession
(absence and at-most-one are supported by reject_matrix but not mined;
the safety parts of succession, altsuccession and chainsuccession are
precedence, altprecedence + altresponse and chainprecedence + chainresponse). Liveness templates (existence, response,
coexistence, responded_existence) can only be violated at the end of the
case and are not used. init is not used either: every prediction is made
after at least one event.

A constraint the prefix has already violated is dropped for the rest of the
case (its automaton is in a failing state, from which every activity would be
"rejecting"); the paper avoids the situation by removing violating traces.
"""

from __future__ import annotations

import warnings

import numpy as np

STEP_TEMPLATES = ["absence", "atmost1", "precedence", "altprecedence", "chainprecedence",
                  "altresponse", "chainresponse", "noncoexistence", "nonsuccession",
                  "nonchainsuccession"]
UNARY = {"absence", "atmost1"}


def _positions(s: list) -> tuple:
    pos = {}
    for i, x in enumerate(s):
        pos.setdefault(x, []).append(i)
    return pos, {(s[i], s[i + 1]) for i in range(len(s) - 1)}


def _pair_satisfied(s: list, a: int, b: int, pos=None, df=None) -> dict:
    """Full LTLf satisfaction of every binary template on trace s, a != b."""
    if pos is None:
        pos, df = _positions(s)
    pa, pb = pos.get(a, []), pos.get(b, [])
    n_ab = sum(1 for i in pa if i + 1 < len(s) and s[i + 1] == b)
    out = {
        "precedence": not pb or (bool(pa) and pa[0] < pb[0]),
        "chainprecedence": n_ab == len(pb) and (not pb or pb[0] > 0),
        "chainresponse": n_ab == len(pa),
        "noncoexistence": not (pa and pb),
        "nonsuccession": not (pa and pb and pa[0] < pb[-1]),
        "nonchainsuccession": (a, b) not in df,
    }
    merged = sorted([(i, 0) for i in pa] + [(i, 1) for i in pb])   # 0 = a, 1 = b
    ok, seen_a = True, False                       # altprecedence: an a before each b
    for _, k in merged:
        if k == 1:
            ok &= seen_a
            seen_a = False
        else:
            seen_a = True
    out["altprecedence"] = ok
    ok, pending = True, False                      # altresponse: each a answered before next a
    for _, k in merged:
        if k == 0:
            ok &= not pending
            pending = True
        else:
            pending = False
    out["altresponse"] = ok and not pending
    return out


# activating activity of each binary template: 0 = a, 1 = b
ACTIVATION = {"precedence": 1, "altprecedence": 1, "chainprecedence": 1,
              "altresponse": 0, "chainresponse": 0, "noncoexistence": 0,
              "nonsuccession": 0, "nonchainsuccession": 0}
# satisfaction when the activating activity occurs and the other one does not
_ALONE = {"precedence": False, "altprecedence": False, "chainprecedence": False,
          "altresponse": False, "chainresponse": False, "noncoexistence": True,
          "nonsuccession": True, "nonchainsuccession": True}


def mine_constraints(train: dict, id2item: dict | None = None, min_support: float = 0.85) -> dict:
 
    from collections import Counter
    seqs = [s for s in train.values() if s]
    n = len(seqs)
    need = min_support * n
    n_has = Counter()
    for s in seqs:
        n_has.update(set(s))
    frequent = {a for a, k in n_has.items() if k >= need}
    co, n_co = Counter(), Counter()
    for s in seqs:
        pos, df = _positions(s)
        present = set(pos)
        for x in present & frequent:               # x activates as a or as b
            for y in present:
                if y == x:
                    continue
                for a, b in ((x, y), (y, x)):
                    if (a, b, "_") in co:
                        continue
                    co[(a, b, "_")] = 0            # marker: counted for this trace
                    n_co[(a, b)] += 1
                    for t, ok in _pair_satisfied(s, a, b, pos, df).items():
                        co[(a, b, t)] += ok
        for k in [k for k in co if k[2] == "_"]:
            del co[k]
    out = {t: [] for t in ACTIVATION}
    acts = sorted(n_has)
    for a in acts:
        for b in acts:
            if a == b:
                continue
            both = n_co[(a, b)]
            for t, who in ACTIVATION.items():
                act = (a, b)[who]
                if act not in frequent:
                    continue
                alone = n_has[act] - both
                if co[(a, b, t)] + alone * _ALONE[t] >= need:
                    out[t].append((a, b))
    return {k: np.array(sorted(v), dtype=np.int64).reshape(-1, 2) for k, v in out.items() if v}


def n_constraints(constraints: dict) -> int:
    return sum(len(v) for v in constraints.values())


def reject_matrix(seq: list, constraints: dict, n_items: int) -> np.ndarray:
    """Bool array (len(seq) - 1, n_items): row t = activities that would
    newly violate a constraint if they occurred right after seq[:t + 1]."""
    T = len(seq)
    R = np.zeros((max(T - 1, 0), n_items), dtype=bool)
    c = constraints
    get = lambda k: c.get(k, np.zeros((0, 1 if k in UNARY else 2), np.int64))
    ab, am1 = get("absence")[:, 0], get("atmost1")[:, 0]
    pr, ap, cp = get("precedence"), get("altprecedence"), get("chainprecedence")
    ar, cr = get("altresponse"), get("chainresponse")
    nc, ns, ncs = get("noncoexistence"), get("nonsuccession"), get("nonchainsuccession")
    viol = {k: np.zeros(len(get(k)), bool) for k in
            ("absence", "atmost1", "precedence", "altprecedence", "chainprecedence",
             "altresponse", "chainresponse", "noncoexistence", "nonsuccession",
             "nonchainsuccession")}
    ap_ok = np.zeros(len(ap), bool)        # an a since the last b
    ar_pending = np.zeros(len(ar), bool)   # an a not yet answered by b
    seen = np.zeros(n_items, bool)
    last = -1

    for t in range(T):
        x = seq[t]
        # --- update automata with event x (violations of the prefix) ---
        viol["absence"] |= ab == x
        viol["atmost1"] |= (am1 == x) & seen[am1]
        viol["precedence"] |= (pr[:, 1] == x) & ~seen[pr[:, 0]]
        viol["altprecedence"] |= (ap[:, 1] == x) & ~ap_ok
        viol["chainprecedence"] |= (cp[:, 1] == x) & (cp[:, 0] != last)
        viol["altresponse"] |= (ar[:, 0] == x) & ar_pending
        viol["chainresponse"] |= (cr[:, 0] == last) & (cr[:, 1] != x)
        viol["noncoexistence"] |= (nc[:, 1] == x) & seen[nc[:, 0]]
        viol["nonsuccession"] |= (ns[:, 1] == x) & seen[ns[:, 0]]
        viol["nonchainsuccession"] |= (ncs[:, 1] == x) & (ncs[:, 0] == last)
        ap_ok[ap[:, 1] == x] = False
        ap_ok[ap[:, 0] == x] = True
        ar_pending[ar[:, 1] == x] = False
        ar_pending[ar[:, 0] == x] = True
        seen[x] = True
        last = x
        if t == T - 1:
            break
        # --- activities that would violate a still-satisfied constraint ---
        r = R[t]
        r[ab[~viol["absence"]]] = True
        r[am1[~viol["atmost1"] & seen[am1]]] = True
        r[pr[~viol["precedence"] & ~seen[pr[:, 0]], 1]] = True
        r[ap[~viol["altprecedence"] & ~ap_ok, 1]] = True
        r[cp[~viol["chainprecedence"] & (cp[:, 0] != last), 1]] = True
        r[ar[~viol["altresponse"] & ar_pending, 0]] = True
        r[nc[~viol["noncoexistence"] & seen[nc[:, 0]], 1]] = True
        r[ns[~viol["nonsuccession"] & seen[ns[:, 0]], 1]] = True
        r[ncs[~viol["nonchainsuccession"] & (ncs[:, 0] == last), 1]] = True
        act = ~viol["chainresponse"] & (cr[:, 0] == last)
        if act.any():                       # after a: everything but b
            allowed = np.zeros(n_items, bool)
            targets = cr[act, 1]
            if len(set(targets.tolist())) == 1:
                allowed[targets[0]] = True
            r |= ~allowed                   
        r[:2] = False                       # PAD and MASK are never targets
    return R
