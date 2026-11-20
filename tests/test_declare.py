"""Declare constraints and step-level rejections (Mezini et al. competitor)."""
import random

import numpy as np
import torch

from data.windows import CausalWindowDataset, RejectWindowDataset
from pmpriors.declare import mine_constraints, reject_matrix
from training.logic_loss import LocalLogicLoss

A, B, C, D = 2, 3, 4, 5
V = 6


def rej(seq, **cons):
    c = {k: np.array(v).reshape(-1, 1 if k in ("absence", "atmost1") else 2) for k, v in cons.items()}
    return reject_matrix(seq, c, V)


def test_templates():
    # precedence(a, b): b rejected until a has occurred
    r = rej([C, A, C], precedence=[[A, B]])
    assert r[0, B] and not r[1, B]
    # chainresponse(a, b): right after a, everything but b is rejected
    r = rej([C, A, B], chainresponse=[[A, B]])
    assert r[1, C] and r[1, A] and not r[1, B] and not r[0, C]
    # nonsuccession(a, b): b rejected once a has occurred
    r = rej([C, A, C], nonsuccession=[[A, B]])
    assert not r[0, B] and r[1, B]
    # atmost1: second occurrence rejected
    r = rej([A, C, C], atmost1=[A])
    assert r[0, A] and r[1, A]
    # altresponse(a, b): a again before b is rejected
    r = rej([A, B, C], altresponse=[[A, B]])
    assert r[0, A] and not r[1, A]
    # PAD and MASK never rejected
    assert not rej([C, A, C], absence=[0, 1]).any()


def test_violated_constraint_is_dropped():
    r = rej([B, C, B], precedence=[[A, B]])      # violated at the first event
    assert not r[:, B].any()


def _toy_log(n=200, seed=0):
    rng = random.Random(seed)
    seqs = {}
    for c in range(n):
        s = [A] + [rng.choice([B, C]) for _ in range(rng.randint(1, 4))] + [D]
        if rng.random() < 0.05:
            s.insert(1, D)                         # rare deviation
        seqs[c] = s
    return seqs, {0: "<PAD>", 1: "<MASK>", A: "a", B: "b", C: "c", D: "d"}


def test_mined_constraints_hold_on_training_traces():
    """Each mined constraint, replayed by the step automata, is satisfied by
    at least the support threshold (safety part), and the miner agrees with a
    brute-force count of full satisfaction."""
    seqs, id2item = _toy_log()
    cons = mine_constraints(seqs, id2item, min_support=0.85)
    assert cons, "nothing mined"
    dummy = V                                      # neutral first event
    for tmpl, arr in cons.items():
        for row in arr:
            one = {tmpl: row.reshape(1, -1)}
            ok = 0
            for s in seqs.values():
                x = [dummy] + s
                r = reject_matrix(x, one, V + 1)
                ok += not any(r[t, x[t + 1]] for t in range(len(x) - 1))
            assert ok / len(seqs) >= 0.85, (tmpl, row, ok / len(seqs))


def test_miner_matches_brute_force():
    """Support with vacuity detection = activated and satisfied."""
    from pmpriors.declare import ACTIVATION, _pair_satisfied
    seqs, id2item = _toy_log()
    cons = mine_constraints(seqs, id2item, min_support=0.85)
    n = len(seqs)
    acts = [A, B, C, D]
    for t, who in ACTIVATION.items():
        mined = {tuple(r) for r in cons.get(t, np.zeros((0, 2), int))}
        brute = {(a, b) for a in acts for b in acts if a != b
                 and sum((a, b)[who] in s and _pair_satisfied(s, a, b)[t]
                         for s in seqs.values()) >= 0.85 * n}
        assert mined == brute, t


def test_reject_windows_align_with_targets():
    seqs, id2item = _toy_log(50)
    cons = mine_constraints(seqs, id2item)
    long = {0: (seqs[0] * 10)[:37]}                # several windows
    ds = RejectWindowDataset(long, 8, cons, V)
    base = CausalWindowDataset(long, 8)
    full = reject_matrix(long[0], cons, V)
    assert len(ds) == len(base) > 2
    s = long[0]
    for i in range(len(ds)):
        item = ds[i]
        tgt = item["target"].tolist()
        n = sum(t != 0 for t in tgt)
        st = ds.starts[i]
        assert tgt[-n:] == s[st + 1:st + 1 + n]
        assert np.array_equal(item["reject"][-n:].numpy(), full[st:st + n])


def test_loss_penalises_rejected_mass():
    logits = torch.zeros(1, 2, V)
    batch = {"target": torch.tensor([[A, 0]]), "reject": torch.zeros(1, 2, V, dtype=torch.bool)}
    base = LocalLogicLoss(0.25)(logits, batch)
    batch["reject"][0, 0, B] = True
    assert LocalLogicLoss(0.25)(logits, batch) > base
