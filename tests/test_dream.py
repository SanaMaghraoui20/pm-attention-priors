"""Theis & Darabi competitors: replay, mask, end-to-end on the same prefixes."""
import numpy as np
import torch

from data.windows import PrefixEvalDataset
from models.dream import DecayReplay, DreamNAP, run_dream


def _times(seqs):
    return {c: [3600.0 * k for k in range(len(s))] for c, s in seqs.items()}


def test_replay_fits_training_cases_and_masks_next(toy_log):
    seqs, ids = toy_log["sequences"], toy_log["item2id"]
    rep = DecayReplay(seqs, _times(seqs), toy_log["id2item"])
    m = rep.rp.start()
    fits = []
    for x in seqs[0]:
        m, _, ok = rep.rp.fire(m, x)
        fits.append(ok)
    assert all(fits)
    f, s = rep.transform(seqs[0], _times(seqs)[0])
    assert f.shape == (len(seqs[0]), rep.n_features) and (s >= 0).all()
    mask = rep.masks(s[:1])[0]                       # after Register: only Assign
    allowed = {i for i in range(len(mask)) if mask[i] == 1.0}
    assert allowed == {ids["Assign"]}
    assert mask[0] == 0 and mask[1] == 0


def test_masked_output_renormalises():
    net = DreamNAP(6, 5, masked=True)
    x, m = torch.rand(3, 6), torch.tensor([[0, 0, 1, 1e-10, 1e-10]] * 3)
    p = net(x, m).exp()
    assert torch.allclose(p.sum(1), torch.ones(3), atol=1e-5)
    assert (p[:, 3:] < 1e-6).all() and (p[:, :2] < 1e-6).all()


def test_run_dream_same_prefixes_as_other_models(toy_log):
    seqs = toy_log["sequences"]
    ids = sorted(seqs)
    train = {c: seqs[c] for c in ids[:40]}
    val = {c: seqs[c] for c in ids[40:50]}
    test = {c: seqs[c] for c in ids[50:]}
    items = lambda d, s: PrefixEvalDataset(d, 16, seed=s).items
    for v in ("dream_nap", "dream_nap_rg"):
        out = run_dream(v, train, val, test, _times(seqs), toy_log["id2item"],
                        items(val, 1), items(test, 2),
                        {"n_epochs": 3, "patience": 2}, 0.2, seed=0)
        p = out["preds"]
        assert list(zip(p["user_id"], p["pos"])) == items(test, 2)
        assert p["rank"].min() >= 1 and out["test_marking_known"] > 0.9


def test_last_batch_of_one_sample(toy_log):
    """Training set of 64k + 1 samples: the last batch holds one sample."""
    seqs = toy_log["sequences"]
    ids = sorted(seqs)
    n = 0
    train = {}
    for c in ids:                                   # grow the training set to 64k + 1 prefixes
        train[c] = seqs[c]
        n += len(seqs[c]) - 1
        if n % 64 == 1:
            break
    assert n % 64 == 1, n
    rest = [c for c in ids if c not in train]
    val = {c: seqs[c] for c in rest[:5]}
    test = {c: seqs[c] for c in rest[5:10]}
    items = lambda d, s: PrefixEvalDataset(d, 16, seed=s).items
    out = run_dream("dream_nap", train, val, test, _times(seqs), toy_log["id2item"],
                    items(val, 1), items(test, 2), {"n_epochs": 1, "patience": 1}, 0.2, seed=0)
    assert len(out["preds"]) == len(items(test, 2))