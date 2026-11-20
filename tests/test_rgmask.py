"""Reachability-graph mask on SASRec (controlled Theis & Darabi 2023)."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.windows import AllowWindowDataset, CausalWindowDataset, PrefixEvalDataset
from models.rgmask import MaskedLoss, MaskedSASRec, ReachabilityMask
from models.sasrec import SASRec
from training.evaluation import collect_predictions
from training.trainer import train_model


def test_mask_after_first_event(toy_log):
    seqs, ids = toy_log["sequences"], toy_log["item2id"]
    rg = ReachabilityMask(seqs, toy_log["id2item"])
    row = rg.for_prefix((ids["Register"],))
    assert {i for i in range(len(row)) if row[i] == 1.0} == {ids["Assign"]}
    assert row[0] == 0 and row[1] == 0


def test_window_masks_align_with_targets(toy_log):
    seqs = toy_log["sequences"]
    rg = ReachabilityMask(seqs, toy_log["id2item"])
    long = {0: (seqs[0] * 6)[:30]}
    ds, base = AllowWindowDataset(long, 8, rg), CausalWindowDataset(long, 8)
    assert len(ds) == len(base) > 2
    full = rg.state_ids(long[0])
    for i in range(len(ds)):
        item, st = ds[i], ds.starts[i]
        n = int((item["target"] != 0).sum())
        assert np.array_equal(item["allow"][-n:].numpy(), rg.table[full[st:st + n]])
        assert (item["allow"][:-n].numpy() == rg.table[rg.unknown]).all()


def test_masked_training_and_prediction(toy_log):
    torch.manual_seed(0)
    seqs, ids, V = toy_log["sequences"], toy_log["item2id"], toy_log["n_items"]
    rg = ReachabilityMask(seqs, toy_log["id2item"])
    m = MaskedSASRec(SASRec(n_items=V, max_len=16, embedding_dim=8, num_blocks=1, num_heads=2,
                            dropout=0.0), rg)
    ev = DataLoader(PrefixEvalDataset(seqs, 16), batch_size=64)
    train_model(m, DataLoader(AllowWindowDataset(seqs, 16, rg), batch_size=16, shuffle=True), ev,
                n_epochs=1, lr=1e-3, patience=1, verbose=False, loss_fn=MaskedLoss())
    s = m.predict(torch.tensor([[0] * 15 + [ids["Register"]]]))[0].exp()
    assert s[ids["Assign"]] > 0.999                       
    assert s[ids["Close"]] < 1e-6
    p = collect_predictions(m, ev)
    assert len(p) == sum(len(x) - 1 for x in seqs.values()) and p["rank"].min() >= 1