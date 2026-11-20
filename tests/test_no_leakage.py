"""The priors must depend on the training cases only."""
import torch

from data.splits import benchmark_split, subset
from models.variants import build_priors


def test_priors_ignore_held_out_cases(toy_log):
    split = benchmark_split(toy_log["sequences"], fold=0)
    train = subset(toy_log["sequences"], split["train"])
    a = build_priors("sasrec_hybrid", train, toy_log["n_items"], toy_log["id2item"],
                     max_hop=4, noise_threshold=0.2)
    
    seqs = dict(toy_log["sequences"])
    for c in split["val"] + split["test"]:
        seqs[c] = [2] * 30
    train2 = subset(seqs, split["train"])
    b = build_priors("sasrec_hybrid", train2, toy_log["n_items"], toy_log["id2item"],
                     max_hop=4, noise_threshold=0.2)
    for k in a:
        assert torch.equal(a[k], b[k])


def test_special_tokens_are_neutral(toy_log):
    p = build_priors("sasrec_hybrid", toy_log["sequences"], toy_log["n_items"],
                     toy_log["id2item"], max_hop=4, noise_threshold=0.2)
    for M in (p["dfgK"][0], p["bp"]):
        assert torch.all(M[:2, :] == 0) and torch.all(M[:, :2] == 0)
