import torch

from data.windows import (CausalWindowDataset, MaskedWindowDataset,
                                   PrefixEvalDataset, _windows)


def test_windows_cover_every_transition():
    seq = list(range(2, 452))
    pairs = {(w[i], w[i + 1]) for w in _windows(seq, 201, 200) for i in range(len(w) - 1)}
    assert len(pairs) == len(seq) - 1


def test_causal_target_is_input_shifted():
    ds = CausalWindowDataset({0: list(range(2, 40))}, max_len=16)
    for k in range(len(ds)):
        x = ds[k]
        real = x["input"] > 0
        assert torch.equal(x["input"][real][1:], x["target"][real][:-1])


def test_prefix_eval_one_sample_per_position():
    seqs = {0: [2, 3, 4, 5], 1: [3, 3, 2]}
    ds = PrefixEvalDataset(seqs, max_len=8)
    assert len(ds) == (4 - 1) + (3 - 1)
    for k in range(len(ds)):
        uid, t = ds.items[k]
        x = ds[k]
        assert x["target"].item() == seqs[uid][t]
        assert x["input"][-1].item() == seqs[uid][t - 1]


def test_bert_eval_ends_with_mask():
    ds = PrefixEvalDataset({0: [2, 3, 4, 5]}, max_len=8, mask_for_bert=True)
    assert all(ds[k]["input"][-1].item() == 1 for k in range(len(ds)))


def test_masked_labels_only_at_mask_and_last_item_mode():
    ds = MaskedWindowDataset({0: list(range(2, 30))}, max_len=32, mask_prob=0.3,
                             seed=1, last_item_prob=1.0)
    x = ds[0]
    assert torch.equal((x["target"] > 0), (x["input"] == 1))
    assert (x["target"] > 0).sum() == 1 and x["input"][-1].item() == 1
