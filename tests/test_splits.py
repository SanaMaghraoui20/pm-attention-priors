from sklearn.model_selection import KFold

from data.splits import benchmark_split


def _seqs(n=88):
    return {i: [2, 3] for i in range(n)}


def test_every_case_tested_exactly_once():
    seen = []
    for f in range(5):
        seen += benchmark_split(_seqs(), fold=f)["test"]
    assert sorted(seen) == list(range(88))


def test_train_val_test_disjoint():
    for f in range(5):
        s = benchmark_split(_seqs(), fold=f)
        assert not set(s["train"]) & set(s["test"])
        assert not set(s["val"]) & set(s["test"])
        assert not set(s["train"]) & set(s["val"])


def test_matches_benchmark_index_logic():
    ids = sorted(_seqs())
    tr, te = list(KFold(5, shuffle=True, random_state=42).split(ids))[2]
    cut = round(len(tr) * 0.8)
    s = benchmark_split(_seqs(), fold=2)
    assert s["test"] == sorted(ids[i] for i in te)
    assert s["val"] == sorted(ids[i] for i in tr[cut:])


def test_learning_curve_subsets_are_nested_and_keep_test():
    subsets = [benchmark_split(_seqs(), fold=0, train_frac=p) for p in (0.1, 0.25, 0.5, 1.0)]
    for a, b in zip(subsets, subsets[1:]):
        assert set(a["train"]) <= set(b["train"])
        assert a["test"] == b["test"] and a["val"] == b["val"]
