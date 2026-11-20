"""A causal model must not see the future: the prediction after a prefix is
the same whether the rest of the case is in the window (training) or not
(prediction)."""
import torch

from models.sasrec import SASRec
from models.bert4rec import BERT4Rec


def _left(x, L):
    return torch.tensor([[0] * (L - len(x)) + x])


def test_sasrec_prefix_prediction_ignores_the_future():
    torch.manual_seed(0)
    m = SASRec(n_items=10, max_len=12, embedding_dim=16, num_blocks=2, num_heads=2, dropout=0.0).eval()
    case = [2, 3, 4, 5, 6, 7]
    with torch.no_grad():
        full = m(_left(case[:-1], 12))[0]              # training window
        for t in range(1, len(case)):
            prefix = m(_left(case[:t], 12))[0, -1]      # prediction after case[:t]
            assert torch.allclose(full[12 - len(case) + t], prefix, atol=1e-5), t


def test_bert4rec_positions_start_at_first_event():
    torch.manual_seed(0)
    m = BERT4Rec(n_items=10, max_len=8, embedding_dim=16, num_blocks=1, num_heads=2, dropout=0.0).eval()
    with torch.no_grad():
        a = m(_left([2, 3, 1], 8))[0, -1]               # MASK after 2 events, window 8
        b = m(_left([2, 3, 1], 5))[0, -1]               # same prefix, other window size
    assert torch.allclose(a, b, atol=1e-5)


def test_gru4rec_prefix_prediction_ignores_the_future():
    """The leading PAD steps must not reach the recurrent state: their number
    depends on the length of the whole case in a training window."""
    from models.gru4rec import GRU4Rec
    torch.manual_seed(0)
    m = GRU4Rec(n_items=10, embedding_dim=16, hidden_dim=16, num_layers=2, dropout=0.0).eval()
    case = [2, 3, 4, 5, 6, 7]
    with torch.no_grad():
        full = m(_left(case[:-1], 12))[0]
        for t in range(1, len(case)):
            prefix = m(_left(case[:t], 12))[0, -1]
            assert torch.allclose(full[12 - len(case) + t], prefix, atol=1e-6), t


def test_every_window_model_ignores_the_future(toy_log):
    """Every configuration that reads the context window, priors and
    competitors included: the scores after a prefix do not depend on the
    events that follow it in a training window (causal models), nor on the
    window size (BERT4Rec, which predicts a MASK appended to the prefix)."""
    from models.variants import COUNT_BASED, DREAM, VARIANTS, backbone, build_model, build_priors
    hps = {"gru4rec": dict(embedding_dim=8, hidden_dim=8, num_layers=2, dropout=0.0),
           "sasrec": dict(embedding_dim=8, num_blocks=2, num_heads=2, dropout=0.0),
           "bert4rec": dict(embedding_dim=8, num_blocks=2, num_heads=2, dropout=0.0)}
    seqs, V = toy_log["sequences"], toy_log["n_items"]
    case = seqs[0][:6]
    checked = []
    for v in VARIANTS:
        if v in COUNT_BASED or v in DREAM:
            continue
        torch.manual_seed(0)
        pri = build_priors(v, seqs, V, toy_log["id2item"], max_hop=4, noise_threshold=0.2)
        m = build_model(v, V, 12, hps[backbone(v)], pri)
        for name, p in m.named_parameters():          # non-zero lambdas: priors really act
            if name.startswith("lambda_pm"):
                torch.nn.init.constant_(p, 0.7)
        m.eval()
        with torch.no_grad():
            if backbone(v) == "bert4rec":
                for t in range(1, len(case)):
                    a = m(_left(case[:t] + [1], 12))[0, -1]
                    b = m(_left(case[:t] + [1], t + 3))[0, -1]
                    assert torch.allclose(a, b, atol=1e-5), (v, t)
            else:
                full = m(_left(case[:-1], 12))[0]
                for t in range(1, len(case)):
                    pref = m(_left(case[:t], 12))[0, -1]
                    assert torch.allclose(full[12 - len(case) + t], pref, atol=1e-5), (v, t)
        checked.append(v)
    assert len(checked) == len([v for v in VARIANTS if v not in COUNT_BASED and v not in DREAM])