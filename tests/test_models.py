"""Every configuration builds, trains one epoch and predicts."""
import pytest
import torch
from torch.utils.data import DataLoader

from data.windows import CausalWindowDataset, MaskedWindowDataset, PrefixEvalDataset, RejectWindowDataset
from training.evaluation import collect_predictions
from training.trainer import train_model
from models.variants import COMPETITORS, COUNT_BASED, DREAM, VARIANTS, backbone, build_model, build_priors
from training.logic_loss import LocalLogicLoss

HPS = {"gru4rec": dict(embedding_dim=8, hidden_dim=8, num_layers=1, dropout=0.1),
       "sasrec": dict(embedding_dim=8, num_blocks=1, num_heads=2, dropout=0.1),
       "bert4rec": dict(embedding_dim=8, num_blocks=2, num_heads=2, dropout=0.1)}


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v not in DREAM])
def test_variant_runs(variant, toy_log):
    torch.manual_seed(0)
    seqs = toy_log["sequences"]
    priors = build_priors(variant, seqs, toy_log["n_items"], toy_log["id2item"],
                          max_hop=4, noise_threshold=0.2)
    arch = backbone(variant)
    model = build_model(variant, toy_log["n_items"], 16, HPS.get(arch, {}), priors)
    is_bert = arch == "bert4rec"
    ev = DataLoader(PrefixEvalDataset(seqs, 16, mask_for_bert=is_bert), batch_size=64)
    if variant in COUNT_BASED:
        model.fit(seqs)
    else:
        loss_fn = None
        if is_bert:
            tr = MaskedWindowDataset(seqs, 16, mask_prob=0.2, last_item_prob=0.5)
        elif variant.endswith("_ltl"):
            tr = RejectWindowDataset(seqs, 16, priors["declare"], toy_log["n_items"])
            loss_fn = LocalLogicLoss(0.25)
        else:
            tr = CausalWindowDataset(seqs, 16)
        train_model(model, DataLoader(tr, batch_size=16, shuffle=True), ev,
                    n_epochs=1, lr=1e-3, patience=1, verbose=False, loss_fn=loss_fn)
    preds = collect_predictions(model, ev)
    assert len(preds) == sum(len(s) - 1 for s in seqs.values())
    assert preds["rank"].min() >= 1
    if "_" in variant and variant not in COUNT_BASED and variant not in COMPETITORS:
        assert any(hasattr(model, a) for a in ("lambda_pm", "lambda_pm_dfg"))