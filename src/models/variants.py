"""
Model configurations of the paper
=================================

   poprec              PopRec (popularity floor)
   pm_only             first-order directly-follows model, no learning
   gru4rec             GRU4Rec baseline
   gru4rec_pm_output   GRU4Rec + DFG-1hop as output bias
   sasrec              SASRec baseline
   sasrec_dfg1         SASRec + DFG-1hop attention bias
   sasrec_dfgK         SASRec + distance-aware DFG (K hops) attention bias
   sasrec_bp           SASRec + behavioural-profile attention bias
   sasrec_hybrid       SASRec + DFG-Khop + behavioural profile, two lambdas
   bert4rec            BERT4Rec baseline
   bert4rec_dfg1       BERT4Rec + DFG-1hop attention bias

Competitors (docs/competitors/):
   sasrec_ltl          SASRec baseline trained with the Local Logic Loss of
                       Mezini et al. (2026) on Declare constraints mined from
                       the training cases
   gru4rec_ltl         the same loss on the GRU4Rec baseline (the authors
                       evaluate it on an LSTM and on a Transformer)
   sasrec_rgmask       the SASRec baseline with the reachability-graph output
                       mask of Theis & Darabi (2023), in training and at
                       prediction: same network, other injection point
   dream_nap           decay replay mining, Theis & Darabi (2019)
   dream_nap_rg        + reachability-graph masking, Theis & Darabi (2023)
                       (models/dream.py: own features and network, no backbone)

PAIRS lists (enhanced, baseline) for the paired tests. Priors are always built
from the training cases of the fold.
"""

from __future__ import annotations

VARIANTS = [
    "poprec", "pm_only",
    "gru4rec", "gru4rec_pm_output",
    "sasrec", "sasrec_dfg1", "sasrec_dfgK", "sasrec_bp", "sasrec_hybrid",
    "bert4rec", "bert4rec_dfg1",
    "sasrec_ltl", "gru4rec_ltl", "dream_nap", "dream_nap_rg",
    "sasrec_rgmask",
]
COMPETITORS = ["sasrec_ltl", "gru4rec_ltl", "dream_nap", "dream_nap_rg", "sasrec_rgmask"]
LOGIC_LOSS = {"sasrec_ltl", "gru4rec_ltl"}
DREAM = {"dream_nap", "dream_nap_rg"}

PAIRS = [
    ("gru4rec_pm_output", "gru4rec"),
    ("sasrec_dfg1", "sasrec"), ("sasrec_dfgK", "sasrec"),
    ("sasrec_bp", "sasrec"), ("sasrec_hybrid", "sasrec"),
    ("bert4rec_dfg1", "bert4rec"),
    ("sasrec_ltl", "sasrec"),
    ("gru4rec_ltl", "gru4rec"),
    ("dream_nap_rg", "dream_nap"),
    ("sasrec_rgmask", "sasrec"),
]

COUNT_BASED = {"poprec", "pm_only"}


def backbone(variant: str) -> str | None:
    return None if variant in COUNT_BASED or variant in DREAM else variant.split("_")[0]


def build_priors(variant: str, train: dict, n_items: int, id2item: dict,
                 max_hop: int, noise_threshold: float, declare_min_support: float = 0.85) -> dict:
    """Prior matrices needed by `variant`, from the training cases only."""
    from pmpriors.dfg import build_dfg_bias_matrix, build_dfg_bias_tensor

    priors = {}
    if variant in ("gru4rec_pm_output", "sasrec_dfg1", "bert4rec_dfg1"):
        priors["dfg1"] = build_dfg_bias_matrix(train, n_items, id2item=id2item)
    if variant in ("sasrec_dfgK", "sasrec_hybrid"):
        priors["dfgK"] = build_dfg_bias_tensor(train, n_items, max_hop=max_hop, id2item=id2item)
    if variant in ("sasrec_bp", "sasrec_hybrid"):
        from pmpriors.behavioral_profile import build_behavioral_profile_bias_matrix
        priors["bp"] = build_behavioral_profile_bias_matrix(
            train=train, id2item=id2item, n_items=n_items,
            noise_threshold=noise_threshold, verbose=False)
    if variant == "sasrec_rgmask":
        from models.rgmask import ReachabilityMask
        priors["rgmask"] = ReachabilityMask(train, id2item, noise_threshold=noise_threshold)
    if variant in LOGIC_LOSS:
        from pmpriors.declare import mine_constraints
        priors["declare"] = mine_constraints(train, id2item, min_support=declare_min_support)
    return priors


def build_model(variant: str, n_items: int, max_len: int, hps: dict, priors: dict):
    """Instantiate `variant`. `hps` are the model hyperparameters only (no lr,
    no mask_prob). Count-based baselines are returned unfitted."""
    if variant == "poprec":
        from models.baselines import PopRec
        return PopRec(n_items)
    if variant == "pm_only":
        from models.baselines import PMOnly
        return PMOnly(n_items)

    if variant.startswith("gru4rec"):
        from models.gru4rec import GRU4Rec
        kw = {}
        if variant == "gru4rec_pm_output":
            kw = dict(use_pm=True, pm_mode="output", dfg_bias_matrix=priors["dfg1"])
        return GRU4Rec(n_items=n_items, **hps, **kw)

    if variant.startswith("sasrec"):
        from models.sasrec import SASRec
        kw = {
            "sasrec": {},
            "sasrec_dfg1": dict(use_pm=True, pm_mode="attention", dfg_bias_matrix=priors.get("dfg1")),
            "sasrec_dfgK": dict(use_pm=True, pm_mode="attention", dfg_bias_matrix=priors.get("dfgK")),
            # The BP matrix goes through the DFG slot for implementation
            # reasons; its weight is reported as lambda_pm_tree (see lambdas()).
            "sasrec_bp": dict(use_pm=True, pm_mode="attention", dfg_bias_matrix=priors.get("bp")),
            "sasrec_ltl": {},             # same network; the loss differs
            "sasrec_rgmask": {},          # same network; masked output (below)
            "sasrec_hybrid": dict(use_pm=True, pm_mode="attention", dfg_bias_matrix=priors.get("dfgK"),
                                  tree_bias_matrix=priors.get("bp"), hybrid=True),
        }[variant]
        model = SASRec(n_items=n_items, max_len=max_len, **hps, **kw)
        if variant == "sasrec_rgmask":
            from models.rgmask import MaskedSASRec
            model = MaskedSASRec(model, priors["rgmask"])
        return model

    if variant.startswith("bert4rec"):
        from models.bert4rec import BERT4Rec
        kw = {}
        if variant == "bert4rec_dfg1":
            kw = dict(use_pm=True, pm_mode="attention", dfg_bias_matrix=priors["dfg1"])
        return BERT4Rec(n_items=n_items, max_len=max_len, **hps, **kw)

    raise ValueError(f"unknown variant {variant!r}")


def lambdas(variant: str, model) -> dict:
    """Learned prior weights, named by the prior they scale."""
    out = {"lambda_pm_dfg": None, "lambda_pm_tree": None}
    if hasattr(model, "lambda_pm_dfg"):
        out["lambda_pm_dfg"] = float(model.lambda_pm_dfg.item())
    elif hasattr(model, "lambda_pm"):
        out["lambda_pm_dfg"] = float(model.lambda_pm.item())
    if hasattr(model, "lambda_pm_tree"):
        out["lambda_pm_tree"] = float(model.lambda_pm_tree.item())
    if variant.endswith("_bp"):
        out["lambda_pm_tree"], out["lambda_pm_dfg"] = out["lambda_pm_dfg"], None
    return out