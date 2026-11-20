"""
Evaluation
==========

"""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    k_values=(1, 5, 10),
    device: str = "cpu",
) -> dict:
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            pass

    max_k = max(k_values)
    max_k_eff = None   

    n = 0
    hits = {k: 0 for k in k_values}
    ndcg = {k: 0.0 for k in k_values}
    rr_sum = 0.0

    all_preds = []
    all_targets = []

    for batch in data_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)          

        scores = model.predict(inputs)                
        B = targets.shape[0]

        if max_k_eff is None:
            max_k_eff = min(max_k, scores.shape[1])

        # Top-max_k ranking
        topk = scores.topk(max_k_eff, dim=1).indices  

        
        target_scores = scores.gather(1, targets.unsqueeze(1))         
        rank = (scores > target_scores).sum(dim=1) + 1                 
        rr_sum += (1.0 / rank.float()).sum().item()

        for k in k_values:
            kk = min(k, max_k_eff)
            in_topk = (topk[:, :kk] == targets.unsqueeze(1))            
            hits[k] += in_topk.any(dim=1).sum().item()
            # NDCG@k: 1/log2(pos+1) if hit at position pos (1-indexed)
            pos = in_topk.float().argmax(dim=1) + 1                     
            gain = torch.where(
                in_topk.any(dim=1),
                1.0 / torch.log2(pos.float() + 1.0),
                torch.zeros_like(pos, dtype=torch.float),
            )
            ndcg[k] += gain.sum().item()

        all_preds.append(topk[:, 0].cpu())
        all_targets.append(targets.cpu())
        n += B

    preds = torch.cat(all_preds)
    targs = torch.cat(all_targets)

    # ---- per-class F1 ----
    classes = sorted(set(targs.tolist()) | set(preds.tolist()))
    f1_per_class = {}
    support = {}
    for c in classes:
        tp = ((preds == c) & (targs == c)).sum().item()
        fp = ((preds == c) & (targs != c)).sum().item()
        fn = ((preds != c) & (targs == c)).sum().item()
        sup = (targs == c).sum().item()
        support[c] = sup
        if tp == 0:
            f1_per_class[c] = 0.0
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1_per_class[c] = 2 * prec * rec / (prec + rec)

    # macro over classes that actually appear as targets
    target_classes = [c for c in classes if support[c] > 0]
    macro_f1 = (sum(f1_per_class[c] for c in target_classes) /
                max(len(target_classes), 1))
    total_support = sum(support[c] for c in target_classes)
    weighted_f1 = (sum(f1_per_class[c] * support[c] for c in target_classes) /
                   max(total_support, 1))

    out = {
        "accuracy":    hits[1] / n if 1 in k_values else None,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "mrr":         rr_sum / n,
    }
    for k in k_values:
        out[f"HR@{k}"] = hits[k] / n
        out[f"NDCG@{k}"] = ndcg[k] / n
   
    if 1 in k_values:
        out["NDCG@1"] = out["HR@1"]

    return out


def format_metrics(m: dict) -> str:
    
    parts = [
        f"acc={m['accuracy']:.4f}",
        f"macroF1={m['macro_f1']:.4f}",
        f"wF1={m['weighted_f1']:.4f}",
        f"MRR={m['mrr']:.4f}",
    ]
    for key in sorted(k for k in m if k.startswith("HR@")):
        if key != "HR@1":
            parts.append(f"{key}={m[key]:.4f}")
    for key in sorted(k for k in m if k.startswith("NDCG@")):
        if key not in ("NDCG@1",):
            parts.append(f"{key}={m[key]:.4f}")
    return "  ".join(parts)


def _as_list(x) -> list:
    
    return x.tolist() if torch.is_tensor(x) else list(x)


@torch.no_grad()
def collect_predictions(model, loader, device: str = "cpu") -> pd.DataFrame:
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "to"):
        try:
            model.to(device)
        except Exception:
            pass
    cols = {"user_id": [], "pos": [], "target": [], "pred": [], "rank": []}
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        scores = model.predict(inputs).float()
        t = scores.gather(1, targets.unsqueeze(1))
        higher = (scores > t).sum(1)
        tied = (scores == t).sum(1) - 1
        rank = higher.float() + 1.0 + tied.float() / 2.0
        cols["user_id"].extend(_as_list(batch["user_id"]))
        cols["pos"].extend(_as_list(batch["pos"]))
        cols["target"].extend(targets.cpu().tolist())
        cols["pred"].extend(scores.argmax(1).cpu().tolist())
        cols["rank"].extend(rank.cpu().tolist())
    return pd.DataFrame(cols)


def _per_row(df: pd.DataFrame, k_values=(5, 10)) -> pd.DataFrame:
    out = pd.DataFrame({"user_id": df["user_id"]})
    r = df["rank"].to_numpy(dtype=float)
    out["accuracy"] = (df["pred"].to_numpy() == df["target"].to_numpy()).astype(float)
    out["mrr"] = 1.0 / r
    for k in k_values:
        hit = r <= k
        out[f"HR@{k}"] = hit.astype(float)
        out[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(r + 1.0), 0.0)
    return out


def macro_f1(df: pd.DataFrame) -> float:
    y, p = df["target"].to_numpy(), df["pred"].to_numpy()
    f1s = []
    for c in np.unique(y):            
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def metrics_from_predictions(df: pd.DataFrame, k_values=(5, 10)) -> dict:
    rows = _per_row(df, k_values)
    metric_cols = [c for c in rows.columns if c != "user_id"]
    micro = rows[metric_cols].mean()
    learner = rows.groupby("user_id")[metric_cols].mean().mean()
    out = {"n_predictions": int(len(df)), "n_learners": int(df["user_id"].nunique()),
           "macro_f1": macro_f1(df)}
    for c in metric_cols:
        out[f"{c}"] = float(micro[c])
        out[f"learner_{c}"] = float(learner[c])
    return out
