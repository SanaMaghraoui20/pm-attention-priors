"""
Trainer
=============================================================
Works with any model that:
  - returns logits of shape (B, L, n_items) from forward(input)
  - has a .predict(input) method returning (B, n_items)

Used for GRU4Rec, SASRec, and (with a mask flag) BERT4Rec.

Training objective
------------------
Cross-entropy at every non-pad position:
    loss = CrossEntropy( logits.flatten(0, 1), target.flatten() )
The PAD token (id=0) is ignored in the loss via `ignore_index=0`.

Early stopping
--------------
The trainer tracks a validation metric (NDCG@5 in the experiments) and stops training
when it does not improve for `patience` consecutive evaluations.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import evaluate, format_metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    lambda_lr_mult: float = 1.0,
    weight_decay: float = 1e-6,
    k_values: list[int] = [1, 5, 10],
    early_stop_metric: str = "NDCG@5",
    patience: int = 5,
    device: str = "cpu",
    verbose: bool = True,
    loss_fn=None,
) -> dict:
    """
    Train `model` on `train_loader` and validate on `val_loader`.

    Returns
    -------
    dict with keys:
      - "best_state"   : best model state_dict (according to val metric)
      - "best_epoch"   : epoch at which the best metric was reached
      - "best_metrics" : the val metrics at that epoch
      - "history"      : list of {epoch, train_loss, val_metrics}
    """
    model = model.to(device)
    # Exclude the PM prior weights (lambda_pm*) from weight decay:
    # decay would systematically pull lambda toward 0, biasing its role as
    # an interpretable diagnostic of the prior's usefulness. Lambda must be
    # driven by the data signal only.
    lambda_params = [p for n, p in model.named_parameters()
                     if n.startswith("lambda_pm") and p.requires_grad]
    other_params  = [p for n, p in model.named_parameters()
                     if not n.startswith("lambda_pm") and p.requires_grad]
    optimizer = torch.optim.Adam(
        [
            {"params": other_params,  "weight_decay": weight_decay,
             "lr": lr},
            {"params": lambda_params, "weight_decay": 0.0,
             "lr": lr * lambda_lr_mult},
        ],
    )
    # loss_fn(logits, batch) -> loss; default: cross-entropy ignoring PAD
    if loss_fn is None:
        ce = nn.CrossEntropyLoss(ignore_index=0)
        loss_fn = lambda logits, batch: ce(logits.reshape(-1, logits.shape[-1]),
                                           batch["target"].reshape(-1))

    history = []
    best_score = -float("inf")
    best_state = None
    best_epoch = -1
    best_metrics = None
    epochs_since_improve = 0

    for epoch in range(1, n_epochs + 1):
        # --- training ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(batch["input"])              # (B, L, V)
            loss = loss_fn(logits, batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # --- validation ---
        model.eval()
        val_metrics = evaluate(model, val_loader, k_values=k_values, device=device)

        # Trace lambda values per epoch (for trajectory plots / diagnostics)
        lambdas = {n: float(p.item()) for n, p in model.named_parameters()
                   if n.startswith("lambda_pm")}

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_metrics": val_metrics,
            "lambdas": lambdas,
        })

        if verbose:
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | {format_metrics(val_metrics)}")

        # --- early stopping ---
        current_score = val_metrics[early_stop_metric]
        if current_score > best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = val_metrics
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement in {early_stop_metric} for {patience} epochs).")
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_state":   best_state,
        "best_epoch":   best_epoch,
        "best_metrics": best_metrics,
        "history":      history,
    }