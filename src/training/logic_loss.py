"""
Local Logic Loss of Mezini et al. (2026)
========================================
A. Mezini, E. Umili, I. Donadello, F. M. Maggi, M. Mancanelli, F. Patrizi.
"Neuro-Symbolic Predictive Process Monitoring." implementation of the authors' code :

    L = alpha * CE_valid + (1 - alpha) * mean_t -log(1 - P_viol(t) + 1e-6)
    P_viol(t) = sum of the predicted probabilities of the activities that
                would violate a constraint after the prefix (batch["reject"])
    CE_valid  = cross-entropy averaged over the steps whose TRUE next activity
                does not violate a constraint (as in the authors' code)

Differences with the authors' code: PAD positions are excluded from both
terms (their batches hold no padding), and the step penalty is averaged over
the non-PAD steps.
"""

import torch
import torch.nn.functional as F


class LocalLogicLoss:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha

    def __call__(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        targets, reject = batch["target"], batch["reject"]          
        real = targets != 0
        ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        target_rejected = reject.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        keep = (real & ~target_rejected).float()
        ce_valid = (ce * keep).sum() / (keep.sum() + 1e-6)
        probs = torch.softmax(logits, dim=-1)
        invalid = (probs * reject.float()).sum(-1)
        penalty = (-torch.log(1.0 - invalid + 1e-6) * real.float()).sum() / (real.float().sum() + 1e-6)
        return self.alpha * ce_valid + (1 - self.alpha) * penalty
