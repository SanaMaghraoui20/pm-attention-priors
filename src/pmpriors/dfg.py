"""
DFG-based Prior Matrices
=========================
Builds bias matrices from the Directly-Follows Graph of the training log.

Two constructions are provided:

1. build_dfg_bias_matrix(...)   -> (V, V)
   The classic 1-hop prior: B = log P, where P is the Laplace-smoothed
   transition matrix. 

2. build_dfg_bias_tensor(...)   -> (K, V, V)
   Distance-aware prior: B_k = log(P^k) for k = 1..K, where (P^k)[i, j] is
   the probability that j occurs exactly k steps after i under the Markov
   dynamics of the log. Intended for pairwise attention biases where the
   (source, target) pair sits at distance d = t - s:

       A_PM[b, t, s] = lambda * B_{min(t-s, K)}[x_s, x_t]

   Rationale: applying the 1-hop matrix to non-adjacent pairs conflates
   "directly follows" with "eventually follows". Matrix powers give the
   semantically correct k-step transition probabilities. Beyond a small K,
   P^k approaches the stationary distribution and carries no information
   about the source item, hence the truncation.

Both constructions optionally z-standardise each matrix over valid
(non-special) entries so that priors of different nature (DFG, behavioural
profile, ...) live on comparable scales and the learnable lambda only has to
encode direction and magnitude of usefulness, not unit conversion.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


PAD_ID = 0
MASK_ID = 1



def _count_matrix(
    train: Dict[int, List[int]],
    n_items: int,
) -> torch.Tensor:
    """Raw directly-follows counts C[i, j] over the training sequences."""
    C = torch.zeros(n_items, n_items, dtype=torch.float64)
    for seq in train.values():
        for a, b in zip(seq[:-1], seq[1:]):
            C[a, b] += 1.0
    return C


def _transition_matrix(
    C: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Laplace-smoothed row-stochastic transition matrix P."""
    V = C.shape[0]
    smoothed = C + alpha
    P = smoothed / smoothed.sum(dim=1, keepdim=True)
    return P


def _standardize(B: torch.Tensor, id2item: Optional[Dict[int, str]] = None) -> torch.Tensor:
    """
    Z-normalise over valid entries. 
    """
    mask = torch.ones_like(B, dtype=torch.bool)
    special_ids = []
    if id2item is not None:
        for k, v in id2item.items():
            if v in {"<PAD>", "<MASK>"}:
                special_ids.append(k)
    else:
        special_ids = [PAD_ID, MASK_ID]
    for k in special_ids:
        if k < B.shape[0]:
            mask[k, :] = False
            mask[:, k] = False

    vals = B[mask]
    mu = vals.mean()
    sigma = vals.std().clamp(min=1e-6)
    B = (B - mu) / sigma
    B[~mask] = 0.0
    return B



def build_dfg_bias_matrix(
    train: Dict[int, List[int]],
    n_items: int,
    alpha: float = 1.0,
    standardize: bool = True,
    id2item: Optional[Dict[int, str]] = None,
) -> torch.Tensor:
    """
    Classic 1-hop DFG prior: B = log P (Laplace-smoothed).

    Parameters
    ----------
    train        : {user_id: [item_id, ...]} training sequences.
    n_items      : vocab size incl. PAD/MASK.
    alpha        : Laplace smoothing constant.
    standardize  : z-normalise over valid entries (recommended; set False to
                   reproduce earlier unstandardised experiments).
    id2item      : optional reverse vocab, used to identify special tokens.

    """
    C = _count_matrix(train, n_items)
    P = _transition_matrix(C, alpha=alpha)
    B = torch.log(P)
    if standardize:
        B = _standardize(B, id2item)
    return B.float()


def build_dfg_bias_tensor(
    train: Dict[int, List[int]],
    n_items: int,
    max_hop: int = 8,
    alpha: float = 1.0,
    standardize: bool = True,
    id2item: Optional[Dict[int, str]] = None,
) -> torch.Tensor:
    """
    Distance-aware DFG prior: B_k = log(P^k) for k = 1..max_hop.

    Use in attention as:
        A_PM[b, t, s] = lambda * B[min(t - s, max_hop) - 1, x_s, x_t]

    Parameters
    ----------
    max_hop : number of hop-specific matrices to precompute. Beyond this
              distance the last matrix is reused (P^k converges to the
              stationary distribution and stops being informative).
    """
    C = _count_matrix(train, n_items)
    P = _transition_matrix(C, alpha=alpha)   

    mats = []
    Pk = P.clone()
    for k in range(1, max_hop + 1):
        Bk = torch.log(Pk.clamp(min=1e-12))
        if standardize:
            Bk = _standardize(Bk, id2item)
        mats.append(Bk.float())
        if k < max_hop:
            Pk = Pk @ P   # P^(k+1)

    return torch.stack(mats, dim=0)   


