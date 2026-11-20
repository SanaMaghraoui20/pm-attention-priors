"""
Supports:
  - No PM prior (baseline)
  - Single PM prior: matrix (V, V)  -> same bias at every pair distance
                     tensor (K, V, V) -> distance-aware bias (hop k uses B_k)
  - Hybrid PM priors (e.g. DFG + behavioural profile) with independent
    learnable scalars lambda_pm_dfg / lambda_pm_tree
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMTransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.bidirectional = bidirectional

        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                        
        pad_mask: torch.Tensor | None = None,   
        attn_bias: torch.Tensor | None = None,  
    ) -> torch.Tensor:
        B, T, D = x.shape

        h = self.ln1(x)
        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_bias is not None:
            scores = scores + attn_bias.unsqueeze(1)

        if not self.bidirectional:
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)

        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        x   = x + self.dropout(out)

        h = self.ln2(x)
        x = x + self.dropout(self.ffn(h))
        return x


def _lookup_bias(
    M: torch.Tensor,          
    input_ids: torch.Tensor,  
    symmetric: bool,
) -> torch.Tensor:
    """
    Build the (B, T, T) pairwise bias from a prior matrix or hop tensor.

    For a matrix (V, V):
        bias[b, t, s] = M[x_s, x_t]                (same at all distances)

    For a tensor (K, V, V):  distance-aware
        d = clamp(t - s, 1, K)
        bias[b, t, s] = M[d-1, x_s, x_t]           (for s < t)
        bias[b, t, t] = M[0,   x_t, x_t]           (diagonal uses hop 1)
        For s > t (bidirectional attention), the mirrored distance is used
        with source/target swapped so the direction of the underlying process
        is preserved:  bias[b, t, s] = M[d-1, x_t, x_s], d = s - t.
    """
    B, T = input_ids.shape
    device = input_ids.device

    # We define bias[b, t, s]; index t = query position (dim 1), s = key (dim 2)
    tgt = input_ids.unsqueeze(2).expand(B, T, T)      # x_t along dim 1 broadcast
    src = input_ids.unsqueeze(1).expand(B, T, T)      # x_s along dim 2 broadcast
    # tgt[b, t, s] = input_ids[b, t] ; src[b, t, s] = input_ids[b, s]

    if M.dim() == 2:
        fwd = M[src, tgt]                              # (B, T, T): x_s -> x_t
        if not symmetric:
            return fwd
        # Bidirectional attention: for keys in the future (s > t) the real
        # temporal transition is x_t -> x_s, so read M in that direction
        # instead of averaging both (averaging destroys the directional
        # information which is the main signal of a DFG).
        bwd = M[tgt, src]                              # x_t -> x_s
        pos = torch.arange(T, device=device)
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).unsqueeze(0).expand(B, T, T)
        return torch.where(dist >= 0, fwd, bwd)

    # Distance-aware (K, V, V)
    K = M.shape[0]
    pos = torch.arange(T, device=device)
    dist = pos.unsqueeze(1) - pos.unsqueeze(0)         # dist[t, s] = t - s
    dist = dist.unsqueeze(0).expand(B, T, T)           # (B, T, T)

    hop_fwd = dist.clamp(min=1, max=K) - 1             # for s <= t
    hop_bwd = (-dist).clamp(min=1, max=K) - 1          # for s > t

    fwd = M[hop_fwd, src, tgt]                          # bias when source precedes target
    if symmetric:
        bwd = M[hop_bwd, tgt, src]                      # mirrored direction
        contrib = torch.where(dist >= 0, fwd, bwd)
        # blend both directions where both are defined (soft symmetrisation
        # is NOT applied here: direction is preserved by construction)
    else:
        # causal models only ever see s <= t; fill s > t arbitrarily (masked)
        contrib = fwd

    return contrib


def build_attn_bias(
    input_ids: torch.Tensor,                 # (B, T)
    dfg_bias:  torch.Tensor | None,          # (V, V) or (K, V, V)
    tree_bias: torch.Tensor | None,          # (V, V) or (K, V, V)
    lambda_dfg:  torch.nn.Parameter | None,
    lambda_tree: torch.nn.Parameter | None,
    symmetric: bool = False,
) -> torch.Tensor:
    """
    Combine up to two priors into the additive attention bias:

        bias = lambda_dfg * lookup(dfg_bias) + lambda_tree * lookup(tree_bias)
    """
    B, T = input_ids.shape
    bias = torch.zeros(B, T, T, device=input_ids.device)

    for M, lam in [(dfg_bias, lambda_dfg), (tree_bias, lambda_tree)]:
        if M is None or lam is None:
            continue
        bias = bias + lam * _lookup_bias(M, input_ids, symmetric)

    return bias
