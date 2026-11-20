"""
SASRec — Causal transformer for sequential recommendation
==========================================================
Supports:
  - use_pm=False              : plain SASRec baseline
  - use_pm=True, pm_mode="attention", hybrid=False : DFG bias in attention
  - use_pm=True, pm_mode="attention", hybrid=True  : DFG + Tree bias (2 priors)
  - use_pm=True, pm_mode="output"  : output score bias (DFG only for now)
"""

import torch
import torch.nn as nn
from .transformer_block import PMTransformerBlock, build_attn_bias


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_len: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        use_pm: bool = False,
        pm_mode: str = "attention",           # "attention" or "output"
        dfg_bias_matrix: torch.Tensor | None = None,
        tree_bias_matrix: torch.Tensor | None = None,
        hybrid: bool = False,
    ):
        super().__init__()
        self.n_items       = n_items
        self.max_len       = max_len
        self.embedding_dim = embedding_dim
        self.use_pm        = use_pm
        self.pm_mode       = pm_mode
        self.hybrid        = hybrid

        # Embeddings
        self.item_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, embedding_dim)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            PMTransformerBlock(embedding_dim, num_heads, dropout, bidirectional=False)
            for _ in range(num_blocks)
        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.dropout  = nn.Dropout(dropout)

        # PM bias matrices 
        self.has_dfg  = use_pm and dfg_bias_matrix is not None
        self.has_tree = use_pm and hybrid and tree_bias_matrix is not None

        if self.has_dfg:
            self.register_buffer("dfg_bias", dfg_bias_matrix.float())
        else:
            self.dfg_bias = None

        if self.has_tree:
            self.register_buffer("tree_bias", tree_bias_matrix.float())
        else:
            self.tree_bias = None

        # Learnable weights
        if use_pm and pm_mode == "attention":
            if self.has_dfg:
                self.lambda_pm_dfg = nn.Parameter(torch.zeros(1))
            if self.has_tree:
                self.lambda_pm_tree = nn.Parameter(torch.zeros(1))
            # Backward compat: expose a `lambda_pm` alias for the DFG λ
            if self.has_dfg and not self.hybrid:
                self.lambda_pm = self.lambda_pm_dfg
        elif use_pm and pm_mode == "output":
            self.lambda_pm = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T) with padding_idx=0
        Returns:   (B, T, V) logits over the item vocabulary
        """
        B, T = input_ids.shape
        device = input_ids.device

        pad_mask = (input_ids == 0)                                   # (B, T)
        # Positions counted from the first event of the window, not from its
        # left edge: with left padding, edge-based positions encode the
        # distance to the END of the case, which a training window reveals
        # (the whole case) but a prefix at prediction time does not. Start-
        # based positions give an event the same position in both.
        positions = ((input_ids != 0).long().cumsum(1) - 1).clamp(min=0)

        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # PM attention bias
        attn_bias = None
        if self.use_pm and self.pm_mode == "attention":
            attn_bias = build_attn_bias(
                input_ids,
                dfg_bias  = self.dfg_bias if self.has_dfg else None,
                tree_bias = self.tree_bias if self.has_tree else None,
                lambda_dfg  = self.lambda_pm_dfg if self.has_dfg else None,
                lambda_tree = self.lambda_pm_tree if self.has_tree else None,
                symmetric = False,
            )

        for block in self.blocks:
            x = block(x, pad_mask=pad_mask, attn_bias=attn_bias)

        x = self.ln_final(x)

        # Logits = x @ item_emb.weight.T
        logits = torch.matmul(x, self.item_emb.weight.T)              # (B, T, V)

        # PM output bias (only from DFG for now)
        if self.use_pm and self.pm_mode == "output" and self.has_dfg:
            # For each position t, add lambda * DFG[input_ids[b, t], :]
            output_bias = self.dfg_bias[input_ids]                    # (B, T, V)
            logits = logits + self.lambda_pm * output_bias

        return logits

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(input_ids)
        last_scores = logits[:, -1, :]
        last_scores[:, 0] = float("-inf")  # PAD
        last_scores[:, 1] = float("-inf")  # MASK
        return last_scores