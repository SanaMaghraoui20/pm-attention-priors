"""
GRU4Rec — Gated Recurrent Unit for Session-based Recommendation
==========================================================================
Reference: Hidasi et al., 2016 (GRU4Rec, ICLR).

This file provides a unified implementation of:
  - GRU4Rec baseline                   (use_pm=False)
  - GRU4Rec + PM as an output bias     (use_pm=True, pm_mode="output")

NOTE: GRU4Rec does not have an attention mechanism, so pm_mode="attention"
is not supported. PM is injected as a late-fusion bias on the output
logits, based on the last observed item.

"""



import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_pm: bool = False,
        pm_mode: str = "output",
        dfg_bias_matrix: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_pm = use_pm
        self.pm_mode = pm_mode

        if use_pm:
            if dfg_bias_matrix is None:
                raise ValueError("use_pm=True requires dfg_bias_matrix")
            if pm_mode != "output":
                raise ValueError(
                    f"GRU4Rec only supports pm_mode='output' (no attention), got {pm_mode!r}"
                )
            assert dfg_bias_matrix.shape == (n_items, n_items)
            self.register_buffer("dfg_bias", dfg_bias_matrix.float())
            self.lambda_pm = nn.Parameter(torch.tensor(0.0))

        self.item_embedding = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, n_items)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # Sequences are LEFT-padded. A recurrent network run over the leading
        # PAD steps changes its hidden state, so the number of PAD steps, which
        # in a training window depends on the length of the whole case, would
        # leak into the prediction. The events are therefore moved to the start
        # (right padding) before the GRU, and the outputs moved back after it:
        # the output at an event depends only on the events before it.
        B, L = input_seq.shape
        shift = L - (input_seq != 0).sum(1, keepdim=True)          
        ar = torch.arange(L, device=input_seq.device).unsqueeze(0)
        to_front = (ar + shift) % L                                 
        back = (ar - shift) % L                                     
        emb = self.item_embedding(input_seq.gather(1, to_front))    
        emb = self.dropout(emb)
        gru_out, _ = self.gru(emb)                                  
        gru_out = gru_out.gather(1, back.unsqueeze(-1).expand(-1, -1, gru_out.shape[-1]))
        gru_out = self.dropout(gru_out)
        scores = self.output(gru_out)                               

        if self.use_pm and self.pm_mode == "output":
            pm_logits = self.dfg_bias[input_seq]           
            scores = scores + self.lambda_pm * pm_logits

        return scores

    @torch.no_grad()
    def predict(self, input_seq: torch.Tensor) -> torch.Tensor:
        self.eval()
        scores = self.forward(input_seq)
        last_scores = scores[:, -1, :]
        last_scores[:, 0] = float("-inf")
        last_scores[:, 1] = float("-inf")
        return last_scores