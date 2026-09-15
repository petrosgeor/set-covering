"""Stop, exchange-count, and value heads for emitter observations."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class PolicyHeads(nn.Module):
    """Project global embeddings into choice logits and net return estimates."""

    def __init__(self, *, model_dim: int, max_exchanges: int) -> None:
        """Build independent stop, count, and value projections."""
        super().__init__()
        self.max_exchanges = max_exchanges
        self.stop_head = nn.Linear(model_dim, 2)
        self.count_head = nn.Linear(model_dim, max_exchanges + 1)
        self.value_head = nn.Linear(model_dim, 1)

    def forward(
        self, g: Float[Tensor, "B d"], *, selected: Bool[Tensor, "B N"]
    ) -> tuple[Float[Tensor, "B 2"], Float[Tensor, "B M_plus_1"], Float[Tensor, "B"]]:
        """Return stop logits, legal exchange-count logits, and value estimates.

        Return ``(stop_logits, count_logits, value)`` with shapes ``[B,2]``,
        ``[B,M+1]``, and ``[B]``. Stop columns mean continue (0) and stop (1);
        count column ``j`` means ``j`` exchanged pairs, and infeasible counts
        are ``-inf``. Values estimate future improvement in the capped objective
        minus continuation costs. Count logits are conditional on continuing.
        Inputs are trusted and unchanged, and outputs remain connected to
        autograd.
        """
        stop_logits = self.stop_head(g)

        num_selected: Int[Tensor, "B"] = selected.sum(dim=-1)
        num_unselected: Int[Tensor, "B"] = selected.shape[1] - num_selected
        # Each pair removes one selected point and adds one unselected point: m <= min(K, N-K).
        count_limit: Int[Tensor, "B"] = torch.minimum(num_selected, num_unselected)
        # The head only offers counts 0 through M, so no additional clamp to M is needed.
        counts: Int[Tensor, "M_plus_1"] = torch.arange(self.max_exchanges + 1, device=g.device)
        # Broadcast [1, M+1] against [B, 1] to mask unavailable counts in each row.
        count_logits = self.count_head(g).masked_fill(counts.unsqueeze(0) > count_limit.unsqueeze(-1), -torch.inf)
        value = self.value_head(g).squeeze(-1)
        return stop_logits, count_logits, value
