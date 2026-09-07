"""Stop, exchange-count, and value heads for emitter observations."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


class PolicyHeads(nn.Module):
    """Project global embeddings into choice logits and remaining improvement estimates."""

    def __init__(self, *, model_dim: int, max_exchanges: int) -> None:
        """Build three independent affine projections.

        Args:
            model_dim: Positive global embedding width d.
            max_exchanges: Nonnegative maximum pair count M.
        """
        super().__init__()
        self.max_exchanges = max_exchanges
        self.stop_head = nn.Linear(model_dim, 2)
        self.count_head = nn.Linear(model_dim, max_exchanges + 1)
        self.value_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        g: Float[Tensor, "B d"],
        *,
        selected: Bool[Tensor, "B N"],
    ) -> tuple[Float[Tensor, "B 2"], Float[Tensor, "B M_plus_1"], Float[Tensor, "B"]]:
        """Return logits for legal choices and a value for each observation.

        Args:
            g: Global embeddings matching the model's floating dtype and device.
            selected: Boolean selections with positive B and N, on g's device.

        Returns:
            Stop logits [B,2], count logits [B,M+1], and values [B]. Stop
            columns are continue (0) and stop (1); count column j means j
            exchanged pairs. Invalid count logits are negative infinity.
            Values estimate remaining capped-objective improvement. Count
            logits are conditional on continuing; this method does not sample
            either decision.

        The caller supplies consistent, valid inputs and controls initialization
        seeds and model placement. Inputs are not validated or modified, and
        outputs remain connected to autograd.
        """
        stop_logits = self.stop_head(g)

        num_selected = selected.sum(dim=-1)
        count_limit = torch.minimum(
            num_selected, selected.shape[1] - num_selected
        ).clamp_max(self.max_exchanges)
        counts = torch.arange(self.max_exchanges + 1, device=g.device)
        count_logits = self.count_head(g).masked_fill(
            counts.unsqueeze(0) > count_limit.unsqueeze(-1), -torch.inf
        )
        value = self.value_head(g).squeeze(-1)
        return stop_logits, count_logits, value
