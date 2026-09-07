"""Stop, exchange-count, and value heads for emitter observations."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn


class PolicyHeads(nn.Module):
    """Project global embeddings into masked logits and remaining score estimates."""

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
        feasible: Bool[Tensor, "B"],
    ) -> tuple[Float[Tensor, "B 2"], Float[Tensor, "B M_plus_1"], Float[Tensor, "B"]]:
        """Return logits for legal choices and a value for each observation.

        Args:
            g: Global embeddings matching the model's floating dtype and device.
            selected: Boolean selections with positive B and N, on g's device.
            feasible: Boolean stop eligibility for the same batch and device.

        Returns:
            Stop logits [B,2], count logits [B,M+1], and values [B]. Stop
            columns are continue (0) and stop (1); count column j means j
            exchanged pairs. Invalid logits are negative infinity. Values
            estimate remaining score improvement. Count logits are conditional
            on continuing; this method does not sample either decision.

        The caller supplies consistent, valid inputs and controls initialization
        seeds and model placement. Inputs are not validated or modified, and
        outputs remain connected to autograd.
        """
        stop_logits = self.stop_head(g)
        stop_mask = torch.stack((torch.zeros_like(feasible), ~feasible), dim=-1)
        stop_logits = stop_logits.masked_fill(stop_mask, -torch.inf)

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
