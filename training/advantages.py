"""Multi-step returns and advantages for collected rollouts."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor


@torch.no_grad()
def compute_returns_and_advantages(
    rewards: Float[Tensor, "T B"],
    values: Float[Tensor, "T B"],
    next_values: Float[Tensor, "T B"],
    terminated: Bool[Tensor, "T B"],
    truncated: Bool[Tensor, "T B"],
    valid: Bool[Tensor, "T B"],
    *,
    gamma: float = 1.0,
) -> tuple[Float[Tensor, "T B"], Float[Tensor, "T B"]]:
    """Compute returns and advantages for a padded rollout.

    Use each step's pre-reset ``next_values`` across gaps and zero invalid outputs.
    """
    num_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)

    for t in reversed(range(num_steps)):
        next_return = next_values[t]

        if t + 1 < num_steps:
            # A missing successor must use this step's pre-reset bootstrap value.
            episode_continues = valid[t + 1] & ~terminated[t] & ~truncated[t]
            next_return = torch.where(
                episode_continues,
                returns[t + 1],
                next_return,
            )

        next_return = next_return.masked_fill(terminated[t], 0.0)
        returns[t] = rewards[t] + gamma * next_return

    returns = returns.masked_fill(~valid, 0.0)
    advantages = (returns - values).masked_fill(~valid, 0.0)
    return returns, advantages
