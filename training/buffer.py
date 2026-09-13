"""Batched rollout storage and discounted return targets."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class RolloutBuffer:
    """Own detached timesteps in reusable, fixed-capacity tensor storage.

    T for populated timesteps, B for batch size, N for points, D for coordinates, and M for exchange width.
    The first add fixes all shapes, devices, and dtypes for this buffer instance.
    """

    def __init__(self, rollout_steps: int) -> None:
        if isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int) or rollout_steps <= 0:
            raise ValueError("rollout_steps must be a positive integer")
        self.rollout_steps = rollout_steps
        self.observations: dict[str, Tensor] = {}
        self.traces: dict[str, Tensor] = {}
        self.rewards: Float[Tensor, "capacity B"] | None = None
        self.values: Float[Tensor, "capacity B"] | None = None
        self.next_values: Float[Tensor, "capacity B"] | None = None
        self.terminated: Bool[Tensor, "capacity B"] | None = None
        self.truncated: Bool[Tensor, "capacity B"] | None = None
        self.valid: Bool[Tensor, "capacity B"] | None = None
        self.clear()

    def clear(self) -> None:
        """Forget the rollout and targets, retaining storage for the next rollout."""
        self.position = 0
        self.returns: Float[Tensor, "T B"] | None = None
        self.advantages: Float[Tensor, "T B"] | None = None

    @torch.no_grad()
    def add(
        self,
        *,
        observation: dict[str, Tensor],
        trace: dict[str, Tensor],
        reward: Float[Tensor, "B"],
        value: Float[Tensor, "B"],
        next_value: Float[Tensor, "B"],
        terminated: Bool[Tensor, "B"],
        truncated: Bool[Tensor, "B"],
        valid: Bool[Tensor, "B"],
    ) -> None:
        """Copy one batched step, with next_value evaluated before any reset."""
        if self.position >= self.rollout_steps:
            raise ValueError("rollout buffer is full")
        points: Float[Tensor, "B N D"] = observation["points"]
        weights: Float[Tensor, "B N"] = observation["weights"]
        demands: Float[Tensor, "B N"] = observation["demands"]
        selected: Bool[Tensor, "B N"] = observation["selected"]
        received_signal: Float[Tensor, "B N"] = observation["received_signal"]
        stop: Bool[Tensor, "B"] = trace["stop"]
        counts: Int[Tensor, "B"] = trace["counts"]
        removals: Int[Tensor, "B M"] = trace["removals"]
        additions: Int[Tensor, "B M"] = trace["additions"]
        observation_fields = {
            "points": points,
            "weights": weights,
            "demands": demands,
            "selected": selected,
            "received_signal": received_signal,
        }
        trace_fields = {"stop": stop, "counts": counts, "removals": removals, "additions": additions}

        if self.rewards is None:
            # Infer storage from real inputs once; clear() keeps these allocations.
            self.observations = {
                key: tensor.new_empty((self.rollout_steps, *tensor.shape)) for key, tensor in observation_fields.items()
            }
            self.traces = {
                key: tensor.new_empty((self.rollout_steps, *tensor.shape)) for key, tensor in trace_fields.items()
            }
            self.rewards = reward.new_empty((self.rollout_steps, *reward.shape))
            self.values = value.new_empty((self.rollout_steps, *value.shape))
            self.next_values = next_value.new_empty((self.rollout_steps, *next_value.shape))
            self.terminated = terminated.new_empty((self.rollout_steps, *terminated.shape))
            self.truncated = truncated.new_empty((self.rollout_steps, *truncated.shape))
            self.valid = valid.new_empty((self.rollout_steps, *valid.shape))

        # Copy into owned storage so later input mutations cannot rewrite history.
        # no_grad keeps these snapshots independent of the collection graph.
        t = self.position
        for key, tensor in observation_fields.items():
            self.observations[key][t].copy_(tensor)
        for key, tensor in trace_fields.items():
            self.traces[key][t].copy_(tensor)
        self.rewards[t].copy_(reward)
        self.values[t].copy_(value)
        self.next_values[t].copy_(next_value)
        self.terminated[t].copy_(terminated)
        self.truncated[t].copy_(truncated)
        self.valid[t].copy_(valid)
        self.position += 1
        self.returns = None
        self.advantages = None

    def stack(self) -> dict[str, Tensor | dict[str, Tensor]]:
        """Export an independent [T, ...] batch without clearing the buffer."""
        #Clone it so callers can modify the exported batch without changing the stored experience.
        end = self.position
        return {
            "observations": {key: tensor[:end].clone() for key, tensor in self.observations.items()},
            "traces": {key: tensor[:end].clone() for key, tensor in self.traces.items()},
            "rewards": self.rewards[:end].clone(),
            "values": self.values[:end].clone(),
            "next_values": self.next_values[:end].clone(),
            "terminated": self.terminated[:end].clone(),
            "truncated": self.truncated[:end].clone(),
            "valid": self.valid[:end].clone(),
        }

    @torch.no_grad()
    def compute_returns_and_advantages(self, *, gamma: float = 1.0) -> None:
        """Compute targets; adjacent valid steps share an episode unless a completion flag separates them."""
        if self.position == 0:
            raise ValueError("cannot compute returns for an empty rollout buffer")
        # Unwritten slots may contain old rollout data and must never enter targets.
        num_steps = self.position
        rewards: Float[Tensor, "T B"] = self.rewards[:num_steps]
        values: Float[Tensor, "T B"] = self.values[:num_steps]
        next_values: Float[Tensor, "T B"] = self.next_values[:num_steps]
        terminated: Bool[Tensor, "T B"] = self.terminated[:num_steps]
        truncated: Bool[Tensor, "T B"] = self.truncated[:num_steps]
        valid: Bool[Tensor, "T B"] = self.valid[:num_steps]
        returns = torch.zeros_like(rewards)

        for t in reversed(range(num_steps)):
            next_return = next_values[t]
            if t + 1 < num_steps:
                # Follow the stored return only within the same episode; gaps and
                # truncations use this step's pre-reset value estimate instead.
                episode_continues = valid[t + 1] & ~terminated[t] & ~truncated[t]
                next_return = torch.where(episode_continues, returns[t + 1], next_return)
            next_return = next_return.masked_fill(terminated[t], 0.0)
            returns[t] = rewards[t] + gamma * next_return

        # Reset's random K-point selection can already succeed; valid excludes slots with no policy decision.
        self.returns = returns.masked_fill(~valid, 0.0)
        self.advantages = (self.returns - values).masked_fill(~valid, 0.0)
