"""Tensor-native episodes for fixed-budget emitter placement.

Each batch row is independent.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
import yaml
from jaxtyping import Bool, Float
from torch import Tensor

SUCCESS_TOLERANCE = 1e-6


@dataclass(frozen=True)
class EnvConfig:
    """Shared search budgets, continuation cost, local seed, and device."""

    num_emitters: int
    max_exchanges: int
    max_steps: int
    seed: int
    device: str
    step_cost: float

    def __post_init__(self) -> None:
        """Require a finite, positive cost for every continuation."""
        if not isfinite(self.step_cost) or self.step_cost <= 0:
            raise ValueError("step_cost must be a positive finite float")


def load_config(path: str | Path) -> EnvConfig:
    """Load the six required ``EnvConfig`` fields from a flat YAML mapping without executable loading."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return EnvConfig(**values)


class BatchedEmitterEnv:
    """Run batched emitter-exchange episodes with caller-supplied valid actions."""

    @torch.no_grad()
    def __init__(
        self,
        points: Float[Tensor, "B N D"],
        weights: Float[Tensor, "B N"],
        config: EnvConfig,
        *,
        demands: Float[Tensor, "B N"],
    ) -> None:
        """Clone points, weights, and demand caps onto the configured device; call ``reset()`` before stepping."""
        self.config = config
        self._points = points.detach().to(config.device).clone()
        self._weights = weights.detach().to(config.device).clone()
        self._demands = demands.detach().to(config.device).clone()
        self._device = self._points.device
        self._contributions = self._build_contributions()
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(config.seed)

    @torch.no_grad()
    def reset(
        self, *, seed: int | None = None, mask: Bool[Tensor, "B"] | None = None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Reset all rows or rows selected by ``mask`` and return full-batch observation and info.

        Use a full reset before masked resets; an empty mask is a no-op. The mask
        must be Boolean with shape ``[B]`` on the configured device. A provided
        seed resets the shared RNG when any row resets. Reset rows receive a
        fresh uniform K-subset and step budget. Rows with weighted unmet demand
        at most ``1e-6`` are marked succeeded and should not receive actions.
        """
        batch_size, num_points = self._points.shape[:2]
        if mask is None:
            mask = torch.ones(batch_size, dtype=torch.bool, device=self._device)
            self._selected = torch.zeros_like(self._weights, dtype=torch.bool)
            self._steps_remaining = torch.zeros(batch_size, dtype=torch.int64, device=self._device)
            self._stopped = torch.zeros_like(mask)
            self._succeeded = torch.zeros_like(mask)
            self._timed_out = torch.zeros_like(mask)

        if mask.any():
            if seed is not None:
                self._generator.manual_seed(seed)
            selected = torch.zeros_like(self._weights[mask], dtype=torch.bool)
            if self.config.num_emitters == num_points:
                selected.fill_(True)
            elif self.config.num_emitters:
                indices = torch.multinomial(
                    torch.ones_like(self._weights[mask]),
                    self.config.num_emitters,
                    replacement=False,
                    generator=self._generator,
                )
                selected.scatter_(1, indices, True)
            self._selected[mask] = selected
            self._steps_remaining[mask] = self.config.max_steps
            self._stopped[mask] = False
            self._succeeded[mask] = False
            self._timed_out[mask] = False

        observation = self._get_observation()
        _, weighted_unmet_demand = self._evaluate_observation(observation)
        self._succeeded[mask] = weighted_unmet_demand[mask] <= SUCCESS_TOLERANCE
        return observation, self._get_info(observation)

    @torch.no_grad()
    def step(
        self, action: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Float[Tensor, "B"], Bool[Tensor, "B"], Bool[Tensor, "B"], dict[str, Tensor]]:
        """Apply a simultaneous valid exchange and return the next transition tuple.

        ``action`` contains signed ternary ``exchanges`` ``[B,N]`` and Boolean
        ``stop`` ``[B]``. Continuing rows exchange equal numbers of selected and
        unselected points up to the cap; stopping rows have zero edits, and
        finished rows freeze. Return ``(observation, reward, terminated,
        truncated, info)``. Explicit stops and successful continuations
        terminate before timeout; only continuing, unsuccessful rows at the
        step limit truncate.
        """
        observation = self._get_observation()
        exchanges, stop = action["exchanges"], action["stop"]
        active: Bool[Tensor, "B"] = ~(self._stopped | self._succeeded | self._timed_out)
        stopping: Bool[Tensor, "B"] = active & stop
        continuing: Bool[Tensor, "B"] = active & ~stop

        self._selected = torch.where(
            continuing[:, None], (self._selected | (exchanges == 1)) & (exchanges != -1), self._selected
        )
        self._steps_remaining = self._steps_remaining - active.to(torch.int64)
        self._stopped = self._stopped | stopping

        next_observation = self._get_observation()
        _, weighted_unmet_demand = self._evaluate_observation(next_observation)
        succeeded: Bool[Tensor, "B"] = continuing & (weighted_unmet_demand <= SUCCESS_TOLERANCE)
        self._succeeded = self._succeeded | succeeded
        self._timed_out = self._timed_out | (continuing & ~succeeded & (self._steps_remaining <= 0))

        reward = self._get_reward(observation, next_observation, continuing)
        terminated = self._stopped | self._succeeded
        truncated = self._timed_out.clone()
        return (next_observation, reward, terminated, truncated, self._get_info(next_observation))

    def _build_contributions(self) -> Float[Tensor, "B N N"]:
        distances = torch.cdist(self._points, self._points, p=2, compute_mode="donot_use_mm_for_euclid_dist")
        distances.diagonal(dim1=-2, dim2=-1).zero_()
        return torch.exp(-distances)

    def _evaluate_observation(self, observation: dict[str, Tensor]) -> tuple[Float[Tensor, "B"], Float[Tensor, "B"]]:
        received_signal = observation["received_signal"]
        capped_signal = torch.minimum(received_signal, observation["demands"])
        objective = (observation["weights"] * capped_signal).sum(dim=1)
        weighted_unmet_demand = (observation["weights"] * (observation["demands"] - received_signal).clamp_min(0)).sum(
            dim=1
        )
        return objective, weighted_unmet_demand

    def _get_reward(
        self, observation: dict[str, Tensor], next_observation: dict[str, Tensor], continuing: Bool[Tensor, "B"]
    ) -> Float[Tensor, "B"]:
        """Return objective improvement minus the cost of each continuation."""
        objective, _ = self._evaluate_observation(observation)
        next_objective, _ = self._evaluate_observation(next_observation)
        improvement = next_objective - objective
        return torch.where(continuing, improvement - self.config.step_cost, torch.zeros_like(improvement))

    def _get_observation(self) -> dict[str, Tensor]:
        """Return dynamic snapshots and borrowed read-only instance tensors."""
        received_signal: Float[Tensor, "B N"] = torch.bmm(
            self._contributions, self._selected.to(self._points.dtype).unsqueeze(-1)
        ).squeeze(-1)
        return {
            "points": self._points,
            "weights": self._weights,
            "demands": self._demands,
            "contributions": self._contributions,
            "selected": self._selected.clone(),
            "received_signal": received_signal,
        }

    def _get_info(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        objective, weighted_unmet_demand = self._evaluate_observation(observation)
        return {
            "objective": objective,
            "weighted_unmet_demand": weighted_unmet_demand,
            "stopped": self._stopped.clone(),
            "succeeded": self._succeeded.clone(),
            "timed_out": self._timed_out.clone(),
            "steps_remaining": self._steps_remaining.clone(),
        }
