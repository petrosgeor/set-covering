"""Tensor-native episodes for fixed-budget emitter placement.

Each batch row is independent.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jaxtyping import Bool, Float
from torch import Tensor


@dataclass(frozen=True)
class EnvConfig:
    """Shared episode budgets, local random seed, and device."""

    num_emitters: int
    max_exchanges: int
    max_steps: int
    seed: int
    device: str


def load_config(path: str | Path) -> EnvConfig:
    """Read a flat, explicit YAML configuration without executable loading.

    Args:
        path: YAML file containing exactly the five EnvConfig fields.

    Returns:
        Shared configuration, with no implicit field defaults.
    """
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
        """Own an instance with contributions exp(-distance); reset starts an episode.

        Args:
            points: Finite float32/float64 coordinates with positive B, N, D.
            weights: Matching-dtype finite, nonnegative receiver weights.
            config: Shared budgets and destination CPU/CUDA device.
            demands: Matching-dtype finite, nonnegative receiver demand caps.
        """
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
        self,
        *,
        seed: int | None = None,
        mask: Bool[Tensor, "B"] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Restart selected rows and return the full batch.

        Args:
            seed: Optional override restarting the shared local random stream
                when at least one row resets. Otherwise the stream advances.
            mask: Boolean [B] on the configured device; true rows get a uniform
                K-subset and fresh counters. None resets all rows. Initialize
                with a full reset before using masks. An empty mask is a no-op.

        Returns:
            Full observation and info dictionaries, including unchanged rows.
        """
        batch_size, num_points = self._points.shape[:2]
        if mask is None:
            mask = torch.ones(batch_size, dtype=torch.bool, device=self._device)
            self._selected = torch.zeros_like(self._weights, dtype=torch.bool)
            self._steps_remaining = torch.zeros(
                batch_size, dtype=torch.int64, device=self._device
            )
            self._stopped = torch.zeros_like(mask)
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
            self._timed_out[mask] = False

        observation = self._get_observation()
        return observation, self._get_info(observation)

    @torch.no_grad()
    def step(
        self, action: dict[str, Tensor]
    ) -> tuple[
        dict[str, Tensor],
        Float[Tensor, "B"],
        Bool[Tensor, "B"],
        Bool[Tensor, "B"],
        dict[str, Tensor],
    ]:
        """Apply simultaneous exchanges and reward the change in objective.

        Args:
            action: Dictionary with signed ternary "exchanges" [B,N] and Boolean
                "stop" [B], on the configured device. Exchanges add unselected
                points and remove selected ones in equal counts up to the cap.
                Stops on active rows carry zero edits. Finished rows freeze.

        Returns:
            Observation, reward, terminated, truncated, and info. Explicit stops
            and the task horizon terminate a row; truncated is always false.
        """
        observation = self._get_observation()
        exchanges, stop = action["exchanges"], action["stop"]
        active = ~(self._stopped | self._timed_out)
        stopping = active & stop
        continuing = active & ~stop

        self._selected = torch.where(
            continuing[:, None],
            (self._selected | (exchanges == 1)) & (exchanges != -1),
            self._selected,
        )
        self._steps_remaining = self._steps_remaining - active.to(torch.int64)
        self._stopped = self._stopped | stopping
        self._timed_out = self._timed_out | (continuing & (self._steps_remaining == 0))

        next_observation = self._get_observation()
        reward = self._get_reward(observation, next_observation)
        terminated = self._stopped | self._timed_out
        truncated = torch.zeros_like(terminated)
        return (
            next_observation,
            reward,
            terminated,
            truncated,
            self._get_info(next_observation),
        )

    def _build_contributions(self) -> Float[Tensor, "B N N"]:
        distances = torch.cdist(
            self._points,
            self._points,
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        distances.diagonal(dim1=-2, dim2=-1).zero_()
        return torch.exp(-distances)

    def _evaluate_observation(
        self, observation: dict[str, Tensor]
    ) -> tuple[
        Float[Tensor, "B"],
        Float[Tensor, "B"],
    ]:
        received_signal = observation["received_signal"]
        capped_signal = torch.minimum(received_signal, observation["demands"])
        objective = (observation["weights"] * capped_signal).sum(dim=1)
        weighted_unmet_demand = (
            observation["weights"]
            * (observation["demands"] - received_signal).clamp_min(0)
        ).sum(dim=1)
        return objective, weighted_unmet_demand

    def _get_reward(
        self, observation: dict[str, Tensor], next_observation: dict[str, Tensor]
    ) -> Float[Tensor, "B"]:
        """Return the objective difference, including on the final transition."""
        objective, _ = self._evaluate_observation(observation)
        next_objective, _ = self._evaluate_observation(next_observation)
        return next_objective - objective

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
            "steps_remaining": self._steps_remaining.clone(),
        }

    def _get_info(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        objective, weighted_unmet_demand = self._evaluate_observation(observation)
        return {
            "objective": objective,
            "weighted_unmet_demand": weighted_unmet_demand,
            "stopped": self._stopped.clone(),
            "timed_out": self._timed_out.clone(),
        }
