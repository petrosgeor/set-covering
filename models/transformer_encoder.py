"""Transformer encoder for batched emitter observations."""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture widths and counts for the transformer encoder."""

    model_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int


def load_config(path: str | Path) -> EncoderConfig:
    """Read the encoder section from an explicit YAML path.

    Args:
        path: YAML file containing an ``encoder`` mapping with the four
            required :class:`EncoderConfig` fields.

    Returns:
        The encoder architecture configuration.
    """
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return EncoderConfig(**values["encoder"])


class TransformerEncoder(nn.Module):
    """Encode point tokens and a global observation summary.

    The point feature width is ``D+4``: ``D`` coordinates followed by weight,
    selection, received signal, and demand margin.  The pooled point
    representation is augmented with the remaining budget fraction, giving
    the global projection a ``d+1`` input.
    """

    def __init__(
        self,
        *,
        config: EncoderConfig,
        coordinate_dim: int,
        max_steps: int,
    ) -> None:
        """Construct an encoder from the caller-owned architecture settings.

        Args:
            config: Representation width ``d`` and transformer layer dimensions.
            coordinate_dim: Coordinate width ``D`` in each point token.
            max_steps: Episode horizon ``T`` used to normalize remaining steps.
        """
        super().__init__()
        self.config = config
        self.coordinate_dim = coordinate_dim
        self.max_steps = max_steps

        self.input_projection = nn.Sequential(
            nn.Linear(coordinate_dim + 4, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=0.0,
                activation="gelu",
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=True,
            )
            for _ in range(config.num_layers)
        )
        self.output_norm = nn.LayerNorm(config.model_dim, eps=1e-5)
        self.global_projection = nn.Sequential(
            nn.Linear(config.model_dim + 1, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )

    def forward(
        self, observation: dict[str, Tensor]
    ) -> tuple[Float[Tensor, "B N d"], Float[Tensor, "B d"]]:
        """Encode point observations and their global summary.

        Args:
            observation: Mapping containing ``points`` ``[B,N,D]``, ``weights``
                ``[B,N]``, Boolean ``selected`` ``[B,N]``,
                ``received_signal`` ``[B,N]``, float ``demands`` ``[B,N]``,
                and integer ``steps_remaining`` ``[B]``.  Additional
                contribution fields are ignored.

        Returns:
            A tuple ``(H, g)``.  ``H`` has one contextual embedding of width
            ``d`` for every point, and ``g`` is a width-``d`` global embedding.
        """
        points: Float[Tensor, "B N D"] = observation["points"]
        weights: Float[Tensor, "B N"] = observation["weights"]
        selected: Bool[Tensor, "B N"] = observation["selected"]
        received_signal: Float[Tensor, "B N"] = observation["received_signal"]
        demands: Float[Tensor, "B N"] = observation["demands"]
        steps_remaining: Int[Tensor, "B"] = observation["steps_remaining"]

        point_features: Float[Tensor, "B N D_plus_4"] = torch.cat(
            (
                points,
                weights.unsqueeze(-1),
                selected.to(dtype=points.dtype).unsqueeze(-1),
                received_signal.unsqueeze(-1),
                (received_signal - demands).unsqueeze(-1),
            ),
            dim=-1,
        )
        point_embeddings: Float[Tensor, "B N d"] = self.input_projection(point_features)
        for layer in self.layers:
            point_embeddings = layer(point_embeddings)
        point_embeddings = self.output_norm(point_embeddings)

        pooled_embedding: Float[Tensor, "B d"] = point_embeddings.mean(dim=1)
        remaining_fraction: Float[Tensor, "B"] = (
            steps_remaining.to(dtype=points.dtype) / self.max_steps
        )
        global_features: Float[Tensor, "B d_plus_1"] = torch.cat(
            (
                pooled_embedding,
                remaining_fraction.unsqueeze(-1),
            ),
            dim=-1,
        )
        global_embedding: Float[Tensor, "B d"] = self.global_projection(global_features)
        return point_embeddings, global_embedding
