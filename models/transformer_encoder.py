"""Transformer encoder for batched emitter observations."""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from jaxtyping import Bool, Float
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture widths and counts for the transformer encoder."""

    model_dim: int
    num_layers: int
    num_heads: int
    feedforward_dim: int


def load_config(path: str | Path) -> EncoderConfig:
    """Load the four required encoder fields from the ``encoder`` YAML mapping."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return EncoderConfig(**values["encoder"])


class TransformerEncoder(nn.Module):
    """Encode coordinates, point scalars, and demand margin into point and global embeddings."""

    def __init__(self, *, config: EncoderConfig, coordinate_dim: int) -> None:
        """Build the encoder from caller-owned architecture settings."""
        super().__init__()
        self.config = config
        self.coordinate_dim = coordinate_dim

        self.input_projection = nn.Sequential(
            nn.Linear(coordinate_dim + 4, config.model_dim), nn.GELU(), nn.Linear(config.model_dim, config.model_dim)
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
            nn.Linear(config.model_dim, config.model_dim), nn.GELU(), nn.Linear(config.model_dim, config.model_dim)
        )

    def forward(self, observation: dict[str, Tensor]) -> tuple[Float[Tensor, "B N d"], Float[Tensor, "B d"]]:
        """Encode point observations into per-point ``H`` and global ``g`` embeddings.

        Use ``points``, ``weights``, ``selected``, ``received_signal``, and
        ``demands``; contribution and timer fields are ignored. Return
        ``(H, g)`` with shapes ``[B,N,d]`` and ``[B,d]``.
        """
        points: Float[Tensor, "B N D"] = observation["points"]
        weights: Float[Tensor, "B N"] = observation["weights"]
        selected: Bool[Tensor, "B N"] = observation["selected"]
        received_signal: Float[Tensor, "B N"] = observation["received_signal"]
        demands: Float[Tensor, "B N"] = observation["demands"]

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
        global_embedding: Float[Tensor, "B d"] = self.global_projection(pooled_embedding)
        return point_embeddings, global_embedding
