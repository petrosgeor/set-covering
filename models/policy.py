"""Assemble the emitter encoder, decision heads, and pointer decoder."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.distributions import Categorical

from models.pointer_decoder import PointerDecoder
from models.policy_heads import PolicyHeads
from models.transformer_encoder import TransformerEncoder


class EmitterPolicy(nn.Module):
    """Construct complete emitter actions and score their ordered traces.

    The caller controls valid inputs, matching widths, dtype and device
    placement, random seeds, and completed rows. Methods preserve inputs and
    autograd; the caller steps or resets the environment.
    """

    def __init__(
        self,
        *,
        encoder: TransformerEncoder,
        heads: PolicyHeads,
        decoder: PointerDecoder,
    ) -> None:
        """Register the caller-owned policy modules.

        Args:
            encoder: Module that maps an observation to point and global
                embeddings.
            heads: Module that produces stop, count, and value outputs.
            decoder: Module that chooses ordered removals and additions.

        Raises:
            ValueError: If the heads and decoder use different exchange caps.
        """
        super().__init__()
        if heads.max_exchanges != decoder.max_exchanges:
            raise ValueError("heads and decoder must share max_exchanges")
        self.encoder = encoder
        self.heads = heads
        self.decoder = decoder

    def forward(
        self,
        observation: dict[str, Tensor],
        *,
        greedy: bool = False,
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        Float[Tensor, "B"],
        Float[Tensor, "B"],
    ]:
        """Construct an action and return its trace log probability and value.

        Args:
            observation: Batched emitter observation accepted by the encoder.
            greedy: If true, take argmax decisions. Selected decisions are
                still scored under their categorical distributions.

        Returns:
            A tuple ``(action, trace, log_prob, value)``. ``action`` contains
            int64 signed ``exchanges`` ``[B,N]`` and Boolean ``stop`` ``[B]``.
            ``trace`` contains Boolean ``stop`` ``[B]``, int64 ``counts``
            ``[B]``, and int64 ``removals``/``additions`` ``[B,M]`` with
            zero-based valid prefixes and ``-1`` padding. ``log_prob`` and
            ``value`` are both ``[B]``.
        """
        point_embeddings, global_embedding = self.encoder(observation)
        selected: Bool[Tensor, "B N"] = observation["selected"]
        stop_logits, count_logits, value = self.heads(
            global_embedding, selected=selected
        )

        stop_distribution = Categorical(logits=stop_logits)
        if greedy:
            stop = stop_logits.argmax(dim=-1).to(dtype=torch.bool)
        else:
            stop = stop_distribution.sample().to(dtype=torch.bool)

        batch_size = selected.shape[0]
        counts: Int[Tensor, "B"] = torch.zeros(
            batch_size, dtype=torch.int64, device=selected.device
        )
        continuing_rows = (~stop).nonzero(as_tuple=True)[0]
        if continuing_rows.numel():
            continuing_count_distribution = Categorical(
                logits=count_logits[continuing_rows]
            )
            if greedy:
                continuing_counts = count_logits[continuing_rows].argmax(dim=-1)
            else:
                continuing_counts = continuing_count_distribution.sample()
            counts.index_copy_(0, continuing_rows, continuing_counts)

        removals, additions, decoder_log_prob = self.decoder(
            point_embeddings,
            global_embedding,
            selected=selected,
            counts=counts,
            greedy=greedy,
        )

        valid_prefix = torch.arange(
            self.decoder.max_exchanges, device=counts.device
        ).unsqueeze(0) < counts.unsqueeze(1)
        removal_addresses = removals.clamp_min(0)
        addition_addresses = additions.clamp_min(0)
        exchanges: Int[Tensor, "B N"] = torch.zeros(
            batch_size,
            selected.shape[1],
            dtype=torch.int64,
            device=selected.device,
        )
        exchanges.scatter_add_(
            1,
            removal_addresses,
            -valid_prefix.to(dtype=torch.int64),
        )
        exchanges.scatter_add_(
            1,
            addition_addresses,
            valid_prefix.to(dtype=torch.int64),
        )

        head_log_prob = self._head_log_prob(stop_logits, count_logits, stop, counts)
        log_prob = head_log_prob + decoder_log_prob
        action = {"exchanges": exchanges, "stop": stop.clone()}
        trace = {
            "stop": stop,
            "counts": counts,
            "removals": removals,
            "additions": additions,
        }
        return action, trace, log_prob, value

    def evaluate(
        self,
        observation: dict[str, Tensor],
        trace: dict[str, Tensor],
    ) -> tuple[Float[Tensor, "B"], Float[Tensor, "B"]]:
        """Score a supplied ordered trace under the current policy.

        Args:
            observation: Batched emitter observation accepted by the encoder.
            trace: Trace with ``stop``, ``counts``, ``removals``, and
                ``additions`` fields matching :meth:`forward`.

        Returns:
            A tuple ``(log_prob, value)`` of batched tensors. The supplied
            point order is scored without sampling.
        """
        point_embeddings, global_embedding = self.encoder(observation)
        selected: Bool[Tensor, "B N"] = observation["selected"]
        stop_logits, count_logits, value = self.heads(
            global_embedding, selected=selected
        )

        stop = trace["stop"]
        counts: Int[Tensor, "B"] = trace["counts"]
        decoder_log_prob = self.decoder.evaluate(
            point_embeddings,
            global_embedding,
            selected=selected,
            counts=counts,
            removals=trace["removals"],
            additions=trace["additions"],
        )
        head_log_prob = self._head_log_prob(stop_logits, count_logits, stop, counts)
        return head_log_prob + decoder_log_prob, value

    def _head_log_prob(
        self,
        stop_logits: Float[Tensor, "B 2"],
        count_logits: Float[Tensor, "B M_plus_1"],
        stop: Bool[Tensor, "B"],
        counts: Int[Tensor, "B"],
    ) -> Float[Tensor, "B"]:
        branch_log_prob = Categorical(logits=stop_logits).log_prob(
            stop.to(dtype=torch.int64)
        )
        count_log_prob = torch.zeros_like(branch_log_prob)
        continuing_rows = (~stop).nonzero(as_tuple=True)[0]
        if continuing_rows.numel():
            count_log_prob.index_copy_(
                0,
                continuing_rows,
                Categorical(logits=count_logits[continuing_rows]).log_prob(
                    counts[continuing_rows]
                ),
            )
        return branch_log_prob + count_log_prob
