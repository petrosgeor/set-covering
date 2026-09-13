"""Assemble the emitter encoder, decision heads, and pointer decoder."""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.distributions import Categorical

from models.pointer_decoder import PointerDecoder
from models.policy_heads import PolicyHeads
from models.transformer_encoder import TransformerEncoder


class EmitterPolicy(nn.Module):
    """Compose the encoder, decision heads, and pointer decoder into emitter actions.

    The caller controls input validity, seeds, completed rows, and environment
    stepping; methods preserve inputs and autograd.
    """

    def __init__(self, *, encoder: TransformerEncoder, heads: PolicyHeads, decoder: PointerDecoder) -> None:
        """Register caller-owned modules and require matching exchange caps."""
        super().__init__()
        if heads.max_exchanges != decoder.max_exchanges:
            raise ValueError("heads and decoder must share max_exchanges")
        self.encoder = encoder
        self.heads = heads
        self.decoder = decoder

    def forward(
        self, observation: dict[str, Tensor], *, greedy: bool = False
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Float[Tensor, "B"], Float[Tensor, "B"]]:
        """Build an action, ordered trace, log probability, and value.

        Return ``(action, trace, log_prob, value)``. ``action`` contains int64
        signed ``exchanges`` ``[B,N]`` and Boolean ``stop`` ``[B]``; ``trace``
        contains ``stop`` ``[B]``, ``counts`` ``[B]``, and ordered
        ``removals``/``additions`` ``[B,M]`` with valid zero-based prefixes
        followed by ``-1`` padding. ``log_prob`` and ``value`` have shape
        ``[B]``. Greedy decoding uses argmax choices while still scoring them
        under their categorical distributions.
        """
        point_embeddings, global_embedding = self.encoder(observation)
        selected: Bool[Tensor, "B N"] = observation["selected"]
        stop_logits, count_logits, value = self.heads(global_embedding, selected=selected)

        stop_distribution = Categorical(logits=stop_logits)
        if greedy:
            stop = stop_logits.argmax(dim=-1).to(dtype=torch.bool)
        else:
            stop = stop_distribution.sample().to(dtype=torch.bool)

        batch_size = selected.shape[0]
        counts: Int[Tensor, "B"] = torch.zeros(batch_size, dtype=torch.int64, device=selected.device)
        continuing_rows = (~stop).nonzero(as_tuple=True)[0]
        if continuing_rows.numel():
            continuing_count_distribution = Categorical(logits=count_logits[continuing_rows])
            if greedy:
                continuing_counts = count_logits[continuing_rows].argmax(dim=-1)
            else:
                continuing_counts = continuing_count_distribution.sample()
            counts.index_copy_(0, continuing_rows, continuing_counts)

        removals, additions, decoder_log_prob = self.decoder(
            point_embeddings, global_embedding, selected=selected, counts=counts, greedy=greedy
        )

        valid_prefix = torch.arange(self.decoder.max_exchanges, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)
        removal_addresses = removals.clamp_min(0)
        addition_addresses = additions.clamp_min(0)
        exchanges: Int[Tensor, "B N"] = torch.zeros(
            batch_size, selected.shape[1], dtype=torch.int64, device=selected.device
        )
        exchanges.scatter_add_(1, removal_addresses, -valid_prefix.to(dtype=torch.int64))
        exchanges.scatter_add_(1, addition_addresses, valid_prefix.to(dtype=torch.int64))

        head_log_prob = self._head_log_prob(stop_logits, count_logits, stop, counts)
        log_prob = head_log_prob + decoder_log_prob
        action = {"exchanges": exchanges, "stop": stop.clone()}
        trace = {"stop": stop, "counts": counts, "removals": removals, "additions": additions}
        return action, trace, log_prob, value

    def evaluate(
        self, observation: dict[str, Tensor], trace: dict[str, Tensor]
    ) -> tuple[Float[Tensor, "B"], Float[Tensor, "B"]]:
        """Score a supplied ordered trace without sampling.

        Return ``(log_prob, value)`` with each tensor shaped ``[B]``. The trace
        follows :meth:`forward`'s format, and the supplied point-choice order is
        scored without sampling.
        """
        point_embeddings, global_embedding = self.encoder(observation)
        selected: Bool[Tensor, "B N"] = observation["selected"]
        stop_logits, count_logits, value = self.heads(global_embedding, selected=selected)

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
        branch_log_prob = Categorical(logits=stop_logits).log_prob(stop.to(dtype=torch.int64))
        count_log_prob = torch.zeros_like(branch_log_prob)
        continuing_rows = (~stop).nonzero(as_tuple=True)[0]
        if continuing_rows.numel():
            count_log_prob.index_copy_(
                0, continuing_rows, Categorical(logits=count_logits[continuing_rows]).log_prob(counts[continuing_rows])
            )
        return branch_log_prob + count_log_prob
