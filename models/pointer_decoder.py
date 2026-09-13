"""Choose ordered emitter removals and additions with a shared GRU."""

import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.distributions import Categorical


class PointerDecoder(nn.Module):
    """Decode point choices conditional on an already-chosen exchange count."""

    def __init__(self, *, model_dim: int, max_exchanges: int) -> None:
        """Initialize count and phase embeddings, pointer projections, and recurrent memory."""
        super().__init__()
        self.max_exchanges = max_exchanges
        self.count_embedding = nn.Embedding(max_exchanges + 1, model_dim)   # this is max exhanges + 1 because we can have zero exchanges
        self.phase_embedding = nn.Embedding(2, model_dim)
        self.initial_memory = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.query_network = nn.Sequential(
            nn.Linear(2 * model_dim + 1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.query_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.key_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.gru = nn.GRUCell(2 * model_dim, model_dim)
        self.score_scale = math.sqrt(model_dim)

    def forward(
        self,
        H: Float[Tensor, "B N d"],
        g: Float[Tensor, "B d"],
        *,
        selected: Bool[Tensor, "B N"],
        counts: Int[Tensor, "B"],
        greedy: bool = False,
    ) -> tuple[Int[Tensor, "B M"], Int[Tensor, "B M"], Float[Tensor, "B"]]:
        """Generate an ordered removal/addition trace and its point-choice log probability.

        Return ``(removals, additions, log_prob)``: ``[B,M]`` index tensors with
        valid ordered prefixes of length ``counts[b]`` and ``-1`` padding, plus
        ``[B]`` log probabilities. Removals are decoded before additions; only
        point choices contribute, and zero-count rows have zero log probability.
        Greedy decoding picks the lowest eligible index on ties. The caller
        supplies valid inputs, which remain unchanged, and the log probabilities
        stay connected to the encoder.
        """
        return self._decode(H, g, selected, counts, greedy=greedy, choices=None)

    def evaluate(
        self,
        H: Float[Tensor, "B N d"],
        g: Float[Tensor, "B d"],
        *,
        selected: Bool[Tensor, "B N"],
        counts: Int[Tensor, "B"],
        removals: Int[Tensor, "B M"],
        additions: Int[Tensor, "B M"],
    ) -> Float[Tensor, "B"]:
        """Score supplied ordered point choices without sampling or consuming RNG.

        Return ``[B]`` conditional point-choice log probabilities, with zero for
        empty traces. The first ``counts[b]`` entries of each tensor must be
        valid distinct choices from the original selected or unselected sets;
        trailing padding is ignored. Inputs are unchanged, and floating inputs
        remain attached to autograd.
        """
        _, _, log_prob = self._decode(H, g, selected, counts, greedy=False, choices=(removals, additions))
        return log_prob

    def _decode(
        self,
        H: Tensor,
        g: Tensor,
        selected: Tensor,
        counts: Tensor,
        *,
        greedy: bool,
        choices: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = H.shape[0]
        # Each row uses only its first counts[b] entries.
        # Remaining entries stay -1 and never participate in decoding or scoring.
        removals = torch.full((batch_size, self.max_exchanges), -1, dtype=torch.int64, device=H.device)
        additions = torch.full_like(removals, -1)
        # Keep empty traces differentiable, with exactly zero input gradients.
        log_prob = (H * 0).sum(dim=(1, 2)) + (g * 0).sum(dim=-1)

        positive_rows = (counts > 0).nonzero(as_tuple=True)[0]
        if positive_rows.numel() == 0:
            return removals, additions, log_prob

        point_embeddings = H[positive_rows]
        pair_counts = counts[positive_rows]
        original_selection = selected[positive_rows]
        memory = self.initial_memory(torch.cat((g[positive_rows], self.count_embedding(pair_counts)), dim=-1))
        keys = self.key_projection(point_embeddings)

        # Complete every removal before starting additions, retaining the memory.
        for phase, output in enumerate((removals, additions)):
            eligible = original_selection.clone() if phase == 0 else ~original_selection
            for step in range(self.max_exchanges):
                # Shorter traces wait while longer rows finish this phase.
                active = (pair_counts > step).nonzero(as_tuple=True)[0]
                if active.numel() == 0:
                    break
                batch_rows = positive_rows[active]
                phase_vectors = self.phase_embedding.weight[phase].expand(active.numel(), -1)
                remaining_fraction = (pair_counts[active] - step).to(dtype=H.dtype) / pair_counts[active]
                query = self.query_network(
                    torch.cat((memory[active], phase_vectors, remaining_fraction.unsqueeze(-1)), dim=-1)
                )
                projected_query = self.query_projection(query)
                logits = torch.bmm(keys[active], projected_query.unsqueeze(-1)).squeeze(-1) / self.score_scale
                logits = logits.masked_fill(~eligible[active], -torch.inf)
                distribution = Categorical(logits=logits)

                if choices is not None:
                    chosen = choices[phase][batch_rows, step]
                elif greedy:
                    chosen = logits.argmax(dim=-1)
                else:
                    chosen = distribution.sample()

                log_prob = log_prob.index_add(0, batch_rows, distribution.log_prob(chosen))
                chosen_embeddings = point_embeddings[active, chosen]
                next_memory = self.gru(torch.cat((chosen_embeddings, phase_vectors), dim=-1), memory[active])
                memory = memory.index_copy(0, active, next_memory)
                output[batch_rows, step] = chosen
                eligible[active, chosen] = False

        return removals, additions, log_prob
