# Deferred research ideas

## Count-aware decoder embeddings

The baseline uses an unconstrained learned count table
`E_count [M+1,d]`, with `e_m = E_count[m]`, as described in
[the model](docs/research/model.md#7-initialize-the-decoders-memory).
Nearby counts need not have similar embeddings.

An experiment could use `e_m = MLP(m)` or regularize adjacent table rows.
A numeric input alone does not guarantee similar embeddings.

Sharing structure across counts might help rarely sampled exchange sizes, but
smoothness could hurt counts that need different selection strategies.
Neither benefit nor novelty has been established.

Compare the free table with one structured variant under matched training and
evaluation budgets. Evaluate decoder choices
at supplied counts within the same supported range, separating decoder quality
from the count head's sampling frequencies. Measure final capped utility,
weighted unmet demand, and learning efficiency across multiple seeds.

The experiment is deferred until the baseline trains reliably.
