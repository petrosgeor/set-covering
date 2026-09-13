# Fixed-budget emitter placement

Select exactly $K$ emitters to maximize weighted received signal, capped at
each receiver's demand. Every point is a receiver and a possible emitter.
Contributions decay exponentially with Euclidean distance.

This is an ongoing research project about learned local search for this
fixed-budget placement problem. The project investigates how the action design
and the amount of search used at inference affect the solutions we find. A
continuing decision earns objective improvement minus a fixed search cost; the
collection limit is external to the policy.

## Research questions

1. **Does learning variable-size exchanges improve local search compared with
   learning one exchange at a time?**

   The structured policy chooses how many emitters to exchange, then chooses the
   points to remove and add. A simpler learned policy makes one
   removal/addition pair per decision. We will compare solution quality and
   training efficiency, and examine when larger moves help.

2. **Can search at inference time improve the solutions found by a trained
   policy?**

   The initial inference experiment will sample multiple policy-guided
   trajectories and keep the best final selection. Each trajectory repeatedly
   generates one complete exchange, applies it, and observes the next state. We
   will compare this with a single trajectory, including deterministic policy
   decoding.

Evaluation will compare solution quality with computational cost. Planned
baselines include objective-based greedy construction and simple local search.
We will use unseen instances and test larger or shifted instances for
generalization.
Decision counts alone will not define compute, since a larger exchange
requires more point selections.

## Status

The batched PyTorch environment and transformer/GRU policy are implemented.
A2C training and the experiments above are unfinished.
Code lives in `envs/` and `models/`; YAML configuration lives in `parameters/`.

## Research documentation

1. [Problem](docs/research/problem.md): formulation, signal, capped utility, and unmet demand.
2. [Environment](docs/research/environment.md): initialization, exchanges, rewards, and episode outcome.
3. [Model](docs/research/model.md): features, attention, policy decisions, and the planned A2C connection.

## Code documentation

- [Environment](docs/code/environment.md): API, tensor shapes, configuration, reset/step behavior, and usage.
- [Encoder](docs/code/transformer_encoder.md): observation features, attention, pooling, shapes, and usage.
- [Policy](docs/code/policy.md): action generation, ordered trace replay, and tensor contracts.
