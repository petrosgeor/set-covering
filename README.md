# Fixed-budget emitter placement

Select a fixed number of points as emitters to maximize the weighted total
received signal while ensuring every receiver meets a minimum threshold.
Every point is a receiver, and selected points also contribute signal according
to their distance from each receiver.

The repository implements a PyTorch environment for emitter-exchange episodes,
a transformer encoder, stop/count/value heads, and a conditional GRU pointer
decoder. Complete policy assembly, environment-action construction, and PPO
training are not implemented yet.

The repository is organized by responsibility: `envs/` contains implemented
environment code, `parameters/` contains YAML configuration, and `models/`
contains neural model implementations.

## Research documentation

Read these documents in order for a guided explanation of one instance:

1. [Problem](docs/research/problem.md): points, emitter selection, received signal,
   objective, and feasibility constraints.
2. [Environment](docs/research/environment.md): initialization, exchanges,
   rewards, and the episode's final outcome.
3. [Model](docs/research/model.md): point features, attention, action construction,
   and the proposed connection to PPO.

## Code documentation

The [environment reference](docs/code/environment.md) covers the implemented
API, tensor shapes, configuration, reset and step behavior, and executable usage.
The [encoder reference](docs/code/transformer_encoder.md) follows the observation
features through attention and pooling, with shape annotations and executable usage.
