# Batched environment

[`BatchedEmitterEnv`](../../envs/emitter.py) runs independent emitter-exchange
episodes.

Construct `BatchedEmitterEnv(points, weights, decay, config, *, demands)`.
`decay` is nonincreasing, and tensors share a float dtype.
`load_config(path)` reads the five required fields from
[`environment.yaml`](../../parameters/environment.yaml): `num_emitters`,
`max_exchanges`, `max_steps`, `seed`, and `device`.

## Observation

`B` counts episodes, `N` points, and `D` coordinates.

| Field | Shape | Type |
| --- | --- | --- |
| `points` | `[B,N,D]` | float32/float64 |
| `weights`, `demands`, `received_signal` | `[B,N]` | Matching float |
| `contributions` | `[B,N,N]` | Matching float |
| `selected` | `[B,N]` | Boolean |
| `steps_remaining` | `[B]` | int64 |

Fixed instance tensors are borrowed, read-only; dynamic observations are
snapshots. Construction detaches, moves, and clones instance tensors.
`received_signal` is the raw product `Az`. `decay` preserves input shape, dtype,
and device.

## Usage

```python
import torch
from envs.emitter import BatchedEmitterEnv, load_config

config = load_config("parameters/environment.yaml")
points = torch.tensor([[[0.0], [0.5], [1.0], [1.5]]], dtype=torch.float32)
demands = torch.tensor([[1.0, 1.25, 0.75, 1.0]], dtype=points.dtype)
env = BatchedEmitterEnv(
    points,
    torch.ones(1, 4, dtype=points.dtype),
    lambda d: (1 - d).clamp_min(0),
    config,
    demands=demands,
)
observation, info = env.reset(seed=0)
```

`step({"exchanges": edits, "stop": stop})` returns
`(observation, reward, terminated, truncated, info)`. Signed integer exchanges
`[B,N]` have equal `-1` and `+1` counts within the cap. Callers validate that
`-1` marks selected and `+1` marks unselected points. Active stops `[B]` carry
zero edits and end rows. Demand satisfaction does not terminate an episode.
Demands are finite, nonnegative, fixed across resets; zero is valid.

Each active step consumes one decision. Stops and horizon exhaustion terminate;
`truncated` is false; finished rows freeze with zero reward. `info`
contains `[B]` fields `objective`, `weighted_unmet_demand`, `stopped`, and
`timed_out`. Reward is next objective minus current objective.

Call full `reset()` before masked reset. Record terminal observations before
`reset(mask=terminated | truncated)`, which restarts completed rows and returns
the full batch. `reset(seed=...)` reseeds the generator.
