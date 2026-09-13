# Batched environment

[`BatchedEmitterEnv`](../../envs/emitter.py) runs independent emitter-exchange
episodes.

Construct `BatchedEmitterEnv(points, weights, config, *, demands)` with a shared
float dtype. `load_config(path)` reads the six required fields from
[`environment.yaml`](../../parameters/environment.yaml): `num_emitters`,
`max_exchanges`, `max_steps`, `seed`, `device`, and `step_cost`. The cost is a
positive finite scalar in the same objective units as the capped utility. The
sample value `0.1` is illustrative and untuned.

## Observation

`B` counts episodes, `N` points, and `D` coordinates.

| Field | Shape | Type |
| --- | --- | --- |
| `points` | `[B,N,D]` | float32/float64 |
| `weights`, `demands`, `received_signal` | `[B,N]` | Matching float |
| `contributions` | `[B,N,N]` | Matching float |
| `selected` | `[B,N]` | Boolean |

Construction detaches, moves, and clones instance tensors. Observations borrow
these fixed tensors as read-only data and snapshot dynamic state.
`received_signal` is the raw product `Az`. Contributions use
`exp(-||x_i-x_j||_2)` with unit diagonal. Coordinates are used unchanged,
without normalization or a scale parameter. The collection countdown is not an
observation, so the policy cannot condition on it.

## Info

`reset()` and `step()` return `[B]` diagnostics in `info`:

| Field | Type | Meaning |
| --- | --- | --- |
| `objective`, `weighted_unmet_demand` | Matching float | Current capped utility and weighted shortfall |
| `stopped`, `succeeded`, `timed_out` | Boolean | Completion reason for each row |
| `steps_remaining` | int64 | External collection slots left |

The success test is `weighted_unmet_demand <= 1e-6` in raw objective units. It
therefore ignores receivers with zero weight. A row that starts in this state
is marked `succeeded` and frozen; callers should skip it. A later `step()` on
that row returns `terminated=True` and zero reward.

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
    config,
    demands=demands,
)
observation, info = env.reset(seed=0)
```

`step({"exchanges": edits, "stop": stop})` returns
`(observation, reward, terminated, truncated, info)`. Signed integer exchanges
`[B,N]` have equal `-1` and `+1` counts within the cap. Callers validate that
`-1` marks selected and `+1` marks unselected points. Active stops `[B]` carry
zero edits and end rows. A continuing exchange can end a row when the success
test passes. Demands are finite, nonnegative, fixed across resets; zero is
valid.

Each active call consumes one external collection slot. For a continuing action,
including a zero-size exchange, reward is

`next_objective - objective - config.step_cost`.

Stopping and calls on frozen rows receive zero reward. A stop or successful
exchange sets `terminated=True`. If the external step limit is reached after a
continuing action without either condition, `truncated=True` and
`terminated=False`. Success or stop takes precedence when both occur on the
last slot.

Call full `reset()` before masked reset. Record terminal observations before
`reset(mask=terminated | truncated)`, which restarts completed rows and returns
the full batch. Reset rows that start successful are reported in `info` and
must be excluded from the next policy call. `reset(seed=...)` reseeds the
generator.

An actor-critic collector resets rows on `terminated | truncated`. It must
bootstrap a truncated row from the final observation before reset, while a
terminated row has no bootstrap term. Return calculations must not use rewards
from the next episode.
