# Batched environment

[`BatchedEmitterEnv`](../../envs/emitter.py) runs independent emitter-exchange
episodes using PyTorch tensors. The [research description](../research/environment.md)
defines the objective, valid actions, and reward equations.

Construct it with `points`, `weights`, a nonincreasing distance-to-contribution `decay` callable,
and `EnvConfig`. `load_config(path)` reads the six required fields from
[`environment.yaml`](../../parameters/environment.yaml): `num_emitters` (selection
size), `max_exchanges` (pair limit), `max_steps` (horizon), `threshold` (minimum
signal), `seed` (local random generator), and `device`. There are no defaults.

## Observation

Here `B` counts episodes, `N` points, and `D` coordinates per point.

| Field | Shape | Type |
| --- | --- | --- |
| `points` | `[B,N,D]` | float32/float64 |
| `weights`, `received_signal` | `[B,N]` | Matching float |
| `contributions` | `[B,N,N]` | Matching float |
| `selected` | `[B,N]` | Boolean |
| `steps_remaining` | `[B]` | int64 |
| `feasible` | `[B]` | Boolean |

Fixed instance tensors are borrowed, read-only; dynamic observations are
snapshots. `decay` preserves its `[B,N,N]` input's shape, dtype, and device.

## Usage

Run from the repository root:

```python
import torch
from envs.emitter import BatchedEmitterEnv, load_config

config = load_config("parameters/environment.yaml")
points = torch.tensor([[[0.0], [0.5], [1.0], [1.5]]])
env = BatchedEmitterEnv(
    points, torch.ones(1, 4), lambda d: torch.exp(-d), config
)
observation, info = env.reset()
```

`step({"exchanges": edits, "stop": stop})` returns
`(observation, reward, terminated, truncated, info)`. Signed integer edits
`[B,N]` remove (`-1`) and add (`+1`) equal numbers of emitters within the cap.
Boolean stops `[B]` require feasibility and zero edits. Inputs are caller-validated.

Every active step consumes one decision. Stops and horizon exhaustion terminate;
`truncated` is always false. Finished rows freeze with zero reward. `info`
contains `[B]` fields `objective`, `shortfall`, `stopped`, and `timed_out`.

Call a full `reset()` first. Record terminal observations before
`reset(mask=terminated | truncated)`, which restarts only completed rows and
returns the full batch.
Use `reset(seed=...)` to reseed its local generator.
