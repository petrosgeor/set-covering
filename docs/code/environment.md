# Batched environment API and usage

[`BatchedEmitterEnv`](../../env.py) runs independent emitter-exchange episodes
with PyTorch tensors. It implements the process described in the
[research environment](../research/environment.md). This reference specifies
its constructor, dictionaries, caller preconditions, and collection lifecycle.

The class follows the Gymnasium-style `reset` and `step` return pattern.
It is not a registered Gymnasium environment or a drop-in vector wrapper.
The [proposed policy](../research/model.md) has no implementation yet.

All rows share the configuration values $K$, $M$, $T$, $\tau$, seed, and device.
The point coordinates, weights, and contributions may differ between rows.
Construction fixes those instance tensors; reset changes the episode state.

## Batched tensors

Each tensor has a leading axis for independent episodes. Let
$B$ be the number of independent rows, $N$ the number of points per row,
and $D$ the coordinate dimension. Every row has the same $N$ and $D$.
The configured device stores all returned tensors.

### Instance and dynamic tensors

| Python field | Symbol | Shape | Dtype and role |
| --- | --- | --- | --- |
| `points` | $X$ | `[B,N,D]` | Input floating dtype; fixed coordinates |
| `weights` | $w$ | `[B,N]` | Same floating dtype as `points`; fixed receiver weights |
| `contributions` | $A$ | `[B,N,N]` | Same floating dtype; fixed receiver-by-emitter contributions |
| `selected` | $z$ | `[B,N]` | Boolean dynamic emitter selection |
| `received_signal` | $r$ | `[B,N]` | Same floating dtype; recomputed from $Az$ |
| `steps_remaining` | $h$ | `[B]` | `torch.int64` dynamic decision count |
| `feasible` | $b$ | `[B]` | Boolean dynamic feasibility indicator |

In `A[b,i,j]`, `i` indexes the receiver and `j` indexes the possible emitter.
The constructor computes

$$
A_{b i j}
=k\!\left(\lVert X_{b i,:}-X_{b j,:}\rVert_2\right).
$$

The `decay` callable receives a distance tensor with shape `[B,N,N]` and must
return a contribution tensor with the same shape, dtype, and device. The
environment computes $A$ once and then recomputes $r=Az$ from the current
selection at each observation. Dense contributions use $O(BN^2)$ storage.

### Actions and return tensors

| Field | Shape | Dtype and role |
| --- | --- | --- |
| `action["exchanges"]` | `[B,N]` | Signed ternary action values `-1`, `0`, `1`; signed integer tensor contract |
| `action["stop"]` | `[B]` | Boolean stop decision |
| `reward` | `[B]` | Same floating dtype as the instance; score difference |
| `terminated` | `[B]` | Boolean completion flag |
| `truncated` | `[B]` | Boolean flag, always false |
| `mask` | `[B]` | Boolean rows selected for a masked reset |

The implementation annotates action values as generic `Tensor` and does not
validate their dtype. The table states the action contract used by the
environment and its tests.

### Information fields

| Python field | Shape | Dtype and role |
| --- | --- | --- |
| `info["objective"]` | `[B]` | Same floating dtype; weighted objective |
| `info["shortfall"]` | `[B]` | Same floating dtype; total threshold shortfall |
| `info["stopped"]` | `[B]` | Boolean persistent completion reason |
| `info["timed_out"]` | `[B]` | Boolean persistent completion reason |

`feasible` remains in the observation so the caller can decide whether to stop.
The `info` dictionary reports the two completion reasons separately. There is
no `done` observation field; callers can compute
`done = terminated | truncated`.

## Public interface

The implementation exposes the following public constructors and methods. The
generated `EnvConfig` constructor accepts the six named fields in the following
signature with no defaults:

```python
@dataclass(frozen=True)
class EnvConfig:
    num_emitters: int
    max_exchanges: int
    max_steps: int
    threshold: float
    seed: int
    device: str

def load_config(path: str | Path) -> EnvConfig: ...

class BatchedEmitterEnv:
    def __init__(
        self,
        points: Float[Tensor, "B N D"],
        weights: Float[Tensor, "B N"],
        decay: Callable[
            [Float[Tensor, "B N N"]],
            Float[Tensor, "B N N"],
        ],
        config: EnvConfig,
    ) -> None: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        mask: Bool[Tensor, "B"] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]: ...

    def step(
        self,
        action: dict[str, Tensor],
    ) -> tuple[
        dict[str, Tensor],
        Float[Tensor, "B"],
        Bool[Tensor, "B"],
        Bool[Tensor, "B"],
        dict[str, Tensor],
    ]: ...
```

`load_config` reads a flat YAML mapping with the six fields of `EnvConfig`:
`num_emitters`, `max_exchanges`, `max_steps`, `threshold`, `seed`, and
`device`. It supplies no implicit field defaults. The dataclass is frozen after
construction.

`reset` returns `(observation, info)`. With `mask=None`, it restarts every row.
With a Boolean `mask`, it restarts only rows whose entries are true and returns
the full batch, including unchanged rows. The first call must be an unmasked
full reset before a masked reset or `step`.

### Step return values

`step` returns
`(next_observation, reward, terminated, truncated, next_info)`. It evaluates
the reward from the observation captured before the action and the observation
after the action. The internal `_get_reward` helper performs this score
difference calculation, including on the final transition.

## Action preconditions and step behavior

The caller must supply valid actions. On each active continuing row,
`exchanges` contains only `-1`, `0`, and `1`. A `-1` removes a selected point;
a `1` adds a point that was unselected before this call. The counts on both
sides must be equal and at most `min(max_exchanges, K, N - K)`.

An all-zero exchange with `stop=False` is a valid no-op. A valid `stop=True`
requires a feasible current selection and an all-zero exchange. Reaching
feasibility alone does not terminate an episode. Valid exchanges may worsen
the score or move a feasible row back into infeasibility.

`step` applies each continuing row's edits simultaneously. It subtracts one
from `steps_remaining` on every row active before the call, including stops
and no-ops. Stopping rows preserve their selection and set `info["stopped"]`.
A continuing row that reaches zero remaining decisions sets `info["timed_out"]`.
A stop on the final available decision records stopped only.

Both completion reasons set `terminated=True`. The return has
`truncated=False` for all rows because there is no external cutoff separate
from the intrinsic task horizon. Finished rows stay frozen until reset:
later calls preserve their selection, remaining count, and completion reason.
Their action contents are ignored and their reward is zero.

The [score and reward definitions](../research/environment.md#score-and-reward)
apply on every transition, including the final one. Feasibility uses a direct
`>= threshold` comparison in the working floating dtype, with no tolerance.
Signals are recomputed from the complete selection rather than incrementally
updated. The return identity holds up to floating-point roundoff.

The final selection is the episode result. The implementation does not keep an
earlier best selection or add a computation-cost reward for stopping.
`step` does not validate ternary values, dtype, membership, equal counts, the
exchange cap, or stop preconditions. It ignores edits on a stopping row even
though the valid-action contract requires zero edits. Invalid calls have no
promised result or exception behavior.

## Reset semantics

### Full and masked reset

A full reset samples a new $K$-element emitter subset independently in every
row. Each subset is uniform among all $K$-subsets and is sampled without
replacement. A row with $K=0$ receives an all-false selection, and a row with
$K=N$ receives an all-true selection. Reset does not reject a selection that
is infeasible.

A masked reset samples new selections only where `mask` is true. Those rows
receive $h=T$ and have both completion flags cleared. Rows where `mask` is
false preserve their dynamic state and completion reasons, including active,
stopped, and timed-out rows. Selecting an active row discards its unfinished
episode. Callers normally use `terminated | truncated` as the mask.

The returned batch includes all rows after either reset form. The reset result
for a selected row is the observation for its new episode. A reset result for
an unselected row remains the observation for the episode already in progress
or the frozen terminal state.

An all-false mask is a no-op. It leaves the state and the environment-local
random stream unchanged, even when `seed` is supplied. A nonempty reset with
`seed=...` reseeds the one local `torch.Generator` before sampling the selected
rows. Without a seed override, a reset consumes random draws only when
$0<K<N$. The $K=0$ and $K=N$ branches construct deterministic selections
without a draw. The same seed and sequence of reset masks reproduce the same
selections on a fixed device and software setup. The implementation makes no
CPU-to-CUDA sampling parity promise.

The generator is local to the environment, so reset sampling does not consume
the process-wide PyTorch random stream. A seed override changes the shared
generator's future state, including future resets of rows not selected by the
current mask.

### Snapshot ownership

Construction detaches and copies `points` and `weights` onto the configured
device, then stores a detached copy of the contribution matrix. Mutating the
original input tensors after construction does not change the instance.

`points`, `weights`, and `contributions` in an observation are borrowed fixed
tensors. They share storage across observations and must be treated as
read-only by the caller. The environment returns independent dynamic snapshots
for `selected`, `steps_remaining`, and completion flags. The signal and scalar
measurements are recomputed outputs. Public environment operations run without
autograd tracking.

## Step and collection lifecycle

The collection loop must record the observation returned by `step` before
resetting completed rows. A masked reset can change a selected row immediately,
so the reset result is the next episode's observation rather than the previous
transition's final observation.

```text
observation, info = environment.reset()
        |
        v
action = policy(observation)
        |
        v
next_observation, reward, terminated, truncated, next_info = \
    environment.step(action)
        |
        +-> store (observation, action, reward, next_observation, ...)
        |
        +-> done = terminated | truncated
        |
        +-> observation, info = environment.reset(mask=done)
```

When a batch contains frozen rows, define
`active = ~(info["stopped"] | info["timed_out"])` before the `step` call and store
transitions only for those rows. Repeated calls on frozen rows
produce repeated terminal observations and zero rewards, not new environment
transitions. There is no automatic restart inside `step`.

## A numerical transition

The following reachable transition illustrates the score crossing from
infeasible to feasible. Its values are an example, not configuration settings.
The same labeled transition is checked by
[`test_worked_transition_cycle_and_snapshots`](../../tests/test_env.py).

Take four one-dimensional points at `0, 0.5, 1, 1.5`, unit weights, $K=2$,
and the illustrative threshold $\tau=0.5$. Use the illustrative decay
$k(d)=\max(0,1-d)$. The contribution matrix is

```text
A = [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.5, 0.0],
     [0.0, 0.5, 1.0, 0.5],
     [0.0, 0.0, 0.5, 1.0]]
```

Both rows below are active with more than one decision remaining. Row 0 makes
one valid exchange. Row 1 submits a valid stop.

| Quantity | Row 0: exchange | Row 1: stop |
| --- | --- | --- |
| Selection before | `[1,1,0,0]` | `[1,0,0,1]` |
| Signal before | `[1.5,1.5,0.5,0]` | `[1,0.5,0.5,1]` |
| Score before | `-0.5` | `3.0` |
| Exchange | `[0,-1,0,+1]` | `[0,0,0,0]` |
| Stop request | `False` | `True` |
| Selection after | `[1,0,0,1]` | `[1,0,0,1]` |
| Signal after | `[1,0.5,0.5,1]` | `[1,0.5,0.5,1]` |
| Score after | `3.0` | `3.0` |
| Reward | `3.5` | `0.0` |
| Terminated | `False` | `True` |
| Truncated | `False` | `False` |

Row 0 repairs the underserved receiver and remains active. Row 1 freezes after
the stop. The reward for row 0 is the full score change, including the
infeasible-to-feasible transition.

## Configuration and executable example

All configuration fields are required and have no implicit defaults. The
checked-in example values are in [`env.yaml`](../../env.yaml).

| Configuration field | Symbol | Usage |
| --- | --- | --- |
| `num_emitters` | $K$ | Number of selected emitters in every row |
| `max_exchanges` | $M$ | Maximum number of exchanged pairs per continuing decision |
| `max_steps` | $T$ | Intrinsic decision horizon for each reset row |
| `threshold` | $\tau$ | Common minimum received signal |
| `seed` | | Initial seed for the environment-local generator |
| `device` | | Destination CPU or CUDA device |

The input tensors determine $B$, $N$, and $D$. The supplied `decay`
callable determines the decay law and any decay-specific parameters.

Run this example from the repository root:

```python
import torch

from env import BatchedEmitterEnv, load_config

config = load_config("env.yaml")
points = torch.tensor([[[0.0], [0.5], [1.0], [1.5]]])  # B=1, N=4, D=1
weights = torch.ones(1, 4)
environment = BatchedEmitterEnv(
    points, weights, lambda distance: (1 - distance).clamp_min(0), config
)
observation, info = environment.reset()

# A no-op preserves the selection and consumes one decision.
action = {
    "exchanges": torch.zeros_like(observation["selected"], dtype=torch.int64),
    "stop": torch.zeros_like(observation["feasible"]),
}
next_observation, reward, terminated, truncated, info = environment.step(action)
done = terminated | truncated

# Record the actual next observation before resetting completed rows.
transition = (observation, action, reward, next_observation, terminated, truncated)
observation, info = environment.reset(mask=done)
```

The caller supplies finite float32 or float64 coordinates and matching finite
weights with nonnegative values. The decay callable is deterministic,
elementwise nonnegative, nonincreasing, finite at zero, and returns the same
shape, dtype, and device as its distance input. The configuration satisfies
$0\leq K\leq N$, $M\geq0$, $T>0$, and finite nonnegative $\tau$.
Dimensions are positive, and all action and mask tensors have the documented
batch shapes on the configured device. The caller invokes `reset()` before
`step()` and follows the valid-action contract. These assumptions are not
validated; invalid calls have no promised result or exception behavior.
