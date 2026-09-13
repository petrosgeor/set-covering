# Complete policy

[`EmitterPolicy`](../../models/policy.py) combines an encoder, decision heads, and
pointer decoder. Construct `EmitterPolicy(*, encoder, heads, decoder)` from
existing modules. The heads and decoder must share an exchange cap or
construction raises `ValueError`.

## Generation and replay

`policy(observation, *, greedy=False)` returns `(action, trace, log_prob, value)`.
`policy.evaluate(observation, trace)` returns `(log_prob, value)` without sampling.
Both preserve inputs and autograd. Here `B` counts decisions, `N` points, and
`M` is the exchange cap.

| Output | Fields | Shape and dtype |
| --- | --- | --- |
| Action | `exchanges`; `stop` | `[B,N]` int64; `[B]` Boolean |
| Trace | `stop`; `counts` | `[B]` Boolean; `[B]` int64 |
| Trace | `removals`, `additions` | `[B,M]` int64 |
| Scores | `log_prob`, `value` | `[B]`, matching model float |

Indices are zero-based, ordered, and padded with `-1` after each count.
Stops have count zero and zero exchanges. Continuing count zero is a no-op.
The complete log probability includes the branch, the count only when
continuing, and any point choices. Replay needs the original observation and
ordered trace; signed exchanges lose the order.

Greedy mode takes argmax decisions and scores the trace under the learned
distributions. The caller controls seeds, matching widths,
dtype/device, valid inputs, and completed rows. The policy never steps or resets
the environment. The action's stop tensor is independent of the retained trace.

## Usage

Use the [encoder example](transformer_encoder.md#usage) to supply
`encoder`, `config`, `observation`, and `env`:

```python
from models.policy import EmitterPolicy
from models.policy_heads import PolicyHeads
from models.pointer_decoder import PointerDecoder

d, M = encoder.config.model_dim, config.max_exchanges
policy = EmitterPolicy(
    encoder=encoder,
    heads=PolicyHeads(model_dim=d, max_exchanges=M),
    decoder=PointerDecoder(model_dim=d, max_exchanges=M),
).to(observation["points"])
with torch.no_grad():
    action, trace, log_prob, value = policy(observation)
    replay_log_prob, replay_value = policy.evaluate(observation, trace)
torch.testing.assert_close(log_prob, replay_log_prob)
next_observation, reward, terminated, truncated, info = env.step(action)
```
