# Searching through emitter exchanges

An episode starts with a selection of $K$ emitters. The policy can change that
selection through complete exchanges. The [problem formulation](problem.md)
defines the instance and utility; the [environment](../../envs/emitter.py)
implements the transitions, and its [code reference](../code/environment.md)
lists the public interface.

There is no fixed task horizon. `max_steps` is an external collection limit
used to bound one run. The policy does not receive that limit or the remaining
number of collection slots.

## A fixed instance and a changing selection

Point positions, weights, demands, contribution matrix $A$, and budget $K$
remain fixed during an episode. The environment computes
$A_{ij}=\exp(-\lVert x_i-x_j\rVert_2)$ from the supplied coordinates. At
decision $t$, the selection is $z_t$ and received signal is $r_t=Az_t$.

The dynamic state is

$$
s_t=(z_t,q_t),
$$

where $q_t$ is active, stopped, or succeeded. The selection and
signal are observations. The [policy](model.md) receives observations while
active. A timeout belongs to collection bookkeeping rather than the task state.
The collection counter is kept in `info`, not in the policy observation. Capped
utility and weighted unmet demand are diagnostics computed from the current
observation.

The exchange cap is $M\geq0$, and $K$ constrains every selection. The external
collection limit $T=$`max_steps` is separate from the task state.

## Initialization

Initialization samples uniformly from selections containing exactly $K$
emitters:

$$
z_0\sim\operatorname{Uniform}\left(
\{z\in\{0,1\}^N:\mathbf 1^\top z=K\}\right).
$$

The sample remains uniform even when it already satisfies demand. The
environment marks such a row as `succeeded` and freezes it immediately. The
success test is

$$
U(z_0)\leq\varepsilon,\qquad \varepsilon=10^{-6},
$$

where $U$ is weighted unmet demand in raw objective units. The tolerance is a
numerical success criterion, not an exact mathematical certificate. Receivers
with zero weight do not affect it, so an instance with zero total weighted
demand succeeds at reset. The result is reported in `info`; a caller should
skip the row, and a later `step()` returns `terminated=True` with zero reward.

For $K=0$ or $K=N$, only one selection exists. Restarting samples again from
the same fixed instance.

## One continuing action

A continuing action removes $m$ selected emitters and adds $m$ unselected
points, where

$$
0\leq m\leq m_{\max},\qquad
m_{\max}=\min(M,K,N-K).
$$

If $\mathcal R_t$ and $\mathcal C_t$ are the removal and addition sets, valid
choices satisfy

$$
\mathcal R_t\subseteq\{i:z_{t,i}=1\},\qquad
\mathcal C_t\subseteq\{i:z_{t,i}=0\},\qquad
|\mathcal R_t|=|\mathcal C_t|=m.
$$

The complete exchange is the signed vector

$$
a_{t,i}=\begin{cases}
-1,&i\in\mathcal R_t,\\
+1,&i\in\mathcal C_t,\\
0,&\text{otherwise}.
\end{cases}
$$

The environment applies all edits together:

$$
z_{t+1}=z_t+a_t.
$$

Equal counts preserve $\mathbf 1^\top z_{t+1}=K$. The policy may choose points
sequentially while building $a_t$, but the environment waits for the complete
exchange before transitioning.

## Continuing without edits and stopping

Choosing $m=0$ gives the continuing no-op $a_t=0$. It leaves the selection
unchanged, consumes one collection slot, and still incurs the search cost. It
is the only continuing exchange when $M=0$, $K=0$, or $K=N$.

Stopping is separate from continuing. It submits a zero exchange, ends the
episode with the current selection, and receives zero reward. A continuing
exchange that reaches the numerical success threshold also ends the episode.
A no-op or a move with no immediate improvement does not establish that the
selection is optimal.

## Utility and reward

The environment reports $F(z)$ and $U(z)$ from the [problem
document](problem.md#capped-utility-and-unmet-demand). For an active row, the
reward is

$$
R_t=\begin{cases}
0,&\text{if the policy stops},\\
F(z_{t+1})-F(z_t)-c,&\text{if the policy continues},
\end{cases}
\qquad c>0.
$$

The cost $c$ is `config.step_cost`, a positive finite value in the same units
as $F$. It applies to every continuation, including a no-op and an exchange
that reaches success or the collection limit. Calls on frozen rows return
zero reward.

## Decision budget and completion

Each active call to `step()` consumes one external collection slot. The internal
counter is returned as `info["steps_remaining"]`; it is not part of the policy
observation.

A stop sets the completion reason to `stopped`. A successful continuing action
sets it to `succeeded`. If a continuing action reaches the external limit
without either condition, the reason is `timed_out`. The environment returns

$$
\texttt{terminated}=\texttt{stopped}\lor\texttt{succeeded},\qquad
\texttt{truncated}=\texttt{timed\_out}.
$$

Stop or success takes precedence when it occurs on the last collection slot.
Completed rows keep their final selection and observation until reset. A
masked reset restarts only the requested rows and returns the full batch; call
`reset(mask=terminated | truncated)` after saving the final observations.

An actor-critic collector resets on `terminated | truncated`. It suppresses the
bootstrap term for a terminated transition. For a truncated transition, it
bootstraps from the final observation returned before reset. Return calculations
must not cross a reset boundary into the next episode.

## Return over an episode

For an episode that reaches stop or numerical success after $L$ continuing
actions, the undiscounted return is

$$
\begin{aligned}
\sum_t R_t
&=\sum_t\left[F(z_{t+1})-F(z_t)\right]-cL\\
&=F(z_{\mathrm{final}})-F(z_0)-cL.
\end{aligned}
$$

The planned first trainer uses $\gamma=1$. The fixed per-decision cost gives
the policy a reason to stop searching, but it is a proxy for search effort, not
a measurement of runtime. Larger exchanges use more internal point choices, so
evaluation must measure computational cost separately.
