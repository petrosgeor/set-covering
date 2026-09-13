# Searching through emitter exchanges

An episode starts with a selection of $K$ emitters and gives the policy at most
$T$ decisions to change it. Each continuing action exchanges equally many
selected and unselected points. The final selection determines the outcome.

The [problem formulation](problem.md) defines the instance and utility. The
[environment](../../envs/emitter.py) implements this process, and its [code
reference](../code/environment.md) lists the public interface.

## A fixed instance and a changing selection

Point positions, weights, demands, contribution matrix $A$, budget $K$, and
horizon remain fixed during an episode. The environment computes
$A_{ij}=\exp(-\lVert x_i-x_j\rVert_2)$ from the supplied coordinates. At
decision $t$, the selection is $z_t$ and received signal is $r_t=Az_t$.

The dynamic state at decision $t$ is

$$
s_t=(z_t,h_t,q_t),
$$

where $h_t$ is the number of decisions left and $q_t$ is active, stopped, or
timed out. Signal is part of the observation; capped utility and unmet demand
are diagnostics. The [policy](model.md) receives observations while active.

The horizon satisfies $T>0$, the exchange cap is $M\geq0$, and $K$ constrains
every selection.

## Initialization

Initialization samples uniformly from the selections containing exactly $K$
emitters:

$$
z_0\sim\operatorname{Uniform}\left(
\{z\in\{0,1\}^N:\mathbf 1^\top z=K\}\right),
\qquad h_0=T,\qquad q_0=\mathrm{active}.
$$

The sample does not depend on demand satisfaction, so unmet demand at the start
is allowed. For $K=0$ or $K=N$, only one selection exists. Restarting samples
again from the same fixed instance.

## One continuing action

A continuing action removes $m$ selected emitters and adds $m$ unselected
points, where

$$
0\leq m\leq m_{\max},\qquad m_{\max}=\min(M,K,N-K).
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

Choosing $m=0$ gives the continuing no-op $a_t=0$: the selection is unchanged,
but one decision is consumed. It is the only continuing exchange when $M=0$,
$K=0$, or $K=N$.

Stopping is separate from continuing and ends the episode with the current
selection. It has zero exchanges. Demand satisfaction does not stop an active
episode, which may stop or continue from any demand state. A learned stop does
not certify optimality.

## Utility and reward

The environment evaluates $F(z)$ and reports $U(z)$ from the [problem
document](problem.md#capped-utility-and-unmet-demand), but reward uses only the
utility difference:

$$
R_t=F(z_{t+1})-F(z_t).
$$

Stops and no-ops receive zero reward. A worsening exchange receives negative
reward, including when it is the final decision.

## Decision budget and completion

Every active decision, including a stop or no-op, consumes one unit:

$$
h_{t+1}=h_t-1.
$$

A stop sets the completion reason to stopped. A continuing action using the last
decision sets it to timed out; a stop on the last decision takes precedence.
Both reasons end the episode. The horizon is part of the task, with no separate
external cutoff. After completion, the state stays fixed until initialization;
further calls return zero reward and keep the completion reason.

A timeout may leave unmet demand. The environment keeps the final selection
rather than restoring an earlier one, so final utility reports the search
outcome and does not establish optimality.

## Return over an episode

For undiscounted rewards and an episode ending after
$t_{\mathrm{end}}\leq T$ decisions,

$$
\begin{aligned}
\sum_{t=0}^{t_{\mathrm{end}}-1}R_t
&=\sum_{t=0}^{t_{\mathrm{end}}-1}
\left[F(z_{t+1})-F(z_t)\right]\\
&=F(z_{t_{\mathrm{end}}})-F(z_0).
\end{aligned}
$$

For a fixed initial selection, maximizing return is the same as maximizing final
utility. There is no computation-cost reward for stopping early. The proposed
policy uses $\gamma=1$.
