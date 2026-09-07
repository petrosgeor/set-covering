# Searching through emitter exchanges

An episode starts with a selection of $K$ emitters and gives the policy at most
$T$ decisions to change it. Each continuing action exchanges equally many
selected and unselected points. The final selection determines the outcome.

The [problem formulation](problem.md) defines the instance, capped utility, and
unmet-demand diagnostic. This document explains the episode process realized by the
[environment implementation](../../envs/emitter.py). Its [code reference](../code/environment.md)
contains the public interface and execution details.

## A fixed instance and a changing selection

The point positions, weights, demands, decay function, contribution matrix $A$,
budget $K$, and horizon stay fixed throughout an episode. At decision time $t$,
the selection is $z_t$ and its signal is $r_t=Az_t$.

Two additional quantities describe the episode's progress: the remaining
number of decisions $h_t$ and a completion reason $q_t$. The completion reason
is active, stopped, or timed out. The dynamic state is therefore

$$
s_t=(z_t,h_t,q_t).
$$

The fixed instance supplies the context for interpreting this state. Signal is
part of the observation, while capped utility and unmet demand are diagnostics
reported with it. The [policy](model.md) receives the observation when active.

The horizon $T>0$ is a parameter of the search process. Another parameter,
$M\geq0$, limits the number of exchanged pairs in one decision. The emitter
budget $K$ constrains every selection in the episode.

## Initialization

Initialization samples uniformly from the selections containing exactly $K$
emitters:

$$
z_0\sim\operatorname{Uniform}\left(
\{z\in\{0,1\}^N:\mathbf 1^\top z=K\}\right),
\qquad h_0=T,\qquad q_0=\mathrm{active}.
$$

This distribution does not condition on demand satisfaction. An episode may
start with unmet demand. When $K=0$ or $K=N$, there is only one possible
selection. Restarting an episode samples a selection for the same fixed instance.

We follow the [four-point example](problem.md#signal-from-the-selected-emitters)
from the problem document. Suppose initialization gives $z_0=(1,1,0,0)^\top$.
Its signal is $(1.5,1.5,0.5,0)^\top$, with $F(z_0)=2.75$ and
$U(z_0)=1.25$.

## One continuing action

A continuing action chooses $m$ emitters to remove and $m$ previously
unselected points to add. The count satisfies

$$
0\leq m\leq m_{\max},\qquad m_{\max}=\min(M,K,N-K).
$$

The cap accounts for the allowed exchange size and the number of available
points on each side. Let $\mathcal R_t$ be the removal set and $\mathcal C_t$
the addition set. Valid choices satisfy

$$
\mathcal R_t\subseteq\{i:z_{t,i}=1\},\qquad
\mathcal C_t\subseteq\{i:z_{t,i}=0\},\qquad
|\mathcal R_t|=|\mathcal C_t|=m.
$$

Represent the complete exchange as a signed vector:

$$
a_{t,i}=\begin{cases}
-1,&i\in\mathcal R_t,\\
+1,&i\in\mathcal C_t,\\
0,&\text{otherwise}.
\end{cases}
$$

All edits take effect together:

$$
z_{t+1}=z_t+a_t.
$$

Equal removal and addition counts preserve $\mathbf 1^\top z_{t+1}=K$.
The policy may choose the points sequentially while constructing this vector,
but the environment makes one transition after the complete action is ready.

For the running example, remove point 2 and add point 4, assuming $M\geq1$:

```text
Selection before: [1, 1, 0, 0]
Complete exchange: [0,-1, 0,+1]
                   -----------
Selection after:  [1, 0, 0, 1]
Signal after:     [1, 0.5, 0.5, 1]
```

The new selection has $F(z_1)=3$ and $U(z_1)=1$. Other valid exchanges can
reduce capped utility or increase unmet demand. Validity constrains which edits
can be submitted; the policy must learn which are useful.

## Continuing without edits and stopping

Choosing $m=0$ gives $a_t=0$. This continuing no-op leaves the selection
unchanged and consumes one decision. If $M=0$, $K=0$, or $K=N$, it is the only
possible continuing exchange.

Stopping is a separate decision that ends the episode with the current
selection. A valid stop has zero exchanges. Demand satisfaction does not end an
episode, and an active episode may stop or continue from any demand state.

After the example's exchange, the policy could stop at $(1,0,0,1)^\top$ or
continue searching. A learned stop decision does not certify optimality.

## Utility and reward

The environment evaluates every selection with the capped utility $F(z)$ and
reports weighted unmet demand $U(z)$ from the [problem document](problem.md#capped-utility-and-unmet-demand).
It uses only the utility for reward. One complete action earns the utility
difference:

$$
R_t=F(z_{t+1})-F(z_t).
$$

For the running example, removing point 2 and adding point 4 changes the
utility from $2.75$ to $3$. The reward is therefore $3-2.75=0.25$.

| Quantity | Before the exchange | After the exchange |
| --- | --- | --- |
| Capped utility $F$ | $2.75$ | $3$ |
| Weighted unmet demand $U$ | $1.25$ | $1$ |
| Reward | - | $0.25$ |

Stops and continuing no-ops receive zero reward because they leave the
selection unchanged. A worsening exchange receives negative reward. The same
utility-difference rule applies to the final decision.

## Decision budget and completion

Every decision made while active consumes one unit of budget, including a stop
or no-op:

$$
h_{t+1}=h_t-1.
$$

A stop sets the completion reason to stopped. A continuing action that uses the
last decision sets it to timed out. A stop on the last available decision takes
precedence and records the reason as stopped.

```text
Initialize selection, h = T
             |
             v
       Active state <--------------------+
             |                           |
        Choose action                    |
             |                           |
       +-----+------+                    |
       |            |                    |
      stop       continue                |
       |            |                    |
Keep selection   Apply complete exchange  |
       |            |                    |
   h = h - 1    h = h - 1                 |
       |            |                    |
    stopped      h == 0? ---- no ---------+
                    |
                   yes
                    |
                 timed out
```

Both reasons end the episode. The horizon is part of the task, and there is no
separate external cutoff. Once completed, the state remains fixed until
initialization starts another episode. Further calls on a completed state
produce zero reward and preserve the completion reason.

A timeout may leave unmet demand. The episode keeps its final selection; it does
not recover an earlier best selection. The final utility records the search
outcome and does not prove that the instance has no better selection.

## Return over an episode

With undiscounted rewards, the intermediate utility values cancel. If the episode ends
after $t_{\mathrm{end}}\leq T$ decisions,

$$
\begin{aligned}
\sum_{t=0}^{t_{\mathrm{end}}-1}R_t
&=\sum_{t=0}^{t_{\mathrm{end}}-1}
\left[F(z_{t+1})-F(z_t)\right]\\
&=F(z_{t_{\mathrm{end}}})-F(z_0).
\end{aligned}
$$

For a fixed initial selection, maximizing this return maximizes the final utility.
There is no extra computation-cost reward for stopping early. The proposed
policy uses discount factor $\gamma=1$ to retain this relationship.

The [model walkthrough](model.md) follows how one observation becomes an
exchange and how its probability connects to this episode-level return.

## Notation reference

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $t$ | Environment decision index | $0,\ldots,t_{\mathrm{end}}-1$ |
| $T,h_t$ | Initial horizon and remaining decisions | Integers, $0\leq h_t\leq T$ |
| $q_t$ | Completion reason | Active, stopped, or timed out |
| $s_t$ | Dynamic state in the fixed instance | $(z_t,h_t,q_t)$ |
| $M,m$ | Maximum and chosen exchange counts | Integers, $0\leq m\leq\min(M,K,N-K)$ |
| $\mathcal R_t,\mathcal C_t$ | Removal and addition sets | Each contains $m$ points |
| $a_t$ | Complete signed exchange | $\{-1,0,1\}^N$ |
| $F(z),U(z)$ | Capped utility and weighted unmet demand | Scalars |
| $R_t$ | Reward for one complete action | $F(z_{t+1})-F(z_t)$ |
| $t_{\mathrm{end}}$ | Number of decisions before completion | Integer in $[1,T]$ |
