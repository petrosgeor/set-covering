# A transformer and pointer policy for emitter exchanges

The [complete policy](../code/policy.md) combines an encoder, policy heads,
and pointer decoder. It chooses stop or continue, chooses an exchange count,
and constructs one environment action with its ordered trace. The
[encoder implementation](../code/transformer_encoder.md), [policy heads](../../models/policy_heads.py),
and [pointer decoder](../../models/pointer_decoder.py) are separate modules;
the complete policy assembles them and evaluates supplied traces. A2C training
is planned but remains unimplemented. Model and feed-forward widths, encoder
depth, and attention head count are configuration parameters. The
[problem](problem.md) and [environment](environment.md) define the task and
transition semantics.

During one action, the observation stays fixed while the decoder constructs
the exchange. The environment changes the selection only after construction
is complete.

## The path through one decision

```text
Observation
    -> point features [B,N,D+4] -> shared projection
    -> transformer -> point representations H [B,N,d]
    -> mean(H) -> global representation g [B,d]
g -> value estimate
g -> stop -> zero exchange and episode termination
  continue -> count m -> no-op when m=0, otherwise GRU pointer
                         -> m removals, then m additions
                         -> one signed exchange and environment transition
```

Here $N$ is the number of points, $D$ the coordinate dimension, $d$ the
representation width, and $B$ the batch size.

## 1. Describe each point

At the current decision, point $i$ has coordinates $x_i$, receiver weight
$w_i$, selection indicator $z_i$, received signal $r_i$, and demand $u_i$.
The indicator is encoded as zero or one, and demand enters through the signed
margin in

$$
v_i=\left[x_{i,:},w_i,z_i,r_i,r_i-u_i\right]
\in\mathbb R^{D+4}.
$$

The margin is negative below demand, zero at the cap, and positive above it.
The environment computes $r_i$ before the policy receives the observation.
The model receives coordinates and current signal, not the contribution matrix
$A$; the fixed law is
$A_{ij}=\exp(-\lVert x_i-x_j\rVert_2)$.

## 2. Embed each point with a shared projection

Each feature vector passes through the same two-layer projection:

$$
e_i=\phi_{\mathrm{in}}(v_i)\in\mathbb R^d,
\qquad
\phi_{\mathrm{in}}:\mathbb R^{D+4}\to\mathbb R^d.
$$

The affine widths are $D+4\to d\to d$, with GELU between them. Stacking the
embeddings gives $E\in\mathbb R^{N\times d}$. Before attention, $e_i$ depends
only on point $i$'s features.

## 3. Let the point representations interact

The transformer produces contextualized point representations:

$$
H=\operatorname{TransformerEncoder}_\theta(E)
\in\mathbb R^{N\times d}.
$$

Each row can depend on every point, including itself. Layers use full
self-attention, residual connections, pre-normalization, a GELU feed-forward
network, and zero dropout. Each layer is initialized independently, and a
final layer normalization produces $H$. There is no causal mask or positional
encoding tied to arbitrary point index; coordinates carry spatial information.
Reordering points reorders $H$ and leaves the pooled global representation
unchanged up to floating-point effects. The implementation uses
$d_a=d_v=d/h_{\mathrm{enc}}$ for each attention head.

## 4. Summarize the current configuration

Mean pooling gives

$$
\bar h=\frac1N\sum_{i=1}^N H_{i,:}\in\mathbb R^d.
$$

The policy retains $H$ for the pointer decoder and projects $\bar h$ into the
global representation:

$$
g=\psi(\bar h)\in\mathbb R^d,
\qquad \psi:\mathbb R^d\to\mathbb R^d.
$$

The global projection has widths $d\to d\to d$ with GELU between layers. The
heads receive a summary of the current configuration, without the external
collection counter.

## 5. Decide whether to stop

The stop head produces logits for continue and stop:

$$
\lambda^{\mathrm{stop}}=gW_{\mathrm{stop}}+\beta_{\mathrm{stop}}
\in\mathbb R^2,
\qquad W_{\mathrm{stop}}\in\mathbb R^{d\times2}.
$$

The policy samples from the categorical distribution defined by these logits.
Both choices remain available in every demand state. Stop emits a zero
exchange and ends action construction; the count and point choices are
skipped. The [environment](environment.md#continuing-without-edits-and-stopping)
determines the completion reason.

## 6. Choose how many pairs to exchange

On the continue branch, the count head produces one logit for each
$m\in\{0,\ldots,M\}$:

$$
\lambda^{\mathrm{count}}=gW_{\mathrm{count}}+\beta_{\mathrm{count}}
\in\mathbb R^{M+1},
\qquad W_{\mathrm{count}}\in\mathbb R^{d\times(M+1)}.
$$

The legal maximum is

$$
m_{\max}=\min(M,K,N-K).
$$

Counts above $m_{\max}$ receive $-\infty$ before the categorical distribution
is formed. The selected count means remove $m$ points selected at the start
of the action and add $m$ points unselected at the start. A zero count
produces a continuing no-op; it does not stop the episode. For $M=0$, $K=0$,
or $K=N$, zero is the only continuing count.

## 7. Initialize the decoder's memory

The decoder embeds the selected count and combines it with the global state:

$$
e_m=\operatorname{Embed}_m(m)\in\mathbb R^d,\qquad
c_0=\operatorname{MLP}_c([g,e_m])\in\mathbb R^d,
$$

where $\operatorname{MLP}_c:\mathbb R^{2d}\to\mathbb R^d$. The memory is fresh
for every environment action, not carried across decisions. For $m>0$, the
decoder completes all removals before starting additions.

## 8. Decode removals and additions

For $m>0$, internal choice $\ell\in\{1,\ldots,2m\}$ has a phase and a
remaining-choice count:

$$
\phi_\ell=
\begin{cases}
\mathrm{remove},&1\leq\ell\leq m,\\
\mathrm{add},&m<\ell\leq2m,
\end{cases}
\qquad
n_\ell=
\begin{cases}
m-\ell+1,&1\leq\ell\leq m,\\
2m-\ell+1,&m<\ell\leq2m.
\end{cases}
$$

At the first step, $c_{\ell-1}=c_0$, $\phi_\ell=\mathrm{remove}$, and
$n_\ell/m=1$. The query network maps $\mathbb R^{2d+1}$ to $\mathbb R^d$.
With $\kappa_i=H_{i,:}W_K$ and
$W_Q,W_K\in\mathbb R^{d\times d_k}$, each step computes

$$
q_\ell=\operatorname{MLP}_q([c_{\ell-1},e_{\phi_\ell},n_\ell/m]),
\qquad
L_{\ell,i}=\frac{\langle q_\ell W_Q,\kappa_i\rangle}{\sqrt{d_k}},
$$

$$
c_\ell=\operatorname{GRUCell}_\theta(
[H_{j_\ell,:},e_{\phi_\ell}],c_{\ell-1}).
$$

The legal points are

$$
\mathcal I_\ell=
\begin{cases}
\{i:z_i=1\}\setminus\{j_1,\ldots,j_{\ell-1}\},&1\leq\ell\leq m,\\
\{i:z_i=0\}\setminus\{j_{m+1},\ldots,j_{\ell-1}\},&m<\ell\leq2m.
\end{cases}
$$

Set the mask to $\mu_{\ell,i}=0$ for $i\in\mathcal I_\ell$ and
$-\infty$ otherwise. The point distribution is

$$
P_\theta(j_\ell=i\mid o,\mathrm{continue},m,j_{<\ell})
=\operatorname{softmax}_i(L_\ell+\mu_\ell).
$$

The same GRU memory carries from removals into additions, while $H$ stays
fixed. Its input has width $2d$ and hidden state width $d$. The count cap
guarantees enough eligible points for both phases. A zero count skips this
loop.

## 9. Submit one complete exchange

The ordered choices define
$\mathcal R=\{j_1,\ldots,j_m\}$ and
$\mathcal C=\{j_{m+1},\ldots,j_{2m}\}$. The policy converts them to the
[signed exchange](environment.md#one-continuing-action), which the caller
submits with continue. The environment applies all edits simultaneously:

$$
a_i=
\begin{cases}
-1,&i\in\mathcal R,\\
+1,&i\in\mathcal C,\\
0,&\text{otherwise}.
\end{cases}
\qquad
z_{\mathrm{next}}=z+a.
$$

Stop instead submits a zero exchange and ends the episode. A continuing
no-op also has a zero exchange, but consumes a decision. The next observation
runs the encoder again and starts a new decoder memory. The environment emits
one reward for the complete action, not for its internal point choices. See the
[environment reward definition](environment.md#utility-and-reward) for the
transition reward.

## 10. Assign a probability to the complete choice sequence

An ordered trace has the branch, count, and point choices:

$$
\xi=(\mathrm{branch},m,j_1,\ldots,j_{2m})
$$

for $m>0$. Its probability is

$$
\pi_\theta(\xi\mid o)=P_\theta(\mathrm{continue}\mid o)
P_\theta(m\mid o,\mathrm{continue})
\prod_{\ell=1}^{2m}
P_\theta(j_\ell\mid o,\mathrm{continue},m,j_{<\ell}).
$$

The policy stores the log probability:

$$
\begin{aligned}
\log\pi_\theta(\xi\mid o)
={}&\log P_\theta(\mathrm{continue}\mid o)
+\log P_\theta(m\mid o,\mathrm{continue})\\
&+\sum_{\ell=1}^{2m}
\log P_\theta(j_\ell\mid o,\mathrm{continue},m,j_{<\ell}).
\end{aligned}
$$

A stop trace contributes only $\log P_\theta(\mathrm{stop}\mid o)$. A
continuing zero-count trace contributes the continue and count-zero terms.
Policy evaluation recomputes these terms from the original observation and
supplied ordered trace without sampling.

Different point orders can produce the same exchange sets. The planned training
interface scores the sampled ordered trace. A2C must replay that order with the
same observation and masks. An unordered exchange probability would require
summing over every trace that yields the exchange; the baseline does not do
that.

## 11. Connect the trace to A2C and the value estimate

One environment action receives one reward, even when its trace has several
internal choices. At environment time $t$, the observation is $o_t$, the
trace is $\xi_t$, and the completed action receives $R_t$. The value head uses
the same global representation:

$$
V_\theta(o_t)=g_tw_V+\beta_V\in\mathbb R,
\qquad w_V\in\mathbb R^d.
$$

It estimates expected future net return, including the search cost charged by
continuing actions. For a trajectory that reaches stop or success, with
$\gamma=1$,

$$
V^\pi(o_t)=\mathbb E_\pi\left[
\sum_{j=t}^{t_{\mathrm{end}}-1}R_j\,\middle|\,o_t\right]
=\mathbb E_\pi\left[
F(z_{t_{\mathrm{end}}})-F(z_t)-cL_t\,\middle|\,o_t\right],
$$

where $L_t$ is the number of continuing actions from $t$ through the end of the
episode. The relationship follows from the [return
definition](environment.md#return-over-an-episode). The value is an estimate of
future net return, not a certificate of the best solution.

The first planned trainer uses A2C. A rollout records the original observation,
the ordered trace, its complete-action reward, and the termination flags. The
policy reevaluates the same trace under the current parameters and uses its log
probability and value to form the actor and critic updates. A true terminal
transition has no bootstrap term. A truncated transition bootstraps from the
final observation returned before reset. A rollout is reset on
`terminated | truncated`, and its return calculation cannot use rewards from
the next episode.

The advantage belongs to the complete environment action, even when its trace
contains several point choices. The exact advantage estimator, entropy term,
and optimizer settings remain unfinished; this design fixes the trace
likelihood and value interfaces.

## Design rationale and limitations

The encoder supplies configuration-aware point representations, while the
GRU decoder makes later choices depend on earlier choices. The policy avoids
enumerating every unordered exchange set, whose count is

$$
\sum_{m=0}^{m_{\max}}\binom Km\binom{N-K}m.
$$

For a positive count it makes $2m$ point choices, each scoring $N$ points.
Dense encoder attention still forms $N^2$ interactions per head and layer, and
the environment stores $N^2$ contributions. Avoiding exchange enumeration does
not establish a wall-clock speedup.

The masks enforce membership and distinct choices; the decoder removes and adds
exactly $m$ points. These constraints do not guarantee utility improvement or
demand satisfaction. The fixed-width GRU memory may limit the decoder for large
exchanges. Final solution quality and generalization require experiments under
a stated inference budget; the architecture has no measured performance claim.

## Notation reference

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $o$ | Active observation | Fixed instance and current episode quantities |
| $N,D,K,M,T$ | Problem parameters and external collection limit | Defined in [problem](problem.md) and [environment](environment.md); $T$ is not encoded |
| $u$ | Receiver demand caps | $\mathbb R_{\geq0}^N$ |
| $c$ | Cost of one continuing action | Positive scalar in utility units |
| $v_i$ | Features of point $i$ | $\mathbb R^{D+4}$ |
| $E,H$ | Embedded and contextualized points | $\mathbb R^{N\times d}$ |
| $d,L_{\mathrm{enc}},h_{\mathrm{enc}}$ | Width, encoder depth, and head count | Symbolic positive integers |
| $d_a,d_v$ | Query/key and value widths of one attention head | Symbolic positive integers |
| $\bar h,g$ | Mean and global representations | $\mathbb R^d$ |
| $m$ | Sampled pair count | $0,\ldots,m_{\max}$ |
| $\ell,n_\ell$ | Internal choice and remaining choices in its phase | $1\leq\ell\leq2m$, $1\leq n_\ell\leq m$ |
| $e_m,e_{\phi_\ell}$ | Count and phase embeddings | $\mathbb R^d$ |
| $c_\ell,q_\ell$ | Decoder memory and query | $\mathbb R^d$ |
| $\kappa_i,\chi_\ell$ | Projected point key and query | $\mathbb R^{d_k}$ |
| $L_\ell,\mu_\ell$ | Point logits and eligibility mask | $N$ entries |
| $j_\ell$ | Chosen point | Integer in $\{1,\ldots,N\}$ |
| $\xi,\pi_\theta(\xi\mid o)$ | Ordered trace and probability | Branch-dependent sequence and scalar |
| $t$ | Environment decision index | Distinct from internal index $\ell$ |
| $V_\theta$ | Value estimate | Scalar |

## Architectural precedents

[Pointer Networks](https://arxiv.org/abs/1506.03134) provides the precedent for
selecting input positions with an attention pointer.
[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
provides related attention-based construction of combinatorial solutions.
These papers motivate the architecture but do not establish performance on
emitter placement.

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
describes a related likelihood-ratio method. The first planned trainer here is
A2C; the trace construction and environment reward define its proposed
connection.
