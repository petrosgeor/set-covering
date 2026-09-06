# A transformer and pointer policy for emitter exchanges

The proposed policy takes an active environment observation and constructs one
complete action. It represents the points with a transformer, chooses whether
to stop and how many pairs to exchange, then chooses the exchange's members.

This architecture is a proposed PPO baseline and has not been implemented.
Model widths, encoder depth, and attention-head count remain parameters.
The [problem](problem.md) and [environment](environment.md) define what the
policy is trying to solve and how its actions change the selection.

The walkthrough follows one instance at one decision. We introduce each
component when its input is ready, then connect the complete action to training.
During action construction, the environment state stays fixed.

## The path through one decision

The policy first describes every point, then lets those descriptions interact.
A summary of the configuration supports the stop and exchange-count decisions.
If an exchange is needed, a pointer decoder selects the participating points.

```text
Point features [N,D+4]
          |
          v
Shared input projection [N,d]
          |
          v
Transformer encoder
          |
          v
Point representations H [N,d] ---------------------+
          |                                        |
          v                                        |
Mean pooling [d]                                   |
          |                                        |
Add remaining budget and feasibility               |
          |                                        |
          v                                        |
Global representation g [d]                        |
          |                                        |
          +--> Value estimate (used for training)   |
          |                                        |
          v                                        |
    Stop or continue                               |
     /          \                                  |
   stop       continue                             |
    |            |                                 |
    |       Choose count m                         |
    |        /         \                           |
    |      m=0         m>0                         |
    |       |           |                          |
    |       |           +--> Pointer decoder <-----+
    |       |                      |
    |       |           m removals, then m additions
    |       |                      |
    v       v                      v
  Stop    No-op              Complete exchange
    \       |                      /
     +------+---------------------+
            |
            v
    One environment transition
```

Here $N$ counts points, $D$ is the coordinate dimension, and $d$ is the learned
representation width. Each matrix row represents one point. The detailed
walkthrough begins with the numbers supplied to that row.

## 1. Describe each point

At the current decision, point $i$ has coordinates $x_i$, receiver weight $w_i$,
selection indicator $z_i$, and received signal $r_i$. Its feature vector is

$$
v_i=\left[x_{i,:},w_i,z_i,r_i,r_i-\tau\right]
\in\mathbb R^{D+4}.
$$

Coordinates contribute $D$ numbers; the other four entries are scalars.
The selection indicator is represented numerically as zero or one.
The final entry is the threshold margin: a negative value identifies an
underserved receiver. It is determined by $r_i$ and the known threshold.

Use the [running instance](problem.md#signal-from-the-selected-emitters) with
selection $(1,1,0,0)^\top$. Its point features are:

| Point $i$ | Position $x_i$ | Weight $w_i$ | Selected $z_i$ | Signal $r_i$ | Margin $r_i-\tau$ |
| --- | --- | --- | --- | --- | --- |
| 1 | $0$ | $1$ | $1$ | $1.5$ | $1$ |
| 2 | $0.5$ | $1$ | $1$ | $1.5$ | $1$ |
| 3 | $1$ | $1$ | $0$ | $0.5$ | $0$ |
| 4 | $1.5$ | $1$ | $0$ | $0$ | $-0.5$ |

For example, $v_4=[1.5,1,0,0,-0.5]$ has five entries because this instance
has $D=1$. Stacking all feature vectors produces an $N\times(D+4)$ input matrix.
The environment has already calculated the signal before the policy receives it.

The baseline assumes a fixed, known decay law for a run. These features include
coordinates and current signal, but there is no separate learned input for the
contribution matrix $A$. An independently varying decay law would require a
separate decision about how the model identifies it.

## 2. Embed each point with a shared projection

A learned projection turns each feature vector into a representation of width
$d$:

$$
e_i=\phi_{\mathrm{in}}(v_i)\in\mathbb R^d,
\qquad
\phi_{\mathrm{in}}:\mathbb R^{D+4}\to\mathbb R^d.
$$

The same parameters are used for every point. There is no separate input
network for point 1 or point 2. The projection's internal width and depth are
unspecified by this design.

```text
v_1 [D+4] --> shared projection --> e_1 [d]
v_2 [D+4] --> shared projection --> e_2 [d]
    ...                                ...
v_N [D+4] --> shared projection --> e_N [d]
```

Stack the embeddings as rows of $E\in\mathbb R^{N\times d}$. At this stage,
$e_i$ depends only on the supplied $v_i$. The signal feature already summarizes
a physical effect of the emitters, but the projection itself has not compared
point tokens. The transformer introduces that interaction next.

## 3. Let the point representations interact

The transformer maps the embeddings to contextualized point representations:

$$
H=\operatorname{TransformerEncoder}_\theta(E)
\in\mathbb R^{N\times d}.
$$

The output still has one row per point. Each row can now depend on the other
points. To see how this happens, consider one attention head in the first layer.
From $E$, learned projections form queries, keys, and values:

$$
Q^{\mathrm{attn}}=EW_Q^{\mathrm{attn}},\qquad
K^{\mathrm{attn}}=EW_K^{\mathrm{attn}},\qquad
V^{\mathrm{attn}}=EW_V^{\mathrm{attn}}.
$$

Let the query and key width be $d_a$, and the value width be $d_v$.
The projection matrices have dimensions $d\times d_a$, $d\times d_a$, and
$d\times d_v$, respectively. The superscript distinguishes the attention keys
$K^{\mathrm{attn}}$ from the emitter budget $K$.

For receiver token $i$, compare its query to every token's key:

$$
s_{ij}=\frac{\langle Q^{\mathrm{attn}}_{i,:},K^{\mathrm{attn}}_{j,:}\rangle}
{\sqrt{d_a}},
\qquad
\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{j'=1}^N\exp(s_{ij'})}.
$$

The weights $\alpha_{ij}$ sum to one over $j$. They determine the mixture of
value vectors received by token $i$:

$$
\widetilde e_i=\sum_{j=1}^N\alpha_{ij}V^{\mathrm{attn}}_{j,:}
\in\mathbb R^{d_v}.
$$

```text
Point i's query
       |
Compare with every point's key
       |
Attention weights over the N points
       |
Weighted mixture of their value vectors
```

For our underserved point 4, this mechanism allows its representation to use
information about selected emitters and possible alternatives. Which information
it uses is learned. The attention weights combine information; the policy's
later pointer distributions select points for an exchange.

Each encoder layer combines multiple heads and includes residual connections,
normalization, and a pointwise feed-forward network. The encoder depth
$L_{\mathrm{enc}}$ and head count $h_{\mathrm{enc}}$ remain symbolic parameters.
The complete encoder returns $H$, with representation width $d$.

Attention is full: each point can attend to every point, including itself.
There is no causal mask or positional encoding based on arbitrary point index.
Coordinates carry the spatial information. With shared point operations,
reordering the points reorders their representations correspondingly. Mean
pooling in the next component gives a summary independent of that ordering.

## 4. Summarize the current configuration

Mean pooling averages the contextualized point representations:

$$
\bar h=\frac1N\sum_{i=1}^N H_{i,:}\in\mathbb R^d.
$$

The policy retains $H$ for the pointer decoder. The pooled vector supplies a
compact summary for decisions about the whole action.

The remaining budget also affects the decision. Let $h$ be the number of
decisions remaining, $T$ the episode horizon, and $b\in\{0,1\}$ the feasibility
indicator. A learned projection combines them with the pooled vector:

$$
g=\psi\left([\bar h,h/T,b]\right)\in\mathbb R^d,
\qquad \psi:\mathbb R^{d+2}\to\mathbb R^d.
$$

```text
Mean point representation [d] ---+
Remaining budget fraction h/T --+--> concatenate [d+2] --> psi --> g [d]
Feasibility b ------------------+
```

In our starting example, $b=0$ because receiver 4 is underserved. The fraction
$h/T$ tells the policy how much opportunity remains to change that selection.
We now have the global representation needed for the stop and count heads.

## 5. Decide whether to stop

The stop head produces two logits, one for stop and one for continue:

$$
\lambda^{\mathrm{stop}}=gW_{\mathrm{stop}}+\beta_{\mathrm{stop}}
\in\mathbb R^2,
\qquad W_{\mathrm{stop}}\in\mathbb R^{d\times2}.
$$

A logit is an unnormalized score. Softmax converts the two scores into a
categorical distribution. If $b=0$, replace the stop logit with $-\infty$
first, giving stop probability zero and continue probability one.

Our initial example is infeasible, so it must continue. For a feasible
selection, either outcome can be sampled. A sampled stop emits a zero exchange
and ends action construction; the count and point choices are skipped.
The [environment](environment.md#continuing-without-edits-and-stopping) defines
the resulting episode termination.

## 6. Choose how many pairs to exchange

On the continue branch, a separate head produces a logit for each count
$0,1,\ldots,M$:

$$
\lambda^{\mathrm{count}}=gW_{\mathrm{count}}+\beta_{\mathrm{count}}
\in\mathbb R^{M+1},
\qquad W_{\mathrm{count}}\in\mathbb R^{d\times(M+1)}.
$$

Counts above $m_{\max}=\min(M,K,N-K)$ are masked with $-\infty$.
Softmax over the remaining logits defines

$$
P_\theta(m=j\mid o,\mathrm{continue})
=\frac{\exp(\lambda^{\mathrm{count}}_j)}
{\sum_{q=0}^{m_{\max}}\exp(\lambda^{\mathrm{count}}_q)},
\qquad 0\leq j\leq m_{\max}.
$$

Here $o$ denotes the current observation, from which $g$ was computed.
The sampled $m$ is the number of pairs: remove $m$ selected points and add
$m$ unselected points. It chooses the edit size before its members.

Suppose our example samples $m=1$, with an exchange cap that allows this.
The policy has committed to one removal and one addition. It has not yet chosen
which points participate.

If $m=0$, action construction ends with a continuing no-op. In particular,
$M=0$, $K=0$, or $K=N$ forces every continuing action to have count zero.
The positive-count decoder below is entered only when $m>0$.

## 7. Initialize the decoder's memory

A learned count embedding represents the sampled exchange size:

$$
e_m=\operatorname{Embed}_m(m)\in\mathbb R^d.
$$

Combine it with the global representation to initialize memory:

$$
c_0=\operatorname{MLP}_c([g,e_m])\in\mathbb R^d,
\qquad \operatorname{MLP}_c:\mathbb R^{2d}\to\mathbb R^d.
$$

The memory summarizes the configuration and the planned exchange size before
any point has been selected. It is initialized afresh for every environment
action. There is no decoder memory carried from the preceding action.

For $m=1$, the decoder must make two internal choices. It chooses a removal
first, then an addition. For general $m$, it completes all $m$ removals before
starting the $m$ additions.

## 8. Construct the first removal query

The first query combines the current memory with a learned removal-phase
embedding $e_{\mathrm{remove}}\in\mathbb R^d$ and a scalar describing how many
removal choices remain. Initially all $m$ remain, so the fraction is $m/m=1$:

$$
q_1=\operatorname{MLP}_q([c_0,e_{\mathrm{remove}},1])\in\mathbb R^d,
\qquad \operatorname{MLP}_q:\mathbb R^{2d+1}\to\mathbb R^d.
$$

The query represents what the decoder is looking for at this choice. To compare
it with the points, project the query and each point representation into a
common space of width $d_k$:

$$
\chi_1=q_1W_Q,\qquad \kappa_i=H_{i,:}W_K,
\qquad W_Q,W_K\in\mathbb R^{d\times d_k}.
$$

All neural feature vectors use a row-vector convention in these projections.
The point keys $\kappa_i$ can be calculated once for the complete action.
Assign point $i$ a pointer logit

$$
L_{1,i}=\frac{\langle\chi_1,\kappa_i\rangle}{\sqrt{d_k}}.
$$

This gives $N$ logits, one per original point. The next component restricts
which of those points may be removed.

## 9. Mask invalid points and select a removal

Only currently selected points are eligible for the first removal. Mask the
other logits and sample from the resulting categorical distribution:

$$
\widetilde L_{1,i}=\begin{cases}L_{1,i},&z_i=1,\\
-\infty,&z_i=0,\end{cases}
\qquad
p_{1,i}=\operatorname{softmax}_i(\widetilde L_1),
\qquad u_1\sim\operatorname{Categorical}(p_1).
$$

In the running example, the selected set is $\{1,2\}$. To illustrate the
sampling process, suppose the distribution assigns probabilities $0.3$ and
$0.7$ to those points. Points 3 and 4 have probability zero. Suppose point 2
is sampled, so $u_1=2$.

These probabilities and choices are illustrative, not outputs of a trained
model. We have recorded a planned removal, but the selection is still
$(1,1,0,0)^\top$. The environment waits for the completed exchange.

## 10. Update the memory after the choice

A gated recurrent unit (GRU) updates the decoder memory using the chosen point's
representation and the current phase:

$$
c_1=\operatorname{GRUCell}_\theta([H_{u_1,:},e_{\mathrm{remove}}],c_0)
\in\mathbb R^d.
$$

The GRU input has width $2d$ and its hidden state has width $d$. Its gates learn
how to combine the previous memory with information about the new choice.
The resulting $c_1$ allows the next query to depend on the removal of point 2.

```text
Chosen point representation H[2] [d] --+
                                      +--> GRU input [2d] --+
Removal phase embedding [d] -----------+                    |
                                                           v
Previous memory c_0 [d] ----------------------------------> GRU --> c_1 [d]
```

The point representations $H$ stay fixed throughout this action. The memory
changes with each choice, and the masks record which points remain eligible.
For $m>1$, the next removal would exclude point 2 and use this updated memory.

## 11. Choose additions and generalize the loop

In our $m=1$ example, the removal phase is complete. The next query uses $c_1$
and the addition-phase embedding $e_{\mathrm{add}}$:

$$
q_2=\operatorname{MLP}_q([c_1,e_{\mathrm{add}},1]).
$$

Point scoring uses the same projected keys. The eligible additions are the
points that were unselected at the start of the action: $\{3,4\}$. Removing
point 2 did not make it an eligible addition.

For illustration, suppose addition probabilities are $0.2$ for point 3 and
$0.8$ for point 4, and point 4 is sampled. The GRU updates with the addition
embedding and $H_{4,:}$. Both choices for this action are now complete.

For a general positive count $m$, let $\ell\in\{1,\ldots,2m\}$ index internal
choices. Its phase $\phi_\ell$ and the number $n_\ell$ of choices still needed
in that phase are

$$
\phi_\ell=\begin{cases}
\mathrm{remove},&1\leq\ell\leq m,\\
\mathrm{add},&m<\ell\leq2m,
\end{cases}
\qquad
n_\ell=\begin{cases}
m-\ell+1,&1\leq\ell\leq m,\\
2m-\ell+1,&m<\ell\leq2m.
\end{cases}
$$

Each iteration follows the same equations:

$$
q_\ell=\operatorname{MLP}_q([c_{\ell-1},e_{\phi_\ell},n_\ell/m]),
\qquad
L_{\ell,i}=\frac{\langle q_\ell W_Q,\kappa_i\rangle}{\sqrt{d_k}},
$$

$$
c_\ell=\operatorname{GRUCell}_\theta([H_{u_\ell,:},e_{\phi_\ell}],c_{\ell-1}).
$$

Before sampling $u_\ell$, exclude points already chosen in the same phase.
Writing $z$ for the original selection throughout this loop, the legal sets are

$$
\mathcal I_\ell=\begin{cases}
\{i:z_i=1\}\setminus\{u_1,\ldots,u_{\ell-1}\},&1\leq\ell\leq m,\\
\{i:z_i=0\}\setminus\{u_{m+1},\ldots,u_{\ell-1}\},&m<\ell\leq2m.
\end{cases}
$$

An empty history excludes nothing. Define $\mu_{\ell,i}=0$ when
$i\in\mathcal I_\ell$ and $-\infty$ otherwise. The general point distribution is

$$
P_\theta(u_\ell=i\mid o,\mathrm{continue},m,u_{<\ell})
=\operatorname{softmax}_i(L_\ell+\mu_\ell).
$$

For example, if $m=2$, the remaining-choice fractions are $1,1/2,1,1/2$.
Two distinct removals precede two distinct additions. The count cap ensures
that each phase has enough eligible points to finish.

## 12. Submit one complete exchange

The ordered choices determine removal set $\mathcal R=\{u_1,\ldots,u_m\}$
and addition set $\mathcal C=\{u_{m+1},\ldots,u_{2m}\}$. Convert them to the
[signed exchange](environment.md#one-continuing-action) and submit it with
continue. The environment then applies every edit simultaneously.

Our example gives

```text
Ordered choices: remove 2, add 4
Complete exchange: [0,-1,0,+1]

Environment selection before: [1,1,0,0]
Environment selection after:  [1,0,0,1]
```

The [environment reward example](environment.md#score-and-reward) evaluates
this exact transition. No reward was produced between the removal choice and
the addition choice. At the next active observation, the encoder runs again
and any new decoder starts with fresh memory.

## 13. Assign a probability to the complete choice sequence

Training needs the probability of the choices that produced an environment
action. Call the ordered trace $U$. For our example,
$U=(\mathrm{continue},1,2,4)$ records the branch, pair count, removal, and addition.

Its probability is a product of conditional probabilities:

$$
\begin{aligned}
\pi_\theta(U\mid o)
={}&P_\theta(\mathrm{continue}\mid o)\,
P_\theta(m=1\mid o,\mathrm{continue})\\
&\times P_\theta(u_1=2\mid o,\mathrm{continue},1)\\
&\times P_\theta(u_2=4\mid o,\mathrm{continue},1,u_1=2).
\end{aligned}
$$

The infeasible initial selection makes the continue probability 1. If the
illustrative count probability is $0.6$, the illustrative trace probability
is $1\times0.6\times0.7\times0.8=0.336$. This assumes at least counts 0 and 1
are legal; it specifies no learned parameter values.

For a general continuing trace with $m>0$, the chain rule gives

$$
\pi_\theta(U\mid o)=P_\theta(\mathrm{continue}\mid o)
P_\theta(m\mid o,\mathrm{continue})
\prod_{\ell=1}^{2m}P_\theta(u_\ell\mid o,\mathrm{continue},m,u_{<\ell}).
$$

Taking logarithms turns the product into a sum:

$$
\begin{aligned}
\log\pi_\theta(U\mid o)
={}&\log P_\theta(\mathrm{continue}\mid o)
+\log P_\theta(m\mid o,\mathrm{continue})\\
&+\sum_{\ell=1}^{2m}\log P_\theta(u_\ell\mid o,\mathrm{continue},m,u_{<\ell}).
\end{aligned}
$$

A stop trace has only $\log P_\theta(\mathrm{stop}\mid o)$.
A continuing zero-count trace has only the continue and count-zero terms.
Unused branches and point choices contribute no probability terms.

Different choice orders can produce the same exchange sets. PPO uses the
probability of the sampled ordered trace. Its update must reevaluate that
same order with the same observation and eligibility rules. An unordered
exchange probability would require summing over all traces producing that
exchange; the baseline does not perform that sum.

## 14. Connect the trace to PPO and the value estimate

One environment action has one reward, even when its trace contains several
internal choices. Restore the environment time index $t$: the observation is
$o_t$, the trace is $U_t$, and the complete action receives $R_t$.

The value head uses the same global representation as the action heads:

$$
V_\theta(o_t)=g_tw_V+\beta_V\in\mathbb R,
\qquad w_V\in\mathbb R^d.
$$

It estimates expected remaining return under the policy. The chosen baseline
uses $\gamma=1$, giving the target quantity

$$
V^\pi(o_t)=\mathbb E_\pi\left[
\sum_{j=t}^{t_{\mathrm{end}}-1}R_j\,\middle|\,o_t\right]
=\mathbb E_\pi\left[F(z_{t_{\mathrm{end}}})-F(z_t)\,\middle|\,o_t\right].
$$

The equality follows from the [telescoping return](environment.md#return-over-an-episode).
The value estimate concerns future score change. It is not a certificate of
the best achievable solution.

For the PPO update, retain the sampled trace and its log probability under the
policy that collected it, with parameters $\theta_{\mathrm{old}}$.
Reevaluate the trace with parameters $\theta$ to form the likelihood ratio

$$
\rho_t(\theta)=\frac{\pi_\theta(U_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(U_t\mid o_t)}
=\exp\left(\log\pi_\theta(U_t\mid o_t)
-\log\pi_{\theta_{\mathrm{old}}}(U_t\mid o_t)\right).
$$

This compares how likely the same sequence is under the two policies.
Its advantage estimate belongs to the complete environment action. The full
PPO loss, advantage estimator, entropy treatment, and optimizer settings remain
unspecified; this design defines their trace-likelihood and value interfaces.

## Design rationale and limitations

The encoder lets each point representation incorporate the configuration.
The recurrent decoder lets later choices depend on earlier choices in the
same action. Together they construct an exchange without explicitly scoring
every unordered exchange set, whose count through the cap is

$$
\sum_{m=0}^{m_{\max}}\binom Km\binom{N-K}m.
$$

The policy makes $2m$ point choices for a positive count, and each choice scores
$N$ points. Dense encoder attention still forms $N^2$ pairwise interactions
per head and layer. The environment also stores $N^2$ contributions.
Avoiding exchange enumeration does not establish a wall-clock speedup.

The masks enforce valid membership and distinct choices, with equal removal
and addition counts. They cannot ensure improvement or feasibility. The GRU
compresses the choice history into a fixed-width memory, which may limit its
usefulness for large exchanges. A decoder with explicit history would be a
separate architectural variant.

Final solution quality and generalization require experiments under a stated
inference budget. This proposed architecture has no measured performance claim.

## Notation reference

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $o$ | Active observation of one instance | Fixed instance and current episode quantities |
| $N,D,K,M,T,\tau$ | Problem and episode parameters | Defined in [problem](problem.md) and [environment](environment.md) |
| $h,b$ | Remaining decisions and feasibility indicator | Scalar integer and $\{0,1\}$ |
| $v_i$ | Features of point $i$ | $\mathbb R^{D+4}$ |
| $E,H$ | Embedded and contextualized points | $\mathbb R^{N\times d}$ |
| $d,L_{\mathrm{enc}},h_{\mathrm{enc}}$ | Model width, encoder depth, and head count | Symbolic positive integers |
| $d_a,d_v$ | Query/key and value widths of an illustrative attention head | Symbolic positive integers |
| $\bar h,g$ | Mean representation and global representation | $\mathbb R^d$ |
| $m$ | Sampled pair count | $0,\ldots,m_{\max}$ |
| $\ell,n_\ell$ | Internal choice index and remaining choices in its phase | $1\leq\ell\leq2m$, $1\leq n_\ell\leq m$ |
| $e_m,e_{\phi_\ell}$ | Learned count and phase embeddings | $\mathbb R^d$ |
| $c_\ell,q_\ell$ | Decoder memory and query | $\mathbb R^d$ |
| $\kappa_i,\chi_\ell$ | Projected point key and query | $\mathbb R^{d_k}$ |
| $L_\ell,\mu_\ell$ | Point logits and eligibility mask | $N$ entries |
| $u_\ell$ | Chosen point | Integer in $\{1,\ldots,N\}$ |
| $U,\pi_\theta(U\mid o)$ | Ordered trace and its probability | Branch-dependent sequence and scalar |
| $t$ | Environment decision index | Distinct from internal index $\ell$ |
| $V_\theta,\rho_t$ | Learned value estimate and PPO likelihood ratio | Scalars |

## Architectural precedents

[Pointer Networks](https://arxiv.org/abs/1506.03134) provides the precedent for
selecting input positions with an attention pointer.
[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
provides related attention-based construction of combinatorial solutions.
These are architectural precedents for the proposal, without establishing its
performance on emitter placement.

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
provides the likelihood-ratio training framework. The trace construction and
environment reward above specify how this proposed policy connects to it.
