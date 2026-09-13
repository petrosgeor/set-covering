# Fixed-budget emitter placement

Choose exactly $K$ points as emitters and maximize weighted useful signal. Each
receiver counts signal only up to its demand.

The [environment](environment.md) describes how a search changes its selection;
the [model](model.md) describes the policy that chooses those changes.

## Points and emitter selection

An instance has $N$ points with coordinates $x_i\in\mathbb R^D$. Each point is
also a receiver with finite, nonnegative weight $w_i$ and demand cap $u_i$.
The binary vector

$$
z=(z_1,\ldots,z_N)^\top\in\{0,1\}^N,
\qquad
z_j=\begin{cases}1,&\text{point }j\text{ is an emitter},\\
0,&\text{otherwise}.\end{cases}
$$

records the selection. The fixed budget is

$$
\sum_{j=1}^N z_j=K,\qquad 0\leq K\leq N.
$$

## Signal from one emitter

The contribution matrix uses coordinates exactly as supplied, with a fixed
exponential distance law:

$$
A\in\mathbb R^{N\times N},\qquad
A_{ij}=\exp\!\left(-\lVert x_i-x_j\rVert_2\right).
$$

Rows index receivers and columns index possible emitters. Since a point has
zero distance from itself, $A_{ii}=\exp(0)=1$, so a selected point contributes
one to itself.

## Signal from the selected emitters

Contributions add across selected columns:

$$
r_i=\sum_{j=1}^N A_{ij}z_j,
\qquad
r=Az\in\mathbb R^N.
$$

## Capped utility and unmet demand

The runtime caps each receiver's signal before applying its weight:

$$
F(z)=\sum_{i=1}^N w_i\min(r_i,u_i)
     =\sum_{i=1}^N w_i\min((Az)_i,u_i).
$$

Because signal and demand are nonnegative, weighted unmet demand is

$$
U(z)=\sum_{i=1}^N w_i[u_i-r_i]_+,
\qquad [a]_+=\max(0,a).
$$

For every receiver, capped signal plus unmet demand equals $u_i$, so

$$
F(z)+U(z)=\sum_{i=1}^Nw_i u_i.
$$

The right-hand side is fixed. Maximizing $F$ therefore equals minimizing $U$.

Additional signal above a cap adds no utility. Zero demand is valid and
contributes zero to both quantities. There is no hard coverage requirement, so a selection with
unmet demand remains valid.

## Fixed-budget optimization

The discrete problem is

$$
\begin{aligned}
\max_{z\in\{0,1\}^N}\quad &F(z)=\sum_{i=1}^Nw_i\min((Az)_i,u_i)\\
\text{subject to}\quad &\mathbf 1^\top z=K.
\end{aligned}
$$

Here $\mathbf 1$ is the vector of ones. The equality preserves the emitter
budget. If no cap binds, $F(z)=w^\top Az$.

## Optional variable-budget extension

A separate formulation can vary the emitter count and penalize activation:

$$
\begin{aligned}
\max_{z\in\{0,1\}^N}\quad
& F(z)-\lambda\mathbf 1^\top z.
\end{aligned}
$$

Here $\lambda\geq0$. This extension is outside the fixed-budget environment
and policy. Under $\mathbf 1^\top z=K$, the penalty is the constant $\lambda K$
and cannot change the optimum.

## Notation reference

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $N,D$ | Number of points and spatial dimension | Positive integers |
| $x_i$ | Coordinates of point $i$ | $\mathbb R^D$ |
| $w$ | Receiver importance weights | $\mathbb R_{\geq0}^N$ |
| $u$ | Receiver demand caps | $\mathbb R_{\geq0}^N$ |
| $K$ | Number of selected emitters | Integer in $[0,N]$ |
| $A$ | Contribution matrix from the fixed exponential law | $\mathbb R^{N\times N}$ |
| $z$ | Emitter selection | $\{0,1\}^N$ with $\mathbf 1^\top z=K$ |
| $r$ | Received signal | $Az\in\mathbb R^N$ |
| $F(z)$ | Capped weighted utility | Scalar |
| $U(z)$ | Weighted unmet demand | Scalar |
| $\lambda$ | Activation penalty in the optional extension | Nonnegative scalar |
