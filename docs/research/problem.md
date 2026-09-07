# Fixed-budget emitter placement

Choose exactly $K$ points as emitters and maximize weighted useful signal.
Each receiver counts received signal only up to its demand.

This document defines one optimization instance. The [environment](environment.md)
explains how a search process changes its selection. The [model](model.md)
explains the proposed policy for choosing those changes.

## Points and emitter selection

An instance has $N$ points with coordinates $x_i\in\mathbb R^D$. The integer $D$
is the spatial dimension. Each point has a finite, nonnegative receiver weight
$w_i$ that expresses its importance and a finite, nonnegative $u_i$ that gives
the highest signal counted as useful for that receiver.

We represent the emitter selection by a binary vector:

$$
z=(z_1,\ldots,z_N)^\top\in\{0,1\}^N,
\qquad
z_j=\begin{cases}1,&\text{point }j\text{ is an emitter},\\
0,&\text{otherwise}.\end{cases}
$$

The budget fixes the number of emitters:

$$
\sum_{j=1}^N z_j=K,\qquad 0\leq K\leq N.
$$

For a running example, take four points on a line at $0,0.5,1,1.5$, with unit
weights, $K=2$, and

$$
u=(1,1.25,0.75,1)^\top.
$$

The selection $z=(1,1,0,0)^\top$ activates points 1 and 2. Every point remains
a receiver, and all numerical values in this example are illustrative.

## Signal from one emitter

An emitter's contribution depends on its distance from the receiver. A general
decay function $k:[0,\infty)\to[0,\infty)$ maps that distance to signal
strength. We assume that $k$ is nonincreasing and finite at zero. The general
formulation leaves its particular law, length scale, and normalization
unspecified.

Collect the contributions into a matrix:

$$
A\in\mathbb R^{N\times N},\qquad
A_{ij}=k\!\left(\lVert x_i-x_j\rVert_2\right).
$$

Row $i$ identifies a receiver; column $j$ identifies a possible emitter.
Thus $A_{ij}$ is the contribution to receiver $i$ if point $j$ is selected.
A selected point receives its own contribution $A_{ii}=k(0)$.

For the running example, use $k(s)=\max(0,1-s)$. Adjacent points contribute
$0.5$ to one another, and points at distance at least 1 contribute zero:

$$
A=\begin{bmatrix}
1&0.5&0&0\\
0.5&1&0.5&0\\
0&0.5&1&0.5\\
0&0&0.5&1
\end{bmatrix}.
$$

For example, row 3 says that receiver 3 gets $0.5$ from emitter 2, 1 from
emitter 3, and $0.5$ from emitter 4. Which contributions are present depends
on $z$.

The dependency structure is

```text
points + decay law -> contribution matrix A
contribution matrix A + selection z -> received signal r = A z
received signal r + demands u + weights w -> utility F and unmet demand U
```

## Signal from the selected emitters

Contributions add. Receiver $i$ gets the sum of the columns selected by $z$:

$$
r_i=\sum_{j=1}^N A_{ij}z_j,
\qquad
r=Az\in\mathbb R^N.
$$

With $z=(1,1,0,0)^\top$, only the first two columns contribute. Receiver 3
gets $A_{31}+A_{32}=0+0.5=0.5$, while receiver 4 gets zero.
The complete calculation is:

```text
Selected emitters: points 1 and 2

A[:,1] = [1,   0.5, 0,   0]
A[:,2] = [0.5, 1,   0.5, 0]
          -------------------
r      = [1.5, 1.5, 0.5, 0]
```

The column labels use mathematical indices starting at 1. This same point
numbering is used throughout the research explanation.

## Capped utility and unmet demand

The runtime caps each receiver's signal at its demand before applying its
weight:

$$
F(z)=\sum_{i=1}^N w_i\min(r_i,u_i)
     =\sum_{i=1}^N w_i\min((Az)_i,u_i).
$$

Because the received signal and demands are nonnegative, define weighted unmet
demand as

$$
U(z)=\sum_{i=1}^N w_i[u_i-r_i]_+,
\qquad [a]_+=\max(0,a).
$$

For each receiver, capped signal plus unmet demand equals its demand. Therefore
the weighted sums satisfy

$$
F(z)+U(z)=\sum_{i=1}^Nw_i u_i.
$$

The total on the right is fixed for an instance, so maximizing $F$ is
equivalent to minimizing $U$.

For the running selection, the capped signal is

$$
\min(r,u)=(1,1.25,0.5,0)^\top,
\qquad F(z)=2.75,
\qquad U(z)=1.25.
$$

Signal above a demand cap does not increase $F$. A zero demand is valid and
contributes zero to both quantities. The optimization state has no hard
coverage condition, so a receiver below demand does not invalidate a selection.

## Fixed-budget optimization

The discrete problem is

$$
\begin{aligned}
\max_{z\in\{0,1\}^N}\quad &F(z)=\sum_{i=1}^Nw_i\min((Az)_i,u_i)\\
\text{subject to}\quad &\mathbf 1^\top z=K.
\end{aligned}
$$

Here $\mathbf 1\in\mathbb R^N$ is the vector of ones. The equality constraint
preserves the emitter budget. When a cap binds, extra signal at that receiver
has zero marginal utility. If no cap binds, the objective reduces to
$w^\top Az$.

## Optional variable-budget extension

A separate formulation can allow the emitter count to vary and charge an
activation penalty $\lambda\geq0$:

$$
\begin{aligned}
\max_{z\in\{0,1\}^N}\quad
& F(z)-\lambda\mathbf 1^\top z.
\end{aligned}
$$

The penalty trades capped utility against the number of activated emitters.
This extension is outside the fixed-budget environment and proposed policy.
If the equality $\mathbf 1^\top z=K$ were retained, the penalty would be the
constant $\lambda K$ and would not change the optimal selection.

## Notation reference

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $N,D$ | Number of points and spatial dimension | Positive integers |
| $x_i$ | Coordinates of point $i$ | $\mathbb R^D$ |
| $w$ | Receiver importance weights | $\mathbb R_{\geq0}^N$ |
| $u$ | Receiver demand caps | $\mathbb R_{\geq0}^N$ |
| $K$ | Number of selected emitters | Integer in $[0,N]$ |
| $k$ | Distance-decay function | Nonnegative, nonincreasing, finite at zero |
| $A$ | Contributions, receiver by emitter | $\mathbb R^{N\times N}$ |
| $z$ | Emitter selection | $\{0,1\}^N$ with $\mathbf 1^\top z=K$ |
| $r$ | Received signal | $Az\in\mathbb R^N$ |
| $F(z)$ | Capped weighted utility | Scalar |
| $U(z)$ | Weighted unmet demand | Scalar |
| $\lambda$ | Activation penalty in the optional extension | Nonnegative scalar |

Continue with the [environment](environment.md) to follow an episode from its
initial selection to its final result.
