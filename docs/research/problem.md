# Fixed-budget emitter placement

Choose exactly $K$ points as emitters so that every receiver gets enough signal,
then maximize the weighted total signal. Every point is a receiver, and a
selected point also contributes signal as an emitter.

This document defines one optimization instance. The [environment](environment.md)
explains how a search process changes its selection. The [model](model.md)
explains the proposed policy for choosing those changes.

## Points and emitter selection

An instance has $N$ points with coordinates $x_i\in\mathbb R^D$. The integer $D$
is the spatial dimension. Each point has a nonnegative receiver weight $w_i$,
which expresses its importance in the objective.

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
weights and $K=2$. The selection $z=(1,1,0,0)^\top$ activates points 1 and 2.
Points 3 and 4 remain receivers. All numerical values in this example are
illustrative; they do not select experimental settings.

## Signal from one emitter

An emitter's contribution depends on its distance from the receiver. A decay
function $k:[0,\infty)\to[0,\infty)$ maps that distance to signal strength.
We assume $k$ is nonincreasing and finite at zero. The general formulation leaves
its particular law, length scale, and normalization unspecified.

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

## Feasibility and objective

Every receiver must reach a common threshold $\tau\geq0$. A selection is
feasible when $r_i\geq\tau$ for every $i$. With the illustrative threshold
$\tau=0.5$, our current selection fails because receiver 4 gets zero.

Among feasible selections, the objective is the weighted total signal. Write
$w=(w_1,\ldots,w_N)^\top$ and define

$$
f(z)=w^\top Az=\sum_{i=1}^N w_i r_i.
$$

The complete optimization problem is

$$
\begin{aligned}
\max_{z\in\{0,1\}^N}\quad & f(z)=w^\top Az\\
\text{subject to}\quad & Az\geq\tau\mathbf 1,\\
&\mathbf 1^\top z=K.
\end{aligned}
$$

Here $\mathbf 1\in\mathbb R^N$ is the vector of ones, and the signal inequality
is componentwise. Receiver weights affect the objective only: even a receiver
with zero weight must meet the threshold. Signal above the threshold continues
to increase the objective without saturation.

In our example, the infeasible selection has objective $3.5$. Replacing emitter
2 with emitter 4 gives $z'=(1,0,0,1)^\top$, signal $(1,0.5,0.5,1)^\top$, and
objective $3$. This selection is feasible even though its total signal is lower.
The feasibility constraints determine which objective values can compete.

An instance is infeasible if no selection of exactly $K$ emitters meets every
threshold. A search that fails to find such a selection has not established
that this is the case.

## Fixed emitter scores and coverage constraints

Once $A$ is fixed, the objective and constraints are linear in the binary
variables. Define the fixed emitter score vector

$$
c=A^\top w,\qquad c_j=\sum_{i=1}^N w_iA_{ij}.
$$

Then $f(z)=c^\top z$. Without the minimum-signal constraints, the optimum selects
the $K$ largest entries of $c$. With those constraints, the chosen columns must
also jointly supply every receiver.

For our unit-weight example, $c=(1.5,2,2,1.5)^\top$. Choosing points 2 and 3
gives signal $(0.5,1.5,1.5,0.5)^\top$ and objective 4. It meets the threshold
and attains the unconstrained top-$K$ value, so it is also optimal for this
illustrative constrained instance.

## Optional variable-budget extension

A separate formulation can allow the emitter count to vary and charge an
activation penalty $\lambda\geq0$:

$$
\begin{aligned}
\max_{z\in\{0,1\}^N}\quad
& w^\top Az-\lambda\mathbf 1^\top z\\
\text{subject to}\quad & Az\geq\tau\mathbf 1.
\end{aligned}
$$

The penalty trades total signal against the number of activated emitters.
This extension is outside the fixed-budget environment and proposed policy.
If the equality $\mathbf 1^\top z=K$ were retained, the penalty would be the
constant $\lambda K$ and would not change the optimal selection.

## Notation reference

| Symbol | Meaning | Dimension or range |
| --- | --- | --- |
| $N,D$ | Number of points and spatial dimension | Positive integers |
| $x_i$ | Coordinates of point $i$ | $\mathbb R^D$ |
| $w$ | Receiver importance weights | $\mathbb R_{\geq0}^N$ |
| $K$ | Number of selected emitters | Integer in $[0,N]$ |
| $\tau$ | Minimum signal at every receiver | Nonnegative scalar |
| $k$ | Distance-decay function | Nonnegative, nonincreasing, finite at zero |
| $A$ | Contributions, receiver by emitter | $\mathbb R^{N\times N}$ |
| $z$ | Emitter selection | $\{0,1\}^N$ with $\mathbf 1^\top z=K$ |
| $r$ | Received signal | $Az\in\mathbb R^N$ |
| $f(z)$ | Weighted total signal | Scalar |
| $c$ | Fixed weighted emitter scores | $A^\top w\in\mathbb R^N$ |
| $\lambda$ | Activation penalty in the optional extension | Nonnegative scalar |

Continue with the [environment](environment.md) to follow an episode from its
initial selection to its final result.
