# Fixed-budget emitter placement

Given a collection of points, select a fixed number as emitters to maximize the
weighted total received signal while ensuring that every point receives at least
a minimum threshold.

Every point is a receiver, and selected points also act as emitters. Each emitter
contributes signal according to its distance from the receiver. Contributions
from multiple emitters add together.

## Mathematical model and notation

An instance consists of the following inputs:

- $N$ points at positions $x_1,\dots,x_N\in\mathbb{R}^{d}$, with Euclidean distance
  $\lVert x_i-x_j\rVert_2$.
- An integer emitter budget $K$, with $0\leq K\leq N$.
- Receiver importance weights $w_1,\dots,w_N\geq0$. Equal importance corresponds
  to $w_i=1$ for every receiver.
- A common minimum received-signal threshold $\tau\geq0$.
- A nonnegative, nonincreasing decay function
  $k:[0,\infty)\to[0,\infty)$, with finite $k(0)$. The formulation leaves the
  particular decay law, length scale, and normalization unspecified.

The decision variable $z_j\in\{0,1\}$ is one if point $j$ is selected as an
emitter and zero otherwise. Let $z=(z_1,\dots,z_N)^\top$ and
$w=(w_1,\dots,w_N)^\top$.

Define the contribution matrix $A\in\mathbb{R}^{N\times N}$ by

$$
A_{ij}=k\!\left(\lVert x_i-x_j\rVert_2\right).
$$

Row $i$ corresponds to receiver $i$, and column $j$ corresponds to potential
emitter $j$. Thus, $A_{ij}$ is the signal that receiver $i$ receives when
emitter $j$ is selected.

The total received signal at point $i$ is

$$
r_i=\sum_{j=1}^{N}A_{ij}z_j.
$$

Writing $r=(r_1,\dots,r_N)^\top\in\mathbb{R}^{N}$ gives $r=Az$.
For example, if only distinct points $p$ and $q$ are selected as emitters, then
receiver $i$ receives $r_i=A_{ip}+A_{iq}$.

## Fixed-budget optimization

Let $\mathbf{1}\in\mathbb{R}^{N}$ be the vector of ones. Choose exactly $K$
emitters by solving

$$
\begin{aligned}
\max_{z\in\{0,1\}^{N}}\quad & w^\top Az\\
\text{subject to}\quad & Az\geq\tau\mathbf{1},\\
& \mathbf{1}^\top z=K,
\end{aligned}
$$

The scalar objective $w^\top Az=\sum_{i=1}^{N}w_i r_i$ is the weighted total
received signal. The inequality is componentwise: every receiver must satisfy
$r_i\geq\tau$. The equality $\mathbf{1}^\top z=K$ fixes the number of selected
emitters.

This model has the following consequences:

- A selected point receives its own emitter's contribution $A_{ii}=k(0)$, in
  addition to contributions from other selected emitters.
- Receiver weights affect the objective only. Even a receiver with $w_i=0$
  must satisfy the minimum-signal requirement.
- Signal above the threshold continues to contribute to the objective without
  saturation. The objective rewards total signal rather than equalizing signal
  across receivers.
- An instance is infeasible if no selection of exactly $K$ emitters satisfies
  all receiver thresholds.

Once $A$ is fixed, this is a binary linear optimization problem: the objective
and constraints are linear in $z$. Defining the emitter score vector
$c=A^\top w$ gives

$$
c_j=\sum_{i=1}^{N}w_iA_{ij},
\qquad
w^\top Az=c^\top z.
$$

Each emitter therefore has a fixed total weighted contribution. Without the
minimum-signal constraints, selecting the $K$ largest scores suffices. With
those constraints, a feasible selection may require lower-scoring emitters to
supply underserved receivers.

## Optional variable-budget extension

An alternative formulation removes the fixed-budget equality and penalizes
the number of emitters. For an activation penalty $\lambda\geq0$, it becomes

$$
\begin{aligned}
\max_{z\in\{0,1\}^{N}}\quad
& w^\top Az-\lambda\mathbf{1}^\top z\\
\text{subject to}\quad & Az\geq\tau\mathbf{1}.
\end{aligned}
$$

The parameter $\lambda$ controls the trade-off between weighted received signal
and emitter count, while every receiver must still meet the threshold. This
extension is outside the initial fixed-budget formulation.

If exactly $K$ emitters were still required, the penalty would be the constant
$\lambda K$ and would not change which selections are optimal.
