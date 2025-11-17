# Starting from an Expandable Term: Derivation from Softmax to Performer Linear Attention

This article begins with a key equation to explain why we can write the kernel function in Softmax in a "q and k already separated" form, thereby obtaining Performer's linear attention approximation.

---

## Problem Background and Objective

In standard attention, the weights are

$$
\alpha_j(q) =
\frac{\exp(q^\top k_j)}
{\sum_{\ell=1}^n \exp(q^\top k_\ell)}.
$$

The core component is the kernel function

$$
K(q,k) = \exp(q^\top k).
$$

The output of Softmax attention can be written as

$$
y(q) =
\frac{\sum_{j=1}^n \exp(q^\top k_j)\, v_j}
{\sum_{j=1}^n \exp(q^\top k_j)}.
$$

Our goal is to find a mapping $\phi$ such that

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k),
$$

so that we can first linearly accumulate all $k_j, v_j$ through $\phi(k_j)$, and finally perform only one inner product with the current $\phi(q)$, thereby achieving **linear time** attention.

---

## Key Starting Point: One Term Can Be Expanded, So $q$ and $k$ Are Separated

The key begins with an "expandable term": for any vectors $q,k \in \mathbb{R}^d$, we have

$$
\|q + k\|^2 = \|q\|^2 + \|k\|^2 + 2 q^\top k.
$$

The purpose of this step is:

- We want to process $\exp(q^\top k)$;
- But directly performing kernel decomposition on $\exp(q^\top k)$ is difficult;
- So we **"fit"** $q^\top k$ into $\tfrac12\|q+k\|^2$, then leverage the Gaussian moment generating function.

Our target to "fit" is:

$$
\exp(q^\top k).
$$

From the expansion formula, we get

$$
\frac12\|q + k\|^2
= \frac12\|q\|^2 + \frac12\|k\|^2 + q^\top k.
$$

Therefore

$$
\exp\!\left(\frac12\|q + k\|^2\right)
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

Here you can clearly see:

> By expanding $\|q+k\|^2$, we completely separate $q$ and $k$ in the exponential term into three parts: one containing only q, one containing only k, and the interaction term $q^\top k$.

This is the rigorous mathematical form of "this term on the left can be expanded, so q and k are therefore separated."

Next, we use the moment generating function of Gaussian random variables to write the left side as an expectation, thereby inversely solving for $\exp(q^\top k)$.

---

## Gaussian Moment Generating Function: How to Transform $\|q+k\|^2$ into an Expectation

Let the random vector $\omega \sim \mathcal{N}(0, I_d)$.
For any $u \in \mathbb{R}^d$, a classical result (Gaussian moment generating function) tells us:

$$
\mathbb{E}_\omega\big[\exp(\omega^\top u)\big]
= \exp\!\left(\frac12\|u\|^2\right).
$$

Taking $u = q + k$, we have

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \exp\!\left(\frac12\|q + k\|^2\right).
$$

Substituting the expansion formula from the previous section:

$$
\exp\!\left(\frac12\|q + k\|^2\right)
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

Therefore

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

Thus we can **solve for** $\exp(q^\top k)$ **separately**:

$$
\exp(q^\top k)
=
\exp\!\left(-\frac12\|q\|^2\right)
\exp\!\left(-\frac12\|k\|^2\right)
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big].
$$

Observe inside the expectation:

$$
\exp(\omega^\top (q + k))
= \exp(\omega^\top q)\,\exp(\omega^\top k),
$$

so

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \mathbb{E}_\omega\big[\exp(\omega^\top q)\,\exp(\omega^\top k)\big].
$$

Substituting this back, we get:

$$
\exp(q^\top k)
=
\mathbb{E}_\omega\Big[
  \exp\!\left(-\tfrac12\|q\|^2\right)\exp(\omega^\top q)\,
  \exp\!\left(-\tfrac12\|k\|^2\right)\exp(\omega^\top k)
\Big].
$$

Define the random feature:

$$
\phi_\omega(x)
= \exp\!\left(-\tfrac12\|x\|^2\right)\exp(\omega^\top x),
$$

yielding the exact equation:

$$
\exp(q^\top k)
= \mathbb{E}_\omega\big[\phi_\omega(q)\,\phi_\omega(k)\big].
$$

This is an **exact equation, not an approximation**.

---

## Finite-Dimensional Random Features: From Expectation to $\phi(q)^\top \phi(k)$

Take $r$ independent samples:

$$
\omega_1,\dots,\omega_r \sim \mathcal{N}(0, I_d),
$$

and define the finite-dimensional feature map:

$$
\phi(x)
= \frac{1}{\sqrt{r}}
\begin{bmatrix}
\phi_{\omega_1}(x)\\
\vdots\\
\phi_{\omega_r}(x)
\end{bmatrix}.
$$

Thus

$$
\phi(q)^\top\phi(k)
= \frac{1}{r}\sum_{j=1}^r \phi_{\omega_j}(q)\phi_{\omega_j}(k).
$$

Taking expectation over the joint distribution:

$$
\mathbb{E}\big[\phi(q)^\top\phi(k)\big]
= \frac1r\sum_{j=1}^r
  \mathbb{E}\big[\phi_{\omega_j}(q)\phi_{\omega_j}(k)\big]
= \mathbb{E}_\omega[\phi_\omega(q)\phi_\omega(k)]
= \exp(q^\top k).
$$

Therefore:

- $\phi(q)^\top\phi(k)$ is an **unbiased estimator** of $\exp(q^\top k)$;
- As $r \to \infty$, the law of large numbers guarantees

$$
\phi(q)^\top\phi(k)
\xrightarrow{\text{a.s.}}
\exp(q^\top k).
$$

---

## Bringing It Back to the Attention Formula, Achieving Linearization

Original softmax attention:

$$
N(q) = \sum_{j=1}^n \exp(q^\top k_j)\, v_j,
\qquad
Z(q) = \sum_{j=1}^n \exp(q^\top k_j).
$$

Using the kernel approximation:

$$
\exp(q^\top k_j) \approx \phi(q)^\top \phi(k_j),
$$

we obtain the approximate version:

$$
\hat{N}(q) = \sum_{j=1}^n (\phi(q)^\top\phi(k_j))\, v_j,
\qquad
\hat{Z}(q) = \sum_{j=1}^n \phi(q)^\top\phi(k_j).
$$

Factoring out $\phi(q)$:

$$
\hat{N}(q)
= \phi(q)^\top\Big(\sum_{j=1}^n \phi(k_j)v_j^\top\Big),
\qquad
\hat{Z}(q)
= \phi(q)^\top\Big(\sum_{j=1}^n \phi(k_j)\Big).
$$

Define

$$
S = \sum_{j=1}^n \phi(k_j)v_j^\top,\qquad
z = \sum_{j=1}^n \phi(k_j),
$$

finally obtaining Performer linear attention:

$$
\hat{y}(q)
= \frac{\phi(q)^\top S}{\phi(q)^\top z}.
$$

As $r \to \infty$,

$$
\hat{y}(q) \xrightarrow{\text{a.s.}} y(q).
$$

---

## Core Logic Chain: From Softmax Kernel to Linear Attention

**Target kernel function:**
$$K(q,k) = \exp(q^\top k)$$

**Moment generating function property of Gaussian random vectors:**
$$\mathbb{E}[\exp(\omega^\top u)] = \exp\left(\frac{1}{2}|u|^2\right)$$
where $\omega \sim \mathcal{N}(0, I)$

**Key substitution:**
Take $u = q + k$

**The most crucial expansion (core of the "fitting method"):**
$$|q+k|^2 = |q|^2 + |k|^2 + 2q^\top k$$

**Substituting to obtain the expectation expression:**
$$\exp(q^\top k) = \exp\left(\frac{|q|^2 + |k|^2 + 2q^\top k}{2} - \frac{|q|^2 + |k|^2}{2}\right)$$

$$= \exp\left(-\frac{|q|^2 + |k|^2}{2}\right) \cdot \exp\left(\frac{|q+k|^2}{2}\right)$$

$$= \exp\left(-\frac{|q|^2 + |k|^2}{2}\right) \cdot \mathbb{E}[\exp(\omega^\top(q+k))]$$

$$= \mathbb{E}\left[\exp\left(\omega^\top q - \frac{|q|^2}{2}\right) \cdot \exp\left(\omega^\top k - \frac{|k|^2}{2}\right)\right]$$

**Define random features:**
$$\phi_\omega(q) = \exp\left(\omega^\top q - \frac{|q|^2}{2}\right)$$

**Obtain the random feature representation of the kernel function:**
$$\exp(q^\top k) = \mathbb{E}_\omega[\phi_\omega(q) \cdot \phi_\omega(k)]$$

**Monte Carlo approximation + linearization:**
$$\text{Attention}(q) = \frac{\sum_{j} \exp(q^\top k_j) v_j}{\sum_j \exp(q^\top k_j)} \approx \frac{\phi(q)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q)^\top \sum_j \phi(k_j)} = \frac{\phi(q)^\top S}{\phi(q)^\top z}$$

---

## Why Is Expanding $|q+k|^2$ the Key?

This expansion achieves the core goal of **"variable separation"**:

1. **Original form** $\exp(q^\top k)$: $q$ and $k$ are coupled together
2. **After expansion** $|q|^2 + |k|^2 + 2q^\top k$: the cross term $2q^\top k$ is isolated
3. **After reorganization** $\exp(\omega^\top q - |q|^2/2) \cdot \exp(\omega^\top k - |k|^2/2)$: completely decoupled into a product of two independent functions

This decoupling enables:
- Attention computation can **first accumulate $S$ and $z$** (linear in sequence length)
- Then use query $q$ for a **one-time lookup** (constant time)
- Thereby reducing complexity from $O(N^2)$ to $O(N)$

This is the **mathematical foundation of Performers and linear Attention**!