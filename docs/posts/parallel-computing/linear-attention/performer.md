# 从可展开的一项开始：Softmax 到 Performer 线性注意力的推导

本文从一个关键等式出发，解释为什么我们可以把 Softmax 中的核函数写成「\(q\) 与 \(k\) 已经分离」的形式，从而得到 Performer 的线性注意力近似。

---

## 问题背景与目标

在标准注意力中，权重为

$$
\alpha_j(q) =
\frac{\exp(q^\top k_j)}
{\sum_{\ell=1}^n \exp(q^\top k_\ell)}.
$$

其中最核心的一项是核函数

$$
K(q,k) = \exp(q^\top k).
$$

Softmax 注意力的输出可以写成

$$
y(q) =
\frac{\sum_{j=1}^n \exp(q^\top k_j)\, v_j}
{\sum_{j=1}^n \exp(q^\top k_j)}.
$$

我们的目标是找到一个映射 \(\phi\)，使得

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k),
$$

这样就可以先把所有 \(k_j, v_j\) 通过 \(\phi(k_j)\) 做线性累加，最后只和当前的 \(\phi(q)\) 做一次内积，从而实现**线性时间**的注意力。

---

## 关键起点：一项可以展开，所以 \(q\) 和 \(k\) 被分离

关键从一个“可展开的一项”开始：对任意向量 \(q,k \in \mathbb{R}^d\)，有

$$
\|q + k\|^2 = \|q\|^2 + \|k\|^2 + 2 q^\top k.
$$

这一步的用处在于：

- 我们想处理的是 \(\exp(q^\top k)\)；
- 但直接对 \(\exp(q^\top k)\) 做核分解比较困难；
- 于是我们把 \(q^\top k\)**“凑”进** \(\tfrac12\|q+k\|^2\) 里，再借助高斯的母函数。

先把要“凑”的目标写出来：

$$
\exp(q^\top k).
$$

利用上面的展开式，有

$$
\frac12\|q + k\|^2
= \frac12\|q\|^2 + \frac12\|k\|^2 + q^\top k.
$$

于是

$$
\exp\!\left(\frac12\|q + k\|^2\right)
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

从这里你就可以清楚地看到：

> 我们通过展开 \(\|q+k\|^2\)，把 \(q\) 和 \(k\) 在指数里面分离成了三块：只含 \(q\) 的，只含 \(k\) 的，以及交互项 \(q^\top k\)。

这就是“这左边的一项是可以展开的，所以 \(q\) 和 \(k\) 因此分离了”的严格数学形式。

接下来要做的事是：利用高斯随机变量的母函数，把左边那一项写成**期望**，从而反向解出 \(\exp(q^\top k)\) 的表示。

---

## 高斯母函数：如何把 \(\|q+k\|^2\) 变成期望

令随机向量 \(\omega \sim \mathcal{N}(0, I_d)\)。
对于任意 \(u \in \mathbb{R}^d\)，有经典结论（高斯的母函数）：

$$
\mathbb{E}_\omega\big[\exp(\omega^\top u)\big]
= \exp\!\left(\frac12\|u\|^2\right).
$$

现在取 \(u = q + k\)，则

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \exp\!\left(\frac12\|q + k\|^2\right).
$$

将上一节的展开代入右边：

$$
\exp\!\left(\frac12\|q + k\|^2\right)
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

于是我们得到

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

现在可以**把 \(\exp(q^\top k)\) 单独解出来**：

$$
\exp(q^\top k)
=
\exp\!\left(-\frac12\|q\|^2\right)
\exp\!\left(-\frac12\|k\|^2\right)
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big].
$$

观察期望里的这一项：

$$
\exp(\omega^\top (q + k))
= \exp(\omega^\top q)\,\exp(\omega^\top k),
$$

所以

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \mathbb{E}_\omega\big[\exp(\omega^\top q)\,\exp(\omega^\top k)\big].
$$

把这一步代回去：

$$
\exp(q^\top k)
=
\mathbb{E}_\omega\Big[
  \exp\!\left(-\tfrac12\|q\|^2\right)\exp(\omega^\top q)\,
  \exp\!\left(-\tfrac12\|k\|^2\right)\exp(\omega^\top k)
\Big].
$$

此时定义随机特征函数

$$
\phi_\omega(x)
= \exp\!\left(-\tfrac12\|x\|^2\right)\exp(\omega^\top x),
$$

就得到一个非常干净的等式：

$$
\exp(q^\top k)
= \mathbb{E}_\omega\big[\phi_\omega(q)\,\phi_\omega(k)\big].
$$

**这一步是严格等式，不是近似。**
我们已经把 softmax 的核函数写成了某个随机特征内积的期望。

---

## 有限维随机特征：从期望到 \(\phi(q)^\top\phi(k)\)

上面得到的是一个“无限维”的随机特征（依赖整个分布）。
接下来用 Monte Carlo 把期望近似为有限维向量的内积。

取 \(r\) 个独立样本

$$
\omega_1,\dots,\omega_r \sim \mathcal{N}(0, I_d),
$$

定义有限维特征映射

$$
\phi(x)
= \frac{1}{\sqrt{r}}
\begin{bmatrix}
\phi_{\omega_1}(x) \\
\vdots \\
\phi_{\omega_r}(x)
\end{bmatrix}.
$$

则有

$$
\phi(q)^\top \phi(k)
= \frac{1}{r}\sum_{j=1}^r \phi_{\omega_j}(q)\,\phi_{\omega_j}(k).
$$

对 \(\omega_1,\dots,\omega_r\) 的联合分布取期望，利用独立同分布的性质：

$$
\mathbb{E}\big[\phi(q)^\top \phi(k)\big]
= \frac1r\sum_{j=1}^r
   \mathbb{E}\big[\phi_{\omega_j}(q)\,\phi_{\omega_j}(k)\big]
= \mathbb{E}_\omega\big[\phi_\omega(q)\,\phi_\omega(k)\big]
= \exp(q^\top k).
$$

因此：

- \(\phi(q)^\top\phi(k)\) 是 \(\exp(q^\top k)\) 的**无偏估计器**；
- 当 \(r \to \infty\) 时，根据大数定律，有
  $$
  \phi(q)^\top\phi(k)
  \xrightarrow{\text{a.s.}}
  \exp(q^\top k).
  $$

这就是 Performer 所说的：
**“softmax kernel 的 unbiased, asymptotically exact approximation”**。

---

## 带回注意力公式，实现线性化

原始 softmax 注意力的分子和分母分别为

$$
N(q) = \sum_{j=1}^n \exp(q^\top k_j)\, v_j,
\qquad
Z(q) = \sum_{j=1}^n \exp(q^\top k_j).
$$

现在用核近似替换：

$$
\exp(q^\top k_j) \approx \phi(q)^\top\phi(k_j),
$$

得到近似版本

$$
\hat{N}(q) = \sum_{j=1}^n \big(\phi(q)^\top\phi(k_j)\big)\, v_j,
\qquad
\hat{Z}(q) = \sum_{j=1}^n \phi(q)^\top\phi(k_j).
$$

把 \(\phi(q)\) 抽出来：

$$
\hat{N}(q)
= \phi(q)^\top\Big(\sum_{j=1}^n \phi(k_j) v_j^\top\Big),
\qquad
\hat{Z}(q)
= \phi(q)^\top\Big(\sum_{j=1}^n \phi(k_j)\Big).
$$

记

$$
S = \sum_{j=1}^n \phi(k_j) v_j^\top \in \mathbb{R}^{r\times m},
\qquad
z = \sum_{j=1}^n \phi(k_j) \in \mathbb{R}^r,
$$

最终得到 Performer 的线性注意力输出

$$
\hat{y}(q)
= \frac{\hat{N}(q)}{\hat{Z}(q)}
= \frac{\phi(q)^\top S}{\phi(q)^\top z}.
$$

此时：

- 所有关于序列 \(\{k_j, v_j\}\) 的信息，都被压缩进了 \(S\) 和 \(z\)；
- 对每个新来的 query，只需 \(O(r)\) 的时间即可计算注意力输出；
- 而当 \(r \to \infty\) 时，有
  $$
  \hat{y}(q) \xrightarrow{\text{a.s.}} y(q),
  $$
  其中 \(y(q)\) 是原始 softmax attention 的输出。

---

## 逻辑闭环总结（强调“展开那一项”的作用）

整条链条的逻辑可以压缩为 6 步：

1. **目标核：**
   $$
   K(q,k) = \exp(q^\top k).
   $$

2. **利用高斯母函数：**
   $$
   \mathbb{E}_\omega[\exp(\omega^\top u)] = \exp\!\left(\tfrac12\|u\|^2\right).
   $$

3. **选择 \(u = q + k\)**，得到
   $$
   \mathbb{E}_\omega[\exp(\omega^\top (q + k))]
   = \exp\!\left(\tfrac12\|q + k\|^2\right).
   $$

4. **关键展开：**
   $$
   \|q + k\|^2 = \|q\|^2 + \|k\|^2 + 2 q^\top k,
   $$
   于是
   $$
   \exp\!\left(\tfrac12\|q + k\|^2\right)
   = \exp\!\left(\tfrac12\|q\|^2\right)
     \exp\!\left(\tfrac12\|k\|^2\right)
     \exp(q^\top k).
   $$
   在这一步，\(q\) 和 \(k\) 在指数中被**显式分离**出来。

5. **反向解出核并写成期望形式：**
   $$
   \exp(q^\top k) = \mathbb{E}_\omega[\phi_\omega(q)\,\phi_\omega(k)],
   $$
   再通过有限个样本构造
   $$
   \exp(q^\top k) \approx \phi(q)^\top\phi(k).
   $$

6. **把核近似带回注意力公式，得到线性时间注意力：**
   $$
   \hat{y}(q) = \frac{\phi(q)^\top S}{\phi(q)^\top z},
   $$
   且随 \(r \to \infty\) 收敛到原始 \(y(q)\)。

你刚刚强调的那句话——

> “你要表示这左边的一项是可以展开的，所以 q 和 k 就因此分离了。”

在数学上就是这一步：
从 \(\|q+k\|^2\) 的展开到指数拆分，把 \(\exp(q^\top k)\) 凑出来并单独解出，
这是整个 Performer 线性注意力推导的关键转折点。
