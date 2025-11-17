# 从可展开的一项开始：Softmax 到 Performer 线性注意力的推导

本文从一个关键等式出发，解释为什么我们可以把 Softmax 中的核函数写成「q 与 k 已经分离」的形式，从而得到 Performer 的线性注意力近似。

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

我们的目标是找到一个映射 \( \phi \)，使得

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k),
$$

这样就可以先把所有 \( k_j, v_j \) 通过 \( \phi(k_j) \) 做线性累加，最后只和当前的 \( \phi(q) \) 做一次内积，从而实现 **线性时间** 的注意力。

---

## 关键起点：一项可以展开，所以 \( q \) 和 \( k \) 被分离

关键从一个“可展开的一项”开始：对于任意向量 \( q,k \in \mathbb{R}^d \)，有

$$
\|q + k\|^2 = \|q\|^2 + \|k\|^2 + 2 q^\top k.
$$

这一步的用处在于：

- 我们想处理的是 \( \exp(q^\top k) \)；
- 但直接对 \( \exp(q^\top k) \) 做核分解较困难；
- 于是我们把 \( q^\top k \) **“凑”进** \( \tfrac12\|q+k\|^2 \) 里，再借助高斯母函数。

我们要“凑”的目标是：

$$
\exp(q^\top k).
$$

由展开式可得

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

这里你清楚地看到：

> 通过展开 \( \|q+k\|^2 \)，我们把 \( q \) 和 \( k \) 在指数项里彻底分离成三块：只含 q 的、只含 k 的，以及交互项 q^\top k。

这是“这左边的一项是可以展开的，所以 q 和 k 因此分离了”的严格数学形式。

接下来利用高斯随机变量母函数，把左边写成期望，从而反向解出 \( \exp(q^\top k) \)。

---

## 高斯母函数：如何把 \( \|q+k\|^2 \) 变成期望

令随机向量 \( \omega \sim \mathcal{N}(0, I_d) \)。
对任意 \( u \in \mathbb{R}^d \)，经典结论（高斯母函数）告诉我们：

$$
\mathbb{E}_\omega\big[\exp(\omega^\top u)\big]
= \exp\!\left(\frac12\|u\|^2\right).
$$

取 \( u = q + k \)，则

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \exp\!\left(\frac12\|q + k\|^2\right).
$$

代入上一节中展开式：

$$
\exp\!\left(\frac12\|q + k\|^2\right)
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

因此

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \exp\!\left(\frac12\|q\|^2\right)
  \exp\!\left(\frac12\|k\|^2\right)
  \exp(q^\top k).
$$

于是我们可以把 \( \exp(q^\top k) \) **单独解出**：

$$
\exp(q^\top k)
=
\exp\!\left(-\frac12\|q\|^2\right)
\exp\!\left(-\frac12\|k\|^2\right)
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big].
$$

观察期望内部：

$$
\exp(\omega^\top (q + k))
= \exp(\omega^\top q)\,\exp(\omega^\top k),
$$

于是

$$
\mathbb{E}_\omega\big[\exp(\omega^\top (q + k))\big]
= \mathbb{E}_\omega\big[\exp(\omega^\top q)\,\exp(\omega^\top k)\big].
$$

把它代回去得到：

$$
\exp(q^\top k)
=
\mathbb{E}_\omega\Big[
  \exp\!\left(-\tfrac12\|q\|^2\right)\exp(\omega^\top q)\,
  \exp\!\left(-\tfrac12\|k\|^2\right)\exp(\omega^\top k)
\Big].
$$

定义随机特征：

$$
\phi_\omega(x)
= \exp\!\left(-\tfrac12\|x\|^2\right)\exp(\omega^\top x),
$$

得到精确等式：

$$
\exp(q^\top k)
= \mathbb{E}_\omega\big[\phi_\omega(q)\,\phi_\omega(k)\big].
$$

这是**严格等式，不是近似**。

---

## 有限维随机特征：从期望到 \( \phi(q)^\top \phi(k) \)

取 \( r \) 个独立样本：

$$
\omega_1,\dots,\omega_r \sim \mathcal{N}(0, I_d),
$$

定义有限维特征映射：

$$
\phi(x)
= \frac{1}{\sqrt{r}}
\begin{bmatrix}
\phi_{\omega_1}(x)\\
\vdots\\
\phi_{\omega_r}(x)
\end{bmatrix}.
$$

于是

$$
\phi(q)^\top\phi(k)
= \frac{1}{r}\sum_{j=1}^r \phi_{\omega_j}(q)\phi_{\omega_j}(k).
$$

对联合分布取期望：

$$
\mathbb{E}\big[\phi(q)^\top\phi(k)\big]
= \frac1r\sum_{j=1}^r
  \mathbb{E}\big[\phi_{\omega_j}(q)\phi_{\omega_j}(k)\big]
= \mathbb{E}_\omega[\phi_\omega(q)\phi_\omega(k)]
= \exp(q^\top k).
$$

因此：

- \( \phi(q)^\top\phi(k) \) 是 \( \exp(q^\top k) \) 的 **无偏估计**；
- 当 \( r \to \infty \) 时，大数定律保证

$$
\phi(q)^\top\phi(k)
\xrightarrow{\text{a.s.}}
\exp(q^\top k).
$$

---

## 带回注意力公式，实现线性化

原始 softmax 注意力：

$$
N(q) = \sum_{j=1}^n \exp(q^\top k_j)\, v_j,
\qquad
Z(q) = \sum_{j=1}^n \exp(q^\top k_j).
$$

使用 kernel 近似：

$$
\exp(q^\top k_j) \approx \phi(q)^\top \phi(k_j),
$$

得到近似版：

$$
\hat{N}(q) = \sum_{j=1}^n (\phi(q)^\top\phi(k_j))\, v_j,
\qquad
\hat{Z}(q) = \sum_{j=1}^n \phi(q)^\top\phi(k_j).
$$

把 \( \phi(q) \) 抽出：

$$
\hat{N}(q)
= \phi(q)^\top\Big(\sum_{j=1}^n \phi(k_j)v_j^\top\Big),
\qquad
\hat{Z}(q)
= \phi(q)^\top\Big(\sum_{j=1}^n \phi(k_j)\Big).
$$

定义

$$
S = \sum_{j=1}^n \phi(k_j)v_j^\top,\qquad
z = \sum_{j=1}^n \phi(k_j),
$$

最终得到 Performer 线性注意力：

$$
\hat{y}(q)
= \frac{\phi(q)^\top S}{\phi(q)^\top z}.
$$

当 \( r \to \infty \) 时，

$$
\hat{y}(q) \xrightarrow{\text{a.s.}} y(q).
$$


---

## 核心逻辑链：从 Softmax Kernel 到线性注意力

**目标核函数：**
$$K(q,k) = \exp(q^\top k)$$

**高斯随机向量的母函数性质：**
$$\mathbb{E}[\exp(\omega^\top u)] = \exp\left(\frac{1}{2}|u|^2\right)$$
其中 $\omega \sim \mathcal{N}(0, I)$

**关键代换：**
取 $u = q + k$

**最关键的展开（"凑法"核心）：**
$$|q+k|^2 = |q|^2 + |k|^2 + 2q^\top k$$

**代入得到期望表达式：**
$$\exp(q^\top k) = \exp\left(\frac{|q|^2 + |k|^2 + 2q^\top k}{2} - \frac{|q|^2 + |k|^2}{2}\right)$$

$$= \exp\left(-\frac{|q|^2 + |k|^2}{2}\right) \cdot \exp\left(\frac{|q+k|^2}{2}\right)$$

$$= \exp\left(-\frac{|q|^2 + |k|^2}{2}\right) \cdot \mathbb{E}[\exp(\omega^\top(q+k))]$$

$$= \mathbb{E}\left[\exp\left(\omega^\top q - \frac{|q|^2}{2}\right) \cdot \exp\left(\omega^\top k - \frac{|k|^2}{2}\right)\right]$$

**定义随机特征：**
$$\phi_\omega(q) = \exp\left(\omega^\top q - \frac{|q|^2}{2}\right)$$

**得到核函数的随机特征表示：**
$$\exp(q^\top k) = \mathbb{E}_\omega[\phi_\omega(q) \cdot \phi_\omega(k)]$$

**Monte Carlo 近似 + 线性化：**
$$\text{Attention}(q) = \frac{\sum_{j} \exp(q^\top k_j) v_j}{\sum_j \exp(q^\top k_j)} \approx \frac{\phi(q)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q)^\top \sum_j \phi(k_j)} = \frac{\phi(q)^\top S}{\phi(q)^\top z}$$

---

## 为什么展开 $|q+k|^2$ 是关键？

这个展开实现了 **"分离变量"** 的核心目标：

1. **原始形式** $\exp(q^\top k)$：$q$ 和 $k$ 耦合在一起
2. **展开后** $|q|^2 + |k|^2 + 2q^\top k$：交叉项 $2q^\top k$ 被隔离出来
3. **重组后** $\exp(\omega^\top q - |q|^2/2) \cdot \exp(\omega^\top k - |k|^2/2)$：完全解耦成两个独立函数的乘积

这种解耦使得：
- Attention 计算可以 **先累积 $S$ 和 $z$**（与序列长度线性）
- 再用 query $q$ **一次性查询**（常数时间）
- 从而将复杂度从 $O(N^2)$ 降到 $O(N)$

这就是 **Performers 和线性 Attention 的数学基础**！