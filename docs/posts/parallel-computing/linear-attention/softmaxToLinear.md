

# 从传统的 Attention 再到 Linear Attention

---

## 0. 最基础的一点：**我们到底在解什么问题？**

我们有一串随时间到来的三元组：

$$
(k_1, v_1), (k_2, v_2), \dots, (k_t, v_t)
$$

以及当前的查询向量（query）

$$
q_t.
$$

我们的目标是构造一个函数：

$$
y_t = f(q_t;{k_i, v_i}_{i\le t})
$$

满足：

1. **基于内容检索**：和 (q_t) 相似的 (k_i) 对应的 (v_i) 影响更大；
2. **对 (v_i) 线性**；
3. **最好可以在线（online）**，随 (t) 增长不想每次 O(t²) 回扫。

传统 self-attention 是一种实现。

---

## 1. 经典 Softmax Attention 的一般形式

相似度：

$$
s_{ti} = q_t^\top k_i,\quad i \le t.
$$

Softmax 权重：

$$
\alpha_{ti}
= \frac{\exp(q_t^\top k_i)}{\sum_{j\le t} \exp(q_t^\top k_j)}.
$$

输出：

$$
y_t = \sum_{i\le t} \alpha_{ti} v_i.
$$

合在一起写：

$$
\boxed{
y_t =
\frac{
\sum_{i\le t} \exp(q_t^\top k_i)v_i
}{
\sum_{j\le t} \exp(q_t^\top k_j)
}}
\tag{1}
$$

---

## 2. Softmax 的 “分子 / 分母” 分解

分子：

$$
N_t(q_t) = \sum_{i\le t} \exp(q_t^\top k_i)v_i
$$

分母：

$$
D_t(q_t) = \sum_{j\le t} \exp(q_t^\top k_j)
$$

于是：

$$
\boxed{
y_t = \frac{N_t(q_t)}{D_t(q_t)}
}
\tag{2}
$$

这是之后 linear attention 的基础。

---

## 3. 用核技巧：用 (\phi(q)^\top\phi(k)) 近似 (\exp(q^\top k))

假设存在特征映射 (\phi(\cdot))，使得：

$$
\exp(q^\top k)\approx \phi(q)^\top \phi(k).
$$

这是一种核方法（kernel trick）：将非线性的 exp 变成线性的内积形式。

---

## 4. 将 Softmax 的分子 / 分母替换成核近似

### 4.1 分子

$$
N_t(q_t)
\approx \sum_{i\le t} \phi(q_t)^\top \phi(k_i), v_i
= \phi(q_t)^\top
\left(
\sum_{i\le t} \phi(k_i)v_i^\top
\right).
$$

定义分子态：

$$
\boxed{
S_{v,t} := \sum_{i\le t} \phi(k_i)v_i^\top
}
\tag{4}
$$

于是：

$$
\boxed{
N_t(q_t)\approx \phi(q_t)^\top S_{v,t}}
\tag{3}
$$

---

### 4.2 分母

$$
D_t(q_t)
\approx
\phi(q_t)^\top
\left(
\sum_{j\le t} \phi(k_j)
\right).
$$

定义分母态：

$$
\boxed{
S_{k,t} := \sum_{j\le t} \phi(k_j)
}
\tag{6}
$$

于是：

$$
\boxed{
D_t(q_t)\approx \phi(q_t)^\top S_{k,t}}
\tag{5}
$$

---

## 5. 核线性注意力的最终形式

代回 Softmax 的“分子/分母”结构：

$$
\boxed{
y_t\approx
\frac{
\phi(q_t)^\top S_{v,t}
}{
\phi(q_t)^\top S_{k,t}
}
}
\tag{7}
$$

两个状态 (S_{v,t}) 和 (S_{k,t}) 分别对应：

* **分子记忆**（带 value）
* **分母记忆**（不带 value）

---

## 6. 在线（online）更新形式

$$
S_{v,t} = S_{v,t-1} + \phi(k_t)v_t^\top,
$$

$$
S_{k,t} = S_{k,t-1} + \phi(k_t).
$$

每来一个新 token：

1. 更新 (S_{v,t})
2. 更新 (S_{k,t})
3. 输出：

$$
y_t \approx \frac{\phi(q_t)^\top S_{v,t}}{\phi(q_t)^\top S_{k,t}}.
$$

计算复杂度从 **O(t²)** 降到 **O(t)**，并完全在线。

---

## 7. 回到Kimi论文中的最开始的版本：只剩一个 (S_t)

某些文章会写一个非常简化的模型：

状态更新：

$$
S_t = S_{t-1} + k_t v_t^\top
$$

读出：

$$
\hat v_t = S_t^\top k_t
$$

损失：

$$
\mathcal{L}_t(S) = -\langle S_t^\top k_t,\ v_t\rangle.
$$

其本质是一个 **只保留 Softmax 分子项的线性版本**，因为：

1. 假设 (q_t = k_t)
2. 假设 (\phi(x)=x)
3. 不做 Softmax 归一化（没分母）

于是：

$$
S_t^\top k_t
= \sum_{i\le t} (k_i^\top k_t) v_i.
\tag{8}
$$

这相当于权重：

$$
\tilde\alpha_{ti}=k_t^\top k_i
$$

而不是：

$$
\alpha_{ti}\propto \exp(q_t^\top k_i).
$$

这是 **“无归一化线性 attention”**。

---

## 8. 整体逻辑链条总结

1. Softmax Attention 本质是 **分子 / 分母**。

2. 指数内积 (\exp(q^\top k)) 可被核近似。

3. 特征映射 (\phi) 让 softmax 的分子 / 分母都变成 **线性可累加** 的结构。

4. 得到：

   $$
   y_t \approx \frac{\phi(q_t)^\top S_{v,t}}{\phi(q_t)^\top S_{k,t}}.
   $$

5. 在线更新：每步只需做一次外积 + 一次加法，即可维护两个状态。

6. 若进一步简化：

   * 不做归一化，仅有一个 (S_t=\sum k_i v_i^\top)。
   * 对应“只有分子”的近似注意力。

