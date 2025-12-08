
# 📘 **Kimi Delta Attention on Ascend**

## **数学推导 + 算法结构 + 昇腾实现设计文档**

作者：周云龙
日期：2025/12/05

---

# Part 1. 数学推导（完整推导链）

目标：从原始递推式推导到可在昇腾上高效实现的 **UT transform + Chunkwise** 结构，并获得最终输出公式 Eq.(9)。

---

# **1. 原始递推公式**

Kimi / Delta Attention 的核心递推为：

$$
S_t = (I - \beta_t k_t k_t^\top),\mathrm{Diag}(\alpha_t), S_{t-1}
+ \beta_t k_t v_t^\top .
$$

其中：

* $$k_t \in \mathbb{R}^{d_k},\quad v_t\in \mathbb{R}^{d_v}$$
* $$\alpha_t\in (0,1)^{d_k}$$（每通道 decay）
* $$\beta_t\in(0,1)$$（时间门控）
* $$S_t\in\mathbb{R}^{d_k\times d_v}$$ 状态矩阵

定义：

$$
A_t = (I-\beta_t k_t k_t^\top)\mathrm{Diag}(\alpha_t),
\qquad
B_t = \beta_t k_t v_t^\top ,
$$

于是：

$$
S_t = A_t S_{t-1} + B_t .
$$

对一个 chunk（长度 C=64）：

$$
S_{t+C} =
A_{t+C}\cdots A_{t+1} S_t
+
\sum_{i=1}^{C} A_{t+C}\cdots A_{t+i+1} B_{t+i}.
$$

我们需要把整段 $A$ 的乘积和所有 $B$ 的贡献进行一次性压缩计算。

---

# **2. Aₜ 的结构：DPLR（Diagonal + rank-1）**

展开：

$$
A_t
= (I-\beta_t k_tk_t^\top)\mathrm{Diag}(\alpha_t)
= \mathrm{Diag}(\alpha_t)

* \beta_t k_t (k_t^\top \mathrm{Diag}(\alpha_t)).
  $$

定义：

$$
u_t = -\beta_t k_t, \qquad
w_t^\top = k_t^\top\mathrm{Diag}(\alpha_t),
$$

则：

$$
A_t = \mathrm{Diag}(\alpha_t) + u_t w_t^\top .
$$

这个结构非常重要，它保证：

> **任意多个 $A_t$ 的乘积仍然保持 “对角矩阵 + 低秩” 的形式。**

---

# **3. DPLR × DPLR 仍是 DPLR**

两个：

$$
A_2A_1 = D_2D_1 + D_2u_1w_1^\top + u_2 w_2^\top D_1
+ u_2(w_2^\top u_1) w_1^\top .
$$

仍然是：

$$
\text{Diagonal} + \text{rank-≤2}.
$$

r 个连续相乘 => rank r。
但 rank 会随着 chunk 的长度线性增长（如最多 64），不适合直接存储。

这时需要 **WY Representation**。

---

# **4. WY Representation：把 rank-r 写成 V T Vᵀ 结构**

WY 定理（Householder 反射积）：

若：

$$
A_i = I - \beta_i v_i v_i^\top ,
$$

则：

$$
A_r \cdots A_1 = I - V T V^\top ,
$$

其中 V 为列拼接，T 为上三角矩阵。

Kimi 的 $A_t$ 是额外乘上了 $\mathrm{Diag}(\alpha_t)$ 的结构，但因为所有对角矩阵可交换，所以 WY 的低秩结构依然成立。

这导致：

$$
A_{t+r}\cdots A_{t+1}
= \mathrm{Diag}(\gamma^r)

* \sum_{i=1}^r \mathrm{Diag}(\gamma^{i\to r}) k_i w_i^\top ,
  $$

其中：

$$
\gamma^{i\to r} = \prod_{u=i}^r \alpha_u .
$$

---

# **5. 将 WY 递推转换为 UT Transform（方程组形式）**

让我们定义：

$$
\tilde k_r = \gamma^{1\to r} \odot k_r.
$$

有重要恒等式：

$$
k_i^\top \mathrm{Diag}(\gamma^{i\to r}) k_r
===========================================

(\tilde k_i)^\top (\tilde k_r).
$$

于是 WY 的递推可写成：

$$
w_r = \beta_r \left(
\tilde k_r - \sum_{i<r} w_i (\tilde k_i^\top \tilde k_r)
\right),
$$

$$
u_r = \beta_r \left(
v_r - \sum_{i<r} u_i (\tilde k_i^\top \tilde k_r)
\right).
$$

定义矩阵堆叠：

* $$\tilde K\in\mathbb{R}^{C\times d_k}$$
* $$W,U\in\mathbb{R}^{C\times d_*}$$

令：

$$
L = \mathrm{StrictTril}(\mathrm{Diag}(\beta), \tilde K \tilde K^\top) .
$$

可以得到：

$$
(I+L) W = \mathrm{Diag}(\beta)\tilde K ,
$$

$$
(I+L) U = \mathrm{Diag}(\beta)V .
$$

这里 $(I+L)$ 是 **64×64 单位下三角矩阵** ——可以用前向代入求解。

最终：

$$
W = M\tilde K,\qquad U = MV,\qquad
M = (I+L)^{-1}\mathrm{Diag}(\beta).
$$

---

# **6. Chunkwise S 更新（论文 Eq.(8)）**

令 $S$ 为上一个 chunk 的状态，则本 chunk 内的贡献为：

$$
X = W S,
\quad
Y = U - X,
\quad
Z = \tilde K^\top Y,
$$

并记 chunk 减衰为：

$$
\gamma^C = \gamma^{1\to C}.
$$

最终：

$$
S_{\text{next}}
===============

\mathrm{Diag}(\gamma^C) S + Z .
$$

这一步全部 GEMM 操作，非常适合 Ascend Cube。

---

# **7. 输出阶段（论文 Eq.(9））**

对于本 chunk 的所有 Query：

## **(1) inter-chunk**

$$
O_{\mathrm{inter}}
==================

(\Gamma^{1\to C}\odot Q), S ,
$$

即用 decay 后的 Q 乘 S。

## **(2) pseudo-value**

$$
\mathrm{pseudo} = U - W S .
$$

## **(3) intra-chunk**

需要构造：

$$
A_{\mathrm{intra}} =
\mathrm{Tril}!\left[
(\Gamma^{1\to C}\odot Q)(K / \Gamma^{1\to C})^\top
\right] ,
$$

于是：

$$
O_{\mathrm{intra}} = A_{\mathrm{intra}},\mathrm{pseudo}.
$$

---

## **最终输出**

$$
\boxed{
O
=

(\Gamma^{1\to C}\odot Q), S
+
\mathrm{Tril}!\left[
(\Gamma^{1\to C}\odot Q)(K / \Gamma^{1\to C})^\top
\right]
,
(U - W S).
}
$$

这与论文 Eq.(9) 完全一致。

---

# Part 2. Chunkwise Forward 总流程（数学版）

每个 chunk（长度 C=64）：

1. 计算前缀衰减
   $$\Gamma^{1\to C}.$$
2. 计算
   $$\tilde K = \Gamma^{1\to C}\odot K.$$
3. 计算 Gram
   $$G = \tilde K \tilde K^\top.$$
4. 构造
   $$L = \mathrm{StrictTril}(\mathrm{Diag}(\beta) G).$$
5. 解线性系统：
   $$(I+L)W = \mathrm{Diag}(\beta)\tilde K,$$
   $$(I+L)U = \mathrm{Diag}(\beta)V.$$
6. 计算
   $$S_{\text{next}} = \mathrm{Diag}(\gamma^C) S + \tilde K^\top(U - W S).$$
7. 输出：
   $$O = (\Gamma^{1\to C}\odot Q) S

   * A_{\mathrm{intra}} (U - W S).$$

---

# Part 3. 昇腾实现设计（Cube/Vec Kernel Mapping）

下面是上述数学步骤在 Ascend NPU 上的映射。

---

## **Step 1: 计算前缀衰减 Γ（Vec Kernel）**

$$
\gamma^r = \prod_{i=1}^r \alpha_i .
$$

C×d_k 的逐元素 prefix multiply，使用 VecAdd/VecMul 即可。

---

## **Step 2: 计算 $\tilde K = \Gamma\odot K$（Vec Kernel）**

逐元素乘。

---

## **Step 3: Gram 矩阵 $G = \tilde K\tilde K^\top$（Cube Kernel）**

$$
(64\times d_k)(d_k\times 64) = 64\times 64.
$$

这是 CubeMatMul 的最优场景。

---

## **Step 4: 构造 L（Vec Kernel）**

$$
L = \mathrm{StrictTril}(\mathrm{Diag}(\beta) G).
$$

* 逐行乘以 $\beta$
* mask 成 StrictTril

---

## **Step 5: UT transform —— 解下三角线性系统（Vec Kernel）**

解：

$$
(I+L)W = RHS_1,
\qquad
(I+L)U = RHS_2.
$$

forward-substitution：

```
for r in 0..63:
    W[r] = RHS[r]
    for i in 0..r-1:
        W[r] -= L[r,i] * W[i]
```

每行都是 d_k 向量 FMA → 典型 Vec kernel。

---

## **Step 6: S_next 更新（Cube + Vec）**

1. $$X = W S$$ → Cube
2. $$Y = U - X$$ → Vec
3. $$Z = \tilde K^\top Y$$ → Cube
4. $$S_{next} = \gamma^C\odot S + Z$$ → Vec

---

## **Step 7: 输出（Cube + Vec）**

### inter-chunk:

$$
O_{\mathrm{inter}} = Q_\mathrm{decay} S
$$

Cube (C×d_k × d_k×d_v)

### pseudo:

$$
\mathrm{pseudo} = U - WS
$$

### intra-chunk:

1. $$Q K^{-1} = Q_\mathrm{decay}(K/\Gamma)^\top$$（Cube）
2. StrictTril（Vec）
3. $$O_{\mathrm{intra}} = A_{\mathrm{intra}} \mathrm{pseudo}$$（Cube）

### 最终：

$$
O = O_{\mathrm{inter}} + O_{\mathrm{intra}}.
$$

---

# Part 4. 完整 Pipeline（可直接写成算子实现文档）

## **输入：**

* $Q,K,V$
* $S_{\text{init}}$
* $\alpha,\beta$

## **输出：**

* 整个序列的 O（或只要最后一 token）

---

## **For each head:**

```
S = zeros(dk, dv)

for each chunk t:
    # 1 prefix decay
    Gamma = prefix_mul(alpha_chunk)

    # 2 K_tilde
    K_tilde = Gamma * K_chunk

    # 3 Gram
    G = K_tilde @ K_tilde^T

    # 4 L
    L = StrictTril( beta * G )

    # 5 UT
    W = solve_lower_tri( I+L, beta*K_tilde )
    U = solve_lower_tri( I+L, beta*V_chunk )

    # 6 Update S
    X = W @ S
    Y = U - X
    Z = K_tilde^T @ Y
    S = Gamma[-1] * S + Z

    # 7 Output
    Q_decay  = Gamma * Q_chunk
    pseudo   = U - W @ S
    K_invdec = K_chunk / Gamma
    A        = StrictTril( Q_decay @ K_invdec^T )
    O_chunk  = Q_decay @ S + A @ pseudo

return all O_chunk
```

---

# Part 5. 工程优化建议（Ascend）

### **1. 尽可能在 UB 内做 Vec 操作**

UT 的 forward-substitution 完全可以在 UB 里做：

* L（64×64）
* W、U（64×d_k）
* 减少 GM 往返

---

### **2. 融合 Vec kernel**

可融合：

* Gamma prefix + K_tilde
* RHS 生成
* pseudo = U - W@S
* 最终 O = O_inter + O_intra

不可跨越 Cube kernel。

---

### **3. 为多 head 并行做 tiling**

多 head 独立，适合批并行。

---

### **4. chunk 内全部操作固定大小（64）——利于 kernel 静态优化**

例如：

* UT 可 unroll
* Cube 的 tile 可完全固定

---

