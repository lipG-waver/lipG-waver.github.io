# Kimi Delta Attention (KDA) Chunkwise 并行算法详解

## 1. 引言

Kimi Delta Attention (KDA) 是 Kimi Linear 模型的核心组件，它在 Gated DeltaNet 的基础上引入了**细粒度的 channel-wise gating 机制**。本文档重点讲解如何通过 **chunkwise parallelization** 来高效计算状态矩阵 $\mathbf{S}$ 和输出矩阵 $\mathbf{O}$。

---

## 2. KDA 的递归形式

KDA 的核心递归更新公式为：

$$
\mathbf{S}_t = \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top \right) \text{Diag}(\boldsymbol{\alpha}_t) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top
$$

$$
\mathbf{o}_t = \mathbf{S}_t^\top \mathbf{q}_t
$$

其中：
- $\mathbf{S}_t \in \mathbb{R}^{d_k \times d_v}$：状态矩阵（关联记忆）
- $\mathbf{q}_t, \mathbf{k}_t \in \mathbb{R}^{d_k}$：query 和 key 向量
- $\mathbf{v}_t \in \mathbb{R}^{d_v}$：value 向量
- $\boldsymbol{\alpha}_t \in [0,1]^{d_k}$：**channel-wise decay gate**（细粒度遗忘门）
- $\beta_t \in [0,1]$：学习率标量

### 直观理解

这个更新可以分解为两步：
1. **Decay + Delta Rule**：先对旧状态施加细粒度衰减 $\text{Diag}(\boldsymbol{\alpha}_t)$，再通过 Householder-like 变换 $(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ 进行纠错
2. **Hebbian Update**：加入新的 key-value 关联 $\beta_t \mathbf{k}_t \mathbf{v}_t^\top$

---

## 3. Chunkwise 展开

将长度为 $L$ 的序列划分为 $L/C$ 个 chunk，每个 chunk 长度为 $C$。

### 符号约定

- $\square_{[t]}$：第 $t$ 个 chunk 内的矩阵（堆叠 chunk 内所有向量）
- $\square_{[t]}^r$：第 $t$ 个 chunk 内第 $r$ 个元素
- $\mathbf{S}_{[t]} := \mathbf{S}_{[t]}^0$：chunk 起始状态
- 累积衰减：$\gamma_{[t]}^{i \to j} := \prod_{k=i}^{j} \alpha_{[t]}^k$

### Chunk 内的状态展开

对于 chunk $[t]$ 内的第 $r$ 个位置：

$$
\mathbf{S}_{[t]}^r = \underbrace{\left( \prod_{i=1}^{r} \left(\mathbf{I} - \beta_{[t]}^i \mathbf{k}_{[t]}^i {\mathbf{k}_{[t]}^i}^\top \right) \text{Diag}(\boldsymbol{\alpha}_{[t]}^i) \right)}_{:= \mathbf{P}_{[t]}^r} \cdot \mathbf{S}_{[t]}^0 + \underbrace{\sum_{i=1}^{r} \left( \prod_{j=i+1}^{r} (\cdots) \right) \cdot \beta_{[t]}^i \mathbf{k}_{[t]}^i {\mathbf{v}_{[t]}^i}^\top}_{:= \mathbf{H}_{[t]}^r}
$$

即：
$$
\mathbf{S}_{[t]}^r = \mathbf{P}_{[t]}^r \cdot \mathbf{S}_{[t]}^0 + \mathbf{H}_{[t]}^r
$$

其中：
- $\mathbf{P}_{[t]}^r$：累积转移矩阵
- $\mathbf{H}_{[t]}^r$：chunk 内累积的新信息

---

## 4. WY 表示法：将累积乘积压缩为稠密形式

### 4.1 转移矩阵 $\mathbf{P}$ 的 WY 表示

**核心思想**：一系列 rank-1 更新可以用紧凑的矩阵乘法表示。

$$
\mathbf{P}_{[t]}^r = \text{Diag}(\boldsymbol{\gamma}_{[t]}^r) - \sum_{i=1}^{r} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^i {\mathbf{w}_{[t]}^i}^\top
$$

辅助向量 $\mathbf{w}_{[t]}^r$ 通过递推计算：

$$
\mathbf{w}_{[t]}^r = \beta_{[t]}^r \left( \text{Diag}(\boldsymbol{\gamma}_{[t]}^r) \mathbf{k}_{[t]}^r - \sum_{i=1}^{r-1} \mathbf{w}_{[t]}^i \left( {\mathbf{k}_{[t]}^i}^\top \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r \right) \right)
$$

### 4.2 信息累积项 $\mathbf{H}$ 的表示

$$
\mathbf{H}_{[t]}^r = \sum_{i=1}^{r} \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^i {\mathbf{u}_{[t]}^i}^\top
$$

辅助向量 $\mathbf{u}_{[t]}^r$ 递推：

$$
\mathbf{u}_{[t]}^r = \beta_{[t]}^r \left( \mathbf{v}_{[t]}^r - \sum_{i=1}^{r-1} \mathbf{u}_{[t]}^i \left( {\mathbf{k}_{[t]}^i}^\top \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r \right) \right)
$$

---

## 5. UT 变换：减少非 MatMul FLOPs

为了更好地利用 Tensor Core，引入 UT 变换将递推转化为矩阵运算：

$$
\mathbf{M}_{[t]} = \left( \mathbf{I} + \text{StrictTril}\left( \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]} \right) \left( \frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}} \right)^\top \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})
$$

然后：

$$
\mathbf{W}_{[t]} = \mathbf{M}_{[t]} \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]} \right), \quad \mathbf{U}_{[t]} = \mathbf{M}_{[t]} \mathbf{V}_{[t]}
$$

**关键点**：下三角矩阵的逆可以通过前向替换（Gaussian elimination）高效计算。

---

## 6. Chunk 间状态更新

Chunk 结束时的状态更新（用于下一个 chunk）：

$$
\mathbf{S}_{[t+1]} = \text{Diag}(\boldsymbol{\gamma}_{[t]}^C) \mathbf{S}_{[t]} + \left( \boldsymbol{\Gamma}_{[t]}^{i \to C} \odot \mathbf{K}_{[t]} \right)^\top \left( \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]} \right)
$$

这里的结构清晰：
- **衰减旧状态**：$\text{Diag}(\boldsymbol{\gamma}_{[t]}^C) \mathbf{S}_{[t]}$
- **加入新信息**：来自当前 chunk 的累积

---

## 7. 输出计算：Inter-chunk 递归 + Intra-chunk 并行

最终输出的计算采用**混合策略**：

$$
\mathbf{O}_{[t]} = \underbrace{\left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{Q}_{[t]} \right) \mathbf{S}_{[t]}}_{\text{inter-chunk}} + \underbrace{\text{Tril}\left( \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{Q}_{[t]} \right) \left( \frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}} \right)^\top \right)}_{\text{intra-chunk}} \underbrace{\left( \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]} \right)}_{\text{"pseudo"-value}}
$$

### 计算策略

1. **Inter-chunk**（跨 chunk）：查询历史状态，**递归**进行
2. **Intra-chunk**（chunk 内）：计算 chunk 内的注意力，**并行**进行

### 示意图理解

```
输出 O = [历史信息贡献] + [当前chunk内信息贡献]
         └── 递归累积的S ──┘   └── 可并行的矩阵乘法 ──┘
```

---

## 8. 与通用 DPLR 的效率对比

### 通用 DPLR 形式

$$
\mathbf{S}_t = (\mathbf{D} - \mathbf{a}_t \mathbf{b}_t^\top) \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top
$$

### KDA 的约束形式

$$
\mathbf{S}_t = \left( \text{Diag}(\boldsymbol{\alpha}_t) - \beta_t \mathbf{k}_t (\mathbf{k}_t \odot \boldsymbol{\alpha}_t)^\top \right) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top
$$

即：$\mathbf{D} = \text{Diag}(\boldsymbol{\alpha}_t)$，$\mathbf{a}_t = \beta_t \mathbf{k}_t$，$\mathbf{b}_t = \mathbf{k}_t \odot \boldsymbol{\alpha}_t$

### KDA 的效率优势

| 对比项 | 通用 DPLR | KDA |
|-------|----------|-----|
| 二级 chunk 矩阵计算次数 | 4 次 | 2 次 |
| 额外矩阵乘法 | 3 次 | 0 次 |
| 数值稳定性 | 需要 log 域计算 | 直接计算 |
| 实测速度 | 基准 | **约 2× 加速** |

**核心洞察**：通过将 $\mathbf{a}$ 和 $\mathbf{b}$ 都绑定到 $\mathbf{k}$，KDA 在保持表达能力的同时大幅减少了计算开销。

---

## 9. 伪代码总结

```python
def chunk_kda(Q, K, V, g, beta, chunk_size=64):
    # 1. 分 chunk
    Q, K, V, g, beta = reshape_to_chunks(...)
    
    # 2. 计算累积衰减
    gc = g.cumsum(dim=-2)
    
    # 3. 构建并求逆 M 矩阵（UT 变换）
    A = compute_kk_attention_matrix(K, gc, beta)
    A_inv = forward_substitution(I + StrictTril(A))
    M = A_inv @ Diag(beta)
    
    # 4. 计算辅助矩阵
    W = M @ (gc.exp() * K)  # 用于状态更新
    U = M @ V               # "pseudo"-value
    
    # 5. 逐 chunk 递归 + 并行输出
    S = initial_state
    for chunk_idx in range(num_chunks):
        # Intra-chunk attention (并行)
        A_qk = compute_qk_attention(Q[chunk_idx], K[chunk_idx], gc[chunk_idx])
        
        # 输出 = inter-chunk + intra-chunk
        O[chunk_idx] = (Q[chunk_idx] * gc.exp()) @ S + A_qk @ (U - W @ S)
        
        # 更新状态 (递归)
        decay = gc[chunk_idx, -1].exp()
        S = S * decay + (K[chunk_idx] * decay_within_chunk).T @ (U - W @ S)
    
    return O
```

---

## 10. 关键要点总结

1. **Chunkwise 并行化**：将序列分成 chunks，chunk 间递归传递状态，chunk 内并行计算

2. **WY 表示法**：将累积的 Householder 变换压缩成紧凑的矩阵形式，支持高效的矩阵乘法

3. **UT 变换**：将递推关系转化为矩阵求逆问题，通过前向替换高效求解

4. **输出的双重结构**：
   - Inter-chunk：历史信息通过递归状态 $\mathbf{S}$ 传递
   - Intra-chunk：当前 chunk 内的注意力可完全并行

5. **效率优势**：相比通用 DPLR，KDA 的约束形式减少了计算量，实现约 2× 加速

6. **硬件友好**：充分利用 Tensor Core 的矩阵乘法能力，最大化 GPU 利用率

---

## 参考

- Kimi Linear Technical Report (arXiv:2510.26692v2)
- Gated DeltaNet [Yang et al., 2025]
- WY Representation [Bischof & Van Loan, 1987]
- UT Transform [Joffrain et al., 2006]