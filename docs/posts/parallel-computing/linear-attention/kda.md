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
- $\square_{[t]}^r$：第 $t$ 个 chunk 内第 $r$ 个元素（$r \in \{1, 2, \ldots, C\}$）
- $\mathbf{S}_{[t]} := \mathbf{S}_{[t]}^0$：chunk 起始状态

### 累积衰减符号详解

为了表示不同范围的累积衰减，我们定义以下符号：

**基本定义**：位置 $i$ 到位置 $j$ 的累积衰减（向量形式）
$$
\boldsymbol{\gamma}_{[t]}^{i \to j} := \prod_{k=i}^{j} \boldsymbol{\alpha}_{[t]}^k \in \mathbb{R}^{d_k}
$$

**简写约定**：从 chunk 起始到位置 $r$ 的累积衰减
$$
\boldsymbol{\gamma}_{[t]}^r := \boldsymbol{\gamma}_{[t]}^{1 \to r} = \prod_{k=1}^{r} \boldsymbol{\alpha}_{[t]}^k
$$

**矩阵形式**：将 chunk 内所有位置的累积衰减堆叠成矩阵
$$
\boldsymbol{\Gamma}_{[t]}^{1 \to C} := \begin{bmatrix} (\boldsymbol{\gamma}_{[t]}^1)^\top \\ (\boldsymbol{\gamma}_{[t]}^2)^\top \\ \vdots \\ (\boldsymbol{\gamma}_{[t]}^C)^\top \end{bmatrix} \in \mathbb{R}^{C \times d_k}
$$

类似地，$\boldsymbol{\Gamma}_{[t]}^{i \to C}$ 表示从位置 $i$ 到 chunk 末尾的累积衰减矩阵：
$$
\boldsymbol{\Gamma}_{[t]}^{i \to C} := \begin{bmatrix} (\boldsymbol{\gamma}_{[t]}^{1 \to C})^\top \\ (\boldsymbol{\gamma}_{[t]}^{2 \to C})^\top \\ \vdots \\ (\boldsymbol{\gamma}_{[t]}^{C \to C})^\top \end{bmatrix} \in \mathbb{R}^{C \times d_k}
$$

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

### 5.1 从递推到矩阵形式的推导

观察 4.2 节中 $\mathbf{u}$ 的递推公式，将其写成矩阵形式。定义：
- $\mathbf{U}_{[t]} \in \mathbb{R}^{C \times d_v}$：所有 $\mathbf{u}_{[t]}^r$ 按行堆叠
- $\mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}$：所有 $\mathbf{v}_{[t]}^r$ 按行堆叠

递推关系可以改写为：
$$
\mathbf{u}_{[t]}^r = \beta_{[t]}^r \mathbf{v}_{[t]}^r - \beta_{[t]}^r \sum_{i=1}^{r-1} \mathbf{u}_{[t]}^i \underbrace{\left( {\mathbf{k}_{[t]}^i}^\top \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r \right)}_{=: A_{ri}}
$$

其中 $A_{ri}$ 是一个标量，表示位置 $r$ 对位置 $i$ 的"注意力权重"。将所有位置的递推写成矩阵形式：
$$
\mathbf{U}_{[t]} = \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]} - \text{StrictTril}(\mathbf{A}_{[t]}) \mathbf{U}_{[t]}
$$

其中 $\text{StrictTril}(\mathbf{A})$ 表示严格下三角部分（对角线为0）。

整理得：
$$
\left( \mathbf{I} + \text{StrictTril}(\mathbf{A}_{[t]}) \right) \mathbf{U}_{[t]} = \text{Diag}(\boldsymbol{\beta}_{[t]}) \mathbf{V}_{[t]}
$$

因此：
$$
\mathbf{U}_{[t]} = \underbrace{\left( \mathbf{I} + \text{StrictTril}(\mathbf{A}_{[t]}) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})}_{=: \mathbf{M}_{[t]}} \mathbf{V}_{[t]}
$$

### 5.2 注意力矩阵 $\mathbf{A}$ 的具体形式

矩阵 $\mathbf{A}_{[t]}$ 的元素为：
$$
A_{ri} = \beta_{[t]}^r \cdot {\mathbf{k}_{[t]}^i}^\top \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r
$$

这可以用矩阵运算表示。注意到：
$$
{\mathbf{k}_{[t]}^i}^\top \text{Diag}(\boldsymbol{\gamma}_{[t]}^{i \to r}) \mathbf{k}_{[t]}^r = \left( \mathbf{k}_{[t]}^i \odot \boldsymbol{\gamma}_{[t]}^{i} \right)^\top \left( \frac{\mathbf{k}_{[t]}^r}{\boldsymbol{\gamma}_{[t]}^{r}} \right) \cdot \boldsymbol{\gamma}_{[t]}^{r}
$$

利用 $\boldsymbol{\gamma}_{[t]}^{i \to r} = \boldsymbol{\gamma}_{[t]}^r / \boldsymbol{\gamma}_{[t]}^i$（逐元素除法），整个矩阵可以写成：
$$
\mathbf{A}_{[t]} = \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]} \right) \left( \frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}} \right)^\top
$$

### 5.3 最终的 UT 变换公式

综合以上推导：

$$
\mathbf{M}_{[t]} = \left( \mathbf{I} + \text{StrictTril}\left( \text{Diag}(\boldsymbol{\beta}_{[t]}) \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]} \right) \left( \frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}} \right)^\top \right) \right)^{-1} \text{Diag}(\boldsymbol{\beta}_{[t]})
$$

然后：

$$
\mathbf{W}_{[t]} = \mathbf{M}_{[t]} \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{K}_{[t]} \right), \quad \mathbf{U}_{[t]} = \mathbf{M}_{[t]} \mathbf{V}_{[t]}
$$

**关键点**：下三角矩阵的逆可以通过前向替换（forward substitution / Gaussian elimination）高效计算，复杂度为 $O(C^2)$ 而非一般矩阵求逆的 $O(C^3)$。

---

## 6. 数值稳定性考量

### 6.1 累积衰减的数值问题

在实际实现中，累积衰减 $\boldsymbol{\gamma}_{[t]}^r = \prod_{k=1}^{r} \boldsymbol{\alpha}_{[t]}^k$ 可能面临数值问题：

**问题1：下溢（Underflow）**
- 当 $\alpha < 1$ 且 chunk 较长时，累积乘积可能趋近于 0
- 例如：$\alpha = 0.99$，$C = 64$ 时，$0.99^{64} \approx 0.52$（尚可）
- 但如果 $\alpha = 0.95$，$C = 64$ 时，$0.95^{64} \approx 0.038$（接近下溢）

**问题2：除法不稳定**
- 公式中出现 $\mathbf{K}_{[t]} / \boldsymbol{\Gamma}_{[t]}^{1 \to C}$，当 $\boldsymbol{\gamma}$ 很小时除法会放大误差

### 6.2 Log 域计算方案

为了解决上述问题，实际实现中通常在 log 域进行计算：

**Step 1**：计算 log 累积衰减
$$
\log \boldsymbol{\gamma}_{[t]}^r = \sum_{k=1}^{r} \log \boldsymbol{\alpha}_{[t]}^k
$$
这可以通过 cumsum 高效计算，且数值稳定。

**Step 2**：在需要实际值时使用 exp
$$
\boldsymbol{\gamma}_{[t]}^r = \exp\left( \log \boldsymbol{\gamma}_{[t]}^r \right)
$$

**Step 3**：对于比值 $\boldsymbol{\gamma}_{[t]}^{i \to r} = \boldsymbol{\gamma}_{[t]}^r / \boldsymbol{\gamma}_{[t]}^i$，在 log 域计算：
$$
\log \boldsymbol{\gamma}_{[t]}^{i \to r} = \log \boldsymbol{\gamma}_{[t]}^r - \log \boldsymbol{\gamma}_{[t]}^i
$$
然后 $\boldsymbol{\gamma}_{[t]}^{i \to r} = \exp(\log \boldsymbol{\gamma}_{[t]}^r - \log \boldsymbol{\gamma}_{[t]}^i)$

### 6.3 KDA 相比通用 DPLR 的稳定性优势

虽然 KDA 也需要处理累积衰减，但相比通用 DPLR：

1. **衰减结构简单**：KDA 的衰减是对角矩阵 $\text{Diag}(\boldsymbol{\alpha})$，只需维护 $d_k$ 个独立的累积值
2. **无需复杂的矩阵 log**：通用 DPLR 可能需要对非对角矩阵做 log，这在数值上更复杂
3. **约束减少了病态情况**：$\mathbf{a} = \beta \mathbf{k}$，$\mathbf{b} = \mathbf{k} \odot \boldsymbol{\alpha}$ 的绑定避免了 $\mathbf{a}$ 和 $\mathbf{b}$ 任意组合可能导致的数值问题

### 6.4 实现建议

```python
# 推荐：在 log 域计算累积衰减
log_alpha = torch.log(alpha + 1e-6)  # 加小常数防止 log(0)
log_gamma_cumsum = log_alpha.cumsum(dim=-2)

# 需要实际值时
gamma = torch.exp(log_gamma_cumsum)

# 计算比值时（用于注意力矩阵）
# gamma[i->r] = exp(log_gamma[r] - log_gamma[i])
log_gamma_diff = log_gamma_cumsum.unsqueeze(-2) - log_gamma_cumsum.unsqueeze(-3)
gamma_ratio = torch.exp(log_gamma_diff)
```

---

## 7. Chunk 间状态更新

Chunk 结束时的状态更新（用于下一个 chunk）：

$$
\mathbf{S}_{[t+1]} = \text{Diag}(\boldsymbol{\gamma}_{[t]}^C) \mathbf{S}_{[t]} + \left( \boldsymbol{\Gamma}_{[t]}^{i \to C} \odot \mathbf{K}_{[t]} \right)^\top \left( \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]} \right)
$$

这里 $\boldsymbol{\Gamma}_{[t]}^{i \to C}$ 是从各位置 $i$ 到 chunk 末尾 $C$ 的累积衰减矩阵（见第3节符号定义）。

结构清晰：
- **衰减旧状态**：$\text{Diag}(\boldsymbol{\gamma}_{[t]}^C) \mathbf{S}_{[t]}$
- **加入新信息**：来自当前 chunk 的累积

---

## 8. 输出计算：Inter-chunk 递归 + Intra-chunk 并行

最终输出的计算采用**混合策略**：

$$
\mathbf{O}_{[t]} = \underbrace{\left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{Q}_{[t]} \right) \mathbf{S}_{[t]}}_{\text{inter-chunk}} + \underbrace{\text{Tril}\left( \left( \boldsymbol{\Gamma}_{[t]}^{1 \to C} \odot \mathbf{Q}_{[t]} \right) \left( \frac{\mathbf{K}_{[t]}}{\boldsymbol{\Gamma}_{[t]}^{1 \to C}} \right)^\top \right)}_{\text{intra-chunk attention}} \underbrace{\left( \mathbf{U}_{[t]} - \mathbf{W}_{[t]} \mathbf{S}_{[t]} \right)}_{\text{"pseudo"-value}}
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

## 9. 与通用 DPLR 的效率对比

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
| 数值稳定性 | 需要复杂的 log 域计算 | 简单的对角 log 域 |
| 实测速度 | 基准 | **约 2× 加速** |

**核心洞察**：通过将 $\mathbf{a}$ 和 $\mathbf{b}$ 都绑定到 $\mathbf{k}$，KDA 在保持表达能力的同时大幅减少了计算开销。

---

## 10. 完整伪代码

```python
def chunk_kda(Q, K, V, alpha, beta, chunk_size=64):
    """
    KDA Chunkwise 并行算法
    
    Args:
        Q: [batch, seq_len, d_k] query
        K: [batch, seq_len, d_k] key
        V: [batch, seq_len, d_v] value
        alpha: [batch, seq_len, d_k] channel-wise decay gate, 值域 (0, 1]
        beta: [batch, seq_len] 学习率标量, 值域 [0, 1]
        chunk_size: chunk 大小 C
    
    Returns:
        O: [batch, seq_len, d_v] 输出
    """
    batch, seq_len, d_k = Q.shape
    d_v = V.shape[-1]
    num_chunks = seq_len // chunk_size
    
    # 1. 分 chunk: [batch, num_chunks, chunk_size, d]
    Q = Q.reshape(batch, num_chunks, chunk_size, d_k)
    K = K.reshape(batch, num_chunks, chunk_size, d_k)
    V = V.reshape(batch, num_chunks, chunk_size, d_v)
    alpha = alpha.reshape(batch, num_chunks, chunk_size, d_k)
    beta = beta.reshape(batch, num_chunks, chunk_size)
    
    # 2. 在 log 域计算累积衰减（数值稳定）
    log_alpha = torch.log(alpha + 1e-6)
    # log_gamma_cumsum[r] = sum_{k=1}^{r} log(alpha[k]) = log(gamma^{1->r})
    log_gamma_cumsum = log_alpha.cumsum(dim=-2)  # [batch, num_chunks, C, d_k]
    
    # gamma_1_to_C[r] = gamma^{1->r} = prod_{k=1}^{r} alpha[k]
    gamma_1_to_C = torch.exp(log_gamma_cumsum)  # Gamma_{[t]}^{1->C} 的每一行
    
    # 3. 构建注意力矩阵 A 并求 M（UT 变换）
    # A[r,i] = beta[r] * (k[i] * gamma^{1->i})^T @ (k[r] / gamma^{1->r}) * gamma^{1->r}
    #        = beta[r] * k[i]^T @ diag(gamma^{i->r}) @ k[r]
    K_scaled = gamma_1_to_C * K  # [batch, num_chunks, C, d_k]
    K_inv_scaled = K / (gamma_1_to_C + 1e-6)  # [batch, num_chunks, C, d_k]
    
    # A = diag(beta) @ K_scaled @ K_inv_scaled^T
    A = torch.einsum('...i,...ri,...rj->...rj', beta, K_scaled, K_inv_scaled)
    # 实际上是: A[r,i] = beta[r] * sum_d(K_scaled[i,d] * K_inv_scaled[r,d])
    A = beta.unsqueeze(-1) * torch.einsum('...id,...jd->...ij', K_scaled, K_inv_scaled)
    
    # 取严格下三角
    mask = torch.tril(torch.ones(chunk_size, chunk_size), diagonal=-1)
    A_strict_tril = A * mask
    
    # 求 (I + StrictTril(A))^{-1} 通过前向替换
    # M = (I + StrictTril(A))^{-1} @ diag(beta)
    M = forward_substitution(A_strict_tril, beta)  # 见下方实现
    
    # 4. 计算辅助矩阵 W 和 U
    W = torch.einsum('...rc,...cd->...rd', M, K_scaled)  # [batch, num_chunks, C, d_k]
    U = torch.einsum('...rc,...cd->...rd', M, V)         # [batch, num_chunks, C, d_v]
    
    # 5. 逐 chunk 递归计算输出
    S = torch.zeros(batch, d_k, d_v)  # 初始状态
    O_list = []
    
    for t in range(num_chunks):
        # 当前 chunk 的数据
        Q_t = Q[:, t]           # [batch, C, d_k]
        K_t = K[:, t]           # [batch, C, d_k]
        W_t = W[:, t]           # [batch, C, d_k]
        U_t = U[:, t]           # [batch, C, d_v]
        gamma_t = gamma_1_to_C[:, t]  # [batch, C, d_k]
        log_gamma_t = log_gamma_cumsum[:, t]  # [batch, C, d_k]
        
        # "Pseudo"-value: U - W @ S
        pseudo_V = U_t - torch.einsum('...cd,...dk->...ck', W_t, S)  # [batch, C, d_v]
        
        # Inter-chunk: 查询历史状态
        Q_scaled = gamma_t * Q_t  # [batch, C, d_k]
        inter_chunk = torch.einsum('...cd,...dk->...ck', Q_scaled, S)  # [batch, C, d_v]
        
        # Intra-chunk attention: Tril(Q_scaled @ K_inv_scaled^T) @ pseudo_V
        K_inv_scaled_t = K_t / (gamma_t + 1e-6)
        attn = torch.einsum('...id,...jd->...ij', Q_scaled, K_inv_scaled_t)  # [batch, C, C]
        attn_tril = attn * torch.tril(torch.ones(chunk_size, chunk_size))
        intra_chunk = torch.einsum('...ij,...jd->...id', attn_tril, pseudo_V)  # [batch, C, d_v]
        
        # 输出
        O_t = inter_chunk + intra_chunk
        O_list.append(O_t)
        
        # 更新状态: S_{t+1} = diag(gamma^C) @ S + (Gamma^{i->C} * K)^T @ (U - W @ S)
        gamma_C = gamma_t[:, -1, :]  # [batch, d_k] chunk 末尾的累积衰减
        
        # gamma^{i->C} = gamma^C / gamma^{1->i}
        gamma_i_to_C = gamma_C.unsqueeze(-2) / (gamma_t + 1e-6)  # [batch, C, d_k]
        K_decay_to_end = gamma_i_to_C * K_t  # [batch, C, d_k]
        
        S = torch.diag_embed(gamma_C) @ S + \
            torch.einsum('...cd,...cv->...dv', K_decay_to_end, pseudo_V)
    
    O = torch.stack(O_list, dim=1).reshape(batch, seq_len, d_v)
    return O


def forward_substitution(A_strict_tril, beta):
    """
    求解 M = (I + StrictTril(A))^{-1} @ diag(beta)
    
    利用下三角矩阵的结构，通过前向替换高效计算。
    复杂度: O(C^2) 而非一般矩阵求逆的 O(C^3)
    
    Args:
        A_strict_tril: [batch, num_chunks, C, C] 严格下三角矩阵
        beta: [batch, num_chunks, C] 对角元素
    
    Returns:
        M: [batch, num_chunks, C, C]
    """
    batch, num_chunks, C, _ = A_strict_tril.shape
    
    # (I + L)^{-1} where L is strict lower triangular
    # 可以通过 Neumann 级数或直接前向替换计算
    # 这里用迭代方式: M = diag(beta) - L @ M，从第一行开始逐行计算
    
    M = torch.zeros_like(A_strict_tril)
    for r in range(C):
        # M[r, :] 的计算只依赖于 M[0:r, :]
        # M[r, r] = beta[r]
        # M[r, i] = -sum_{j=i}^{r-1} A[r,j] * M[j,i] for i < r
        M[..., r, r] = beta[..., r]
        for i in range(r):
            M[..., r, i] = -torch.sum(
                A_strict_tril[..., r, :r] * M[..., :r, i], 
                dim=-1
            )
    
    return M
```

---

## 11. 关键要点总结

1. **Chunkwise 并行化**：将序列分成 chunks，chunk 间递归传递状态，chunk 内并行计算

2. **符号体系**：
   - $\boldsymbol{\gamma}_{[t]}^r$：从 chunk 起始到位置 $r$ 的累积衰减（向量）
   - $\boldsymbol{\Gamma}_{[t]}^{1 \to C}$：所有 $\boldsymbol{\gamma}_{[t]}^r$ 堆叠成的矩阵
   - $\boldsymbol{\gamma}_{[t]}^{i \to r}$：从位置 $i$ 到位置 $r$ 的累积衰减

3. **WY 表示法**：将累积的 Householder 变换压缩成紧凑的矩阵形式，支持高效的矩阵乘法

4. **UT 变换**：将递推关系 $\mathbf{u}^r = \beta^r(\mathbf{v}^r - \sum_{i<r}...)$ 转化为矩阵方程 $(\mathbf{I} + \text{StrictTril}(\mathbf{A}))\mathbf{U} = \text{Diag}(\boldsymbol{\beta})\mathbf{V}$，通过前向替换 $O(C^2)$ 高效求解

5. **数值稳定性**：
   - 累积衰减在 log 域计算：`log_gamma = log_alpha.cumsum()`
   - 需要实际值时再 exp：`gamma = exp(log_gamma)`
   - 比值通过 log 差计算：`gamma_ratio = exp(log_gamma_r - log_gamma_i)`

6. **输出的双重结构**：
   - Inter-chunk：历史信息通过递归状态 $\mathbf{S}$ 传递
   - Intra-chunk：当前 chunk 内的注意力可完全并行

7. **效率优势**：相比通用 DPLR，KDA 的约束形式减少了计算量，实现约 2× 加速

8. **硬件友好**：充分利用 Tensor Core 的矩阵乘法能力，最大化 GPU 利用率

---

## 参考

- Kimi Linear Technical Report (arXiv:2510.26692v2)
- Gated DeltaNet [Yang et al., 2025]
- WY Representation [Bischof & Van Loan, 1987]
- UT Transform [Joffrain et al., 2006]