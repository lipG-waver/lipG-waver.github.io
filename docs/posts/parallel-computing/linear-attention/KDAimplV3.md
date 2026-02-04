# KDA Chunkwise 算法在昇腾 NPU 上的实现方案 v3

## 版本演进

| 版本 | Cube-Vec 通信次数 | 主要改进 |
|------|------------------|----------|
| v1 | 11 次 | 基础实现 |
| v2 | 11 次 | 修正状态衰减方向 |
| **v3** | **6 次** | 算子融合，通信次数减半 |

---

## 1. 优化核心思路

### 1.1 通信瓶颈分析

v2 版本的 11 次通信：
```
Vec₁ → Cube₂ → Vec₃ → Cube₄ → Vec₅ → Cube₆ₐ → Vec₆ᵦ → Cube₆꜀ → Vec₇ → Cube₈ → Vec₉
```

**关键洞察**：
1. 多个连续的 Vec 操作可以合并
2. 某些 Cube 输出可以直接被下一个 Cube 使用（无需经过 Vec）
3. 预计算可以减少运行时依赖

### 1.2 融合策略

| 融合类型 | 原始阶段 | 融合后 | 节省通信 |
|---------|---------|--------|---------|
| Vec 合并 | Phase 3 + 5 | Phase A | 1 次 |
| Vec 合并 | Phase 6b + 7 | Phase C | 1 次 |
| Vec 合并 | Phase 7 + 9 | Phase D | 1 次 |
| Cube 流水 | Phase 4 → 6a | 共享 L2 | 1 次 |
| 预计算 | gc 复用 | 常驻 L2 | 1 次 |

---

## 2. v3 优化后的流程

### 2.1 新的 6 阶段流水线

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         KDA Chunk v3: 6 次通信                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase A: Vec 超级预处理                                                 ║   │
│  ║  • 累积衰减: gc = cumsum(log(α))                                       ║   │
│  ║  • K_scaled = K ⊙ exp(gc)                                              ║   │
│  ║  • K_div = K ⊙ exp(-gc)                                                ║   │
│  ║  • Q_scaled = Q ⊙ exp(gc)                                              ║   │
│  ║  • decay_to_end = exp(gc[-1] - gc)  ← 预计算，后续复用                  ║   │
│  ║  • K_decay = K ⊙ decay_to_end       ← 预计算，Phase D 使用             ║   │
│  ║  • gamma_C = exp(gc[-1])            ← 预计算，Phase D 使用             ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│         │                                                                      │
│         │ 通信 ①: K_scaled, K_div, Q_scaled, K_decay, V → L2                  │
│         ▼                                                                      │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase B: Cube 注意力 + 辅助矩阵 (融合)                                  ║   │
│  ║  • A_raw = K_scaled @ K_div.T                                          ║   │
│  ║  ↓ 直接传递给下一步，不回 Vec                                           ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│         │                                                                      │
│         │ 通信 ②: A_raw → L2 → Vec                                            │
│         ▼                                                                      │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase C: Vec UT变换 (融合 A⊙β 和前向替换)                               ║   │
│  ║  • A = A_raw ⊙ β[:, None]                                              ║   │
│  ║  • M = forward_substitute(A, β)                                        ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│         │                                                                      │
│         │ 通信 ③: M, K_scaled, V, Q_scaled, S_prev → L2 → Cube                │
│         ▼                                                                      │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase D: Cube 超级矩阵乘 (4 个 matmul 融合)                             ║   │
│  ║  • W = M @ K_scaled                                                    ║   │
│  ║  • U = M @ V                                                           ║   │
│  ║  • WS = W @ S_prev          ← W 直接复用，不传回                        ║   │
│  ║  • O_inter = Q_scaled @ S_prev                                         ║   │
│  ║  • A_qk_raw = Q_scaled @ K_div.T                                       ║   │
│  ║  共 5 个 matmul，但 W 是中间结果不传回                                  ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│         │                                                                      │
│         │ 通信 ④: U, WS, O_inter, A_qk_raw → L2 → Vec                         │
│         ▼                                                                      │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase E: Vec 超级融合 (V_pseudo + tril + 输出预备)                      ║   │
│  ║  • V_pseudo = U - WS                                                   ║   │
│  ║  • A_qk = tril(A_qk_raw)                                               ║   │
│  ║  合并了原 Phase 5 + 6b + 部分 7                                         ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│         │                                                                      │
│         │ 通信 ⑤: A_qk, V_pseudo, K_decay → L2 → Cube                         │
│         ▼                                                                      │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase F: Cube 最终矩阵乘 (2 个 matmul)                                  ║   │
│  ║  • O_intra = A_qk @ V_pseudo                                           ║   │
│  ║  • S_delta = K_decay.T @ V_pseudo                                      ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│         │                                                                      │
│         │ 通信 ⑥: O_inter, O_intra, S_delta → L2 → Vec                        │
│         ▼                                                                      │
│  ══════════════════════════════════════════════════════════════════════════   │
│  ║ Phase G: Vec 最终融合 (输出 + 状态)                                     ║   │
│  ║  • O = O_inter + O_intra                                               ║   │
│  ║  • S_new = gamma_C[:, None] * S_prev + S_delta                         ║   │
│  ║  gamma_C 在 Phase A 已预计算并常驻                                      ║   │
│  ══════════════════════════════════════════════════════════════════════════   │
│                                                                                │
│  输出: O [C, d], S_new [d, d]                                                  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 通信次数对比

| 版本 | 通信模式 | 次数 |
|------|---------|------|
| v2 | V→C→V→C→V→C→V→C→V→C→V | 11 |
| v3 | V→C→V→C→V→C→V | **6** |

**节省 45% 的通信开销！**

---

## 3. 详细实现

### 3.1 主函数

```python
def kda_chunk_v3(Q, K, V, alpha, beta, S_prev, eps=1e-6):
    """
    KDA Chunkwise v3: 通信优化版本
    
    通信次数: 6 次 (vs v2 的 11 次)
    
    Args:
        Q: [C, d_k] = [64, 128]
        K: [C, d_k] = [64, 128]
        V: [C, d_v] = [64, 128]
        alpha: [C, d_k]  衰减门, (0, 1]
        beta: [C]        学习率, [0, 1]
        S_prev: [d_k, d_v] = [128, 128]
    
    Returns:
        O: [C, d_v]
        S_new: [d_k, d_v]
    """
    C, d_k = K.shape
    d_v = V.shape[1]
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase A: Vec 超级预处理                                          ║
    # ║ 目标: 一次性计算所有需要的 scaled/decay 矩阵                      ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Vec():
        # 累积衰减 (log 域)
        log_alpha = log(alpha + eps)
        gc = cumsum(log_alpha, dim=0)         # [C, d_k]
        
        # 基础 scaled 矩阵
        Gamma = exp(gc)                       # [C, d_k]
        Gamma_inv = exp(-gc)                  # [C, d_k]
        
        K_scaled = K * Gamma                  # [C, d_k]
        K_div = K * Gamma_inv                 # [C, d_k]
        Q_scaled = Q * Gamma                  # [C, d_k]
        
        # ★ 预计算: 状态更新所需 (原本在 Phase 7)
        gc_C = gc[-1:, :]                     # [1, d_k]
        decay_to_end = exp(gc_C - gc)         # [C, d_k]
        K_decay = K * decay_to_end            # [C, d_k]
        
        # ★ 预计算: 状态衰减因子 (原本在 Phase 9)
        gamma_C = exp(gc[-1, :])              # [d_k], 常驻 Vec 寄存器
    
    # ═══ 通信 ①: Vec → L2 ═══
    sync_to_L2(K_scaled, K_div, Q_scaled, K_decay, V)
    # gamma_C 保留在 Vec 寄存器中，Phase G 直接使用
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase B: Cube 注意力矩阵                                         ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Cube():
        A_raw = matmul(K_scaled, K_div.T)     # [C, C]
    
    # ═══ 通信 ②: Cube → L2 → Vec ═══
    sync_to_L2(A_raw)
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase C: Vec UT 变换                                             ║
    # ║ 融合: A⊙β + 前向替换                                             ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Vec():
        # A = Diag(β) @ A_raw
        A = A_raw * beta[:, None]             # [C, C]
        
        # M = (I + StrictTril(A))^{-1} @ Diag(β)
        M = forward_substitution_optimized(A, beta)  # [C, C]
    
    # ═══ 通信 ③: Vec → L2 → Cube ═══
    sync_to_L2(M)
    # K_scaled, K_div, Q_scaled, V, S_prev 已在 L2 中
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase D: Cube 超级矩阵乘                                         ║
    # ║ 融合: W, U, WS, O_inter, A_qk_raw (5 个 matmul)                  ║
    # ║ 关键: W 是中间结果，不传回 Vec                                   ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Cube():
        # 辅助矩阵
        W = matmul(M, K_scaled)               # [C, d_k]
        U = matmul(M, V)                      # [C, d_v]
        
        # ★ W @ S_prev 直接在 Cube 中完成，W 不传回
        WS = matmul(W, S_prev)                # [C, d_v]
        
        # 输出相关
        O_inter = matmul(Q_scaled, S_prev)    # [C, d_v]
        A_qk_raw = matmul(Q_scaled, K_div.T)  # [C, C]
    
    # ═══ 通信 ④: Cube → L2 → Vec ═══
    # 注意: W 不传回，只传 U, WS, O_inter, A_qk_raw
    sync_to_L2(U, WS, O_inter, A_qk_raw)
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase E: Vec 超级融合                                            ║
    # ║ 融合: V_pseudo + tril_mask                                       ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Vec():
        # Pseudo-value (原 Phase 5)
        V_pseudo = U - WS                     # [C, d_v]
        
        # Tril mask (原 Phase 6b)
        A_qk = tril_mask(A_qk_raw)            # [C, C]
    
    # ═══ 通信 ⑤: Vec → L2 → Cube ═══
    sync_to_L2(A_qk, V_pseudo)
    # K_decay 已在 L2 中 (Phase A 预计算)
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase F: Cube 最终矩阵乘                                         ║
    # ║ O_intra + S_delta (2 个 matmul)                                  ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Cube():
        O_intra = matmul(A_qk, V_pseudo)      # [C, d_v]
        S_delta = matmul(K_decay.T, V_pseudo) # [d_k, d_v]
    
    # ═══ 通信 ⑥: Cube → L2 → Vec ═══
    sync_to_L2(O_intra, S_delta)
    # O_inter 已在 L2 中
    
    # ╔══════════════════════════════════════════════════════════════════╗
    # ║ Phase G: Vec 最终融合                                            ║
    # ║ 输出合并 + 状态更新                                              ║
    # ║ gamma_C 从 Phase A 常驻寄存器                                    ║
    # ╚══════════════════════════════════════════════════════════════════╝
    with Vec():
        # 输出
        O = O_inter + O_intra                 # [C, d_v]
        
        # 状态更新 (gamma_C 已在寄存器中)
        # Diag(γ^C) @ S_prev: 行缩放
        S_new = gamma_C[:, None] * S_prev + S_delta  # [d_k, d_v]
    
    return O, S_new
```

### 3.2 优化的前向替换

```python
def forward_substitution_optimized(A, beta):
    """
    优化的前向替换算法
    
    计算: M = (I + StrictTril(A))^{-1} @ Diag(β)
    
    优化点:
    1. 利用 Vec 的 8192 元素并行能力
    2. 按行并行计算，减少循环开销
    3. 内存访问模式优化
    
    Args:
        A: [C, C] = [64, 64]
        beta: [C] = [64]
    
    Returns:
        M: [C, C]
    """
    C = A.shape[0]
    
    # L = StrictTril(A)
    L = strict_tril(A)
    
    # R = (I + L)^{-1}
    # 利用下三角结构: R[i,j] = -Σ_{k=j}^{i-1} L[i,k] * R[k,j]
    R = eye(C)
    
    # ★ 优化: 向量化行计算
    for i in range(1, C):
        # 一次计算 R[i, 0:i] 的所有元素
        # R[i, :i] = -L[i, :i] @ R[:i, :i] 的下三角部分
        # 
        # 但由于依赖关系，仍需顺序处理
        # 可以用 Vec 并行计算每个 j 的累加
        for j in range(i):
            # R[i,j] = -Σ_{k=j}^{i-1} L[i,k] * R[k,j]
            # Vec 并行: 一次加载 L[i, j:i] 和 R[j:i, j]
            R[i, j] = -dot(L[i, j:i], R[j:i, j])
    
    # M = R @ Diag(β)
    # 每列乘以对应的 β[j]
    M = R * beta[None, :]
    
    return M


def forward_substitution_vectorized(A, beta):
    """
    更激进的向量化版本
    
    利用 Neumann 级数展开:
    (I + L)^{-1} = I - L + L^2 - L^3 + ...
    
    由于 L 是严格下三角，L^C = 0，级数有限项
    但 C=64 时最多需要 64 项，不实际
    
    替代方案: 分块前向替换
    """
    C = A.shape[0]
    BLOCK = 8  # 分块大小
    
    L = strict_tril(A)
    R = eye(C)
    
    # 按 8x8 块处理
    for bi in range(0, C, BLOCK):
        for bj in range(0, bi, BLOCK):
            # 处理块 R[bi:bi+BLOCK, bj:bj+BLOCK]
            # 依赖 R[bj:bi, bj:bj+BLOCK] 和 L[bi:bi+BLOCK, bj:bi]
            
            # 块内计算可以向量化
            for i in range(bi, min(bi + BLOCK, C)):
                for j in range(bj, min(bj + BLOCK, i)):
                    R[i, j] = -dot(L[i, j:i], R[j:i, j])
        
        # 处理对角块 R[bi:bi+BLOCK, bi:bi+BLOCK]
        for i in range(bi + 1, min(bi + BLOCK, C)):
            for j in range(bi, i):
                R[i, j] = -dot(L[i, j:i], R[j:i, j])
    
    M = R * beta[None, :]
    return M
```

---

## 4. L2 Buffer 优化策略

### 4.1 数据生命周期分析

```
Phase:    A          B          C          D          E          F          G
          │          │          │          │          │          │          │
K_scaled  ████████████████████████████████████       │          │          │
K_div     ████████████████████████████████████████████          │          │
Q_scaled  ██████████████████████████████████████████████████████│          │
V         ████████████████████████████████████       │          │          │
K_decay   ██████████████████████████████████████████████████████████████████
S_prev    ████████████████████████████████████████████████████████████████████ (常驻)
gamma_C   ██ (Vec 寄存器常驻) ██████████████████████████████████████████████████
          │          │          │          │          │          │          │
A_raw     │          ██████████████        │          │          │          │
M         │          │          ████████████          │          │          │
U         │          │          │          ████████████████████████          │
WS        │          │          │          ████████████████████████          │
O_inter   │          │          │          ████████████████████████████████████
A_qk_raw  │          │          │          ████████████████████████          │
A_qk      │          │          │          │          ████████████████████████
V_pseudo  │          │          │          │          ████████████████████████
O_intra   │          │          │          │          │          ████████████████
S_delta   │          │          │          │          │          ████████████████
```

### 4.2 分区策略

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         L2 Buffer 分区 (v3)                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │
│  │   常驻区        │  │   Ping Buffer   │  │   Pong Buffer          │   │
│  │                 │  │                 │  │                        │   │
│  │  S_prev: 32KB   │  │   ~64KB         │  │   ~64KB                │   │
│  │  K_decay: 16KB  │  │                 │  │                        │   │
│  │  Q_scaled: 16KB │  │  轮换使用       │  │  轮换使用              │   │
│  │                 │  │                 │  │                        │   │
│  │  共 64KB        │  │                 │  │                        │   │
│  └─────────────────┘  └─────────────────┘  └────────────────────────┘   │
│                                                                          │
│  总计: ~192KB per head                                                   │
│                                                                          │
│  关键优化:                                                               │
│  • K_decay 在 Phase A 计算后常驻，Phase F 直接使用                       │
│  • Q_scaled 常驻，Phase B 和 D 都需要                                    │
│  • gamma_C 保留在 Vec 寄存器，不占用 L2                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.3 通信数据量对比

| 通信点 | v2 数据量 | v3 数据量 | 说明 |
|--------|----------|----------|------|
| ① | 48 KB | 80 KB | v3 预传 K_decay |
| ② | 8 KB | 8 KB | A_raw |
| ③ | 8 KB | 8 KB | M |
| ④ | 64 KB | 56 KB | v3 不传 W |
| ⑤ | 16 KB | 24 KB | A_qk + V_pseudo |
| ⑥ | 48 KB | 48 KB | O_intra + S_delta + O_inter |
| ⑦-⑪ | 40 KB | - | v3 消除 |
| **总计** | **232 KB** | **224 KB** | 略少 |
| **通信次数** | **11** | **6** | **减少 45%** |

**关键洞察**：通信次数比数据量更重要！每次通信都有固定开销（同步、调度）。

---

## 5. Cube-Vec 时序图 (v3)

```
时间 →
     ═══════════════════════════════════════════════════════════════════════════════

Cube      idle       ║▓▓▓▓▓▓▓║      idle      ║▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓║     idle     ║▓▓▓▓▓▓▓▓▓▓▓▓▓▓║
                     │Phase B│                │      Phase D          │              │   Phase F    │
                     │A_raw  │                │ W,U,WS,O_inter,A_qk   │              │O_intra,S_delt│
                     │1 mm   │                │ 5 mm (W 不传回)        │              │ 2 mm         │
                     └───┬───┘                └───────────┬───────────┘              └──────┬───────┘
                         │                                │                                 │
     L2 Buffer ══════════╪════════════════════════════════╪═════════════════════════════════╪══════════
                         │                                │                                 │
Vec  ║▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓║      ║▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓║        ║▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓║              ║▓▓▓▓▓▓▓▓▓▓▓▓▓║
     │     Phase A       │      │   Phase C      │        │    Phase E       │              │  Phase G    │
     │ 预处理 + 预计算    │      │   UT 变换      │        │ V_pseudo + tril  │              │ O + S_new   │
     │ gc,Γ,K_sc,K_dec.. │      │ A⊙β + M       │        │                  │              │             │
     └───────────────────┘      └────────────────┘        └──────────────────┘              └─────────────┘

Phase:        A                B        C                D                 E                 F         G

通信:         ①────────────────②────────③────────────────④─────────────────⑤─────────────────⑥

v3: 6 次通信  ●                         ●                                  ●                          ●
v2: 11次通信  ●    ●    ●    ●    ●    ●    ●    ●    ●    ●    ●
```

---

## 6. 多 Chunk 流水线优化

### 6.1 Chunk 间重叠

```python
def kda_multi_chunk_pipeline(chunks_data, S_init):
    """
    多 Chunk 流水线处理
    
    关键: 当 Chunk[t] 执行 Phase F 时，
         Chunk[t+1] 可以开始 Phase A
    """
    num_chunks = len(chunks_data)
    S = S_init
    O_list = []
    
    # 预热: 第一个 chunk
    Q0, K0, V0, alpha0, beta0 = chunks_data[0]
    
    # Phase A[0]
    precomputed_0 = vec_phase_A(K0, Q0, alpha0)
    sync_to_L2(precomputed_0)
    
    for t in range(num_chunks):
        Q, K, V, alpha, beta = chunks_data[t]
        
        if t == 0:
            precomputed = precomputed_0
        else:
            precomputed = precomputed_next
        
        # Phase B[t]
        A_raw = cube_phase_B(precomputed)
        sync_to_L2(A_raw)
        
        # Phase C[t] || 预取 Phase A[t+1]
        if t + 1 < num_chunks:
            # 并行执行
            parallel(
                lambda: vec_phase_C(A_raw, beta),
                lambda: begin_prefetch_phase_A(chunks_data[t+1])
            )
        else:
            M = vec_phase_C(A_raw, beta)
        
        sync_to_L2(M)
        
        # Phase D[t]
        U, WS, O_inter, A_qk_raw = cube_phase_D(M, precomputed, V, S)
        sync_to_L2(U, WS, O_inter, A_qk_raw)
        
        # Phase E[t] || 完成 Phase A[t+1]
        if t + 1 < num_chunks:
            parallel(
                lambda: vec_phase_E(U, WS, A_qk_raw),
                lambda: finish_prefetch_phase_A(chunks_data[t+1])
            )
            precomputed_next = get_prefetch_result()
        else:
            V_pseudo, A_qk = vec_phase_E(U, WS, A_qk_raw)
        
        sync_to_L2(V_pseudo, A_qk)
        
        # Phase F[t]
        O_intra, S_delta = cube_phase_F(A_qk, V_pseudo, precomputed['K_decay'])
        sync_to_L2(O_intra, S_delta)
        
        # Phase G[t]
        O, S = vec_phase_G(O_inter, O_intra, S, S_delta, precomputed['gamma_C'])
        O_list.append(O)
    
    return stack(O_list), S
```

### 6.2 流水线时序

```
Chunk:     0                    1                    2
           │                    │                    │
      ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
      │         │          │         │          │         │
Vec   │ A₀      │          │ A₁      │          │ A₂      │
      │    ┌────┴────┐     │    ┌────┴────┐     │    ┌────┴────┐
      │    │ C₀      │     │    │ C₁      │     │    │ C₂      │
      │    │    ┌────┴────┐│    │    ┌────┴────┐│    │    ┌────┴────┐
      │    │    │ E₀      ││    │    │ E₁      ││    │    │ E₂      │
      │    │    │    ┌────┴┴────┤    │    ┌────┴┴────┤    │    ┌────┴───
      │    │    │    │ G₀       │    │    │ G₁       │    │    │ G₂
      └────┴────┴────┴──────────┴────┴────┴──────────┴────┴────┴────────
           │         │          │         │          │         │
Cube       │ B₀      │          │ B₁      │          │ B₂      │
           │    ┌────┴────┐     │    ┌────┴────┐     │    ┌────┴────┐
           │    │ D₀      │     │    │ D₁      │     │    │ D₂      │
           │    │    ┌────┴────┐│    │    ┌────┴────┐│    │    ┌────┴──
           │    │    │ F₀      ││    │    │ F₁      ││    │    │ F₂
           └────┴────┴─────────┴┴────┴────┴─────────┴┴────┴────┴────────

重叠区域: A₁ 与 E₀ 并行, A₂ 与 E₁ 并行 ...
```

---

## 7. 性能预估

### 7.1 单 Chunk 计算量

| Phase | 计算类型 | FLOPs | 说明 |
|-------|---------|-------|------|
| A | Vec | ~80K | 6 个 [C,d] 逐元素 |
| B | Cube | 524K | 1 mm: [C,d]@[d,C] |
| C | Vec | ~12K | A⊙β + forward_sub |
| D | Cube | 4.7M | 5 mm (W 中间结果) |
| E | Vec | ~12K | U-WS + tril |
| F | Cube | 1.5M | 2 mm |
| G | Vec | ~20K | O + S_new |

**总计**: Cube ~6.7M MACs, Vec ~124K ops

### 7.2 通信开销对比

假设每次通信固定开销 T_sync = 1μs，带宽 B = 100 GB/s

| 版本 | 同步开销 | 数据传输 | 总通信时间 |
|------|---------|---------|-----------|
| v2 | 11 × 1μs = 11μs | 232KB / 100GB/s = 2.3μs | **13.3μs** |
| v3 | 6 × 1μs = 6μs | 224KB / 100GB/s = 2.2μs | **8.2μs** |

**通信时间减少 38%！**

### 7.3 端到端预估

| 组件 | v2 时间 | v3 时间 | 说明 |
|------|--------|--------|------|
| Cube 计算 | 3μs | 3μs | 相同 |
| Vec 计算 | 0.5μs | 0.5μs | 相同 |
| 通信 | 13.3μs | 8.2μs | **-38%** |
| **总计** | **16.8μs** | **11.7μs** | **-30%** |

---

## 8. 完整参考实现 (PyTorch)

```python
import torch

def kda_chunk_v3_reference(Q, K, V, alpha, beta, S_prev, eps=1e-6):
    """
    KDA Chunkwise v3 参考实现
    用于验证昇腾实现的正确性
    """
    C, d_k = K.shape
    d_v = V.shape[1]
    
    # ═══ Phase A: 超级预处理 ═══
    log_alpha = torch.log(alpha + eps)
    gc = torch.cumsum(log_alpha, dim=0)
    
    Gamma = torch.exp(gc)
    Gamma_inv = torch.exp(-gc)
    
    K_scaled = K * Gamma
    K_div = K * Gamma_inv
    Q_scaled = Q * Gamma
    
    # 预计算
    gc_C = gc[-1:, :]
    decay_to_end = torch.exp(gc_C - gc)
    K_decay = K * decay_to_end
    gamma_C = torch.exp(gc[-1, :])
    
    # ═══ Phase B: A_raw ═══
    A_raw = K_scaled @ K_div.T
    
    # ═══ Phase C: UT 变换 ═══
    A = A_raw * beta[:, None]
    
    L = torch.tril(A, diagonal=-1)
    R = torch.eye(C, device=A.device, dtype=A.dtype)
    for i in range(1, C):
        for j in range(i):
            R[i, j] = -torch.sum(L[i, j:i] * R[j:i, j])
    M = R * beta[None, :]
    
    # ═══ Phase D: 超级 matmul ═══
    W = M @ K_scaled
    U = M @ V
    WS = W @ S_prev  # W 是中间结果
    O_inter = Q_scaled @ S_prev
    A_qk_raw = Q_scaled @ K_div.T
    
    # ═══ Phase E: 融合 ═══
    V_pseudo = U - WS
    A_qk = torch.tril(A_qk_raw)
    
    # ═══ Phase F: 最终 matmul ═══
    O_intra = A_qk @ V_pseudo
    S_delta = K_decay.T @ V_pseudo
    
    # ═══ Phase G: 最终融合 ═══
    O = O_inter + O_intra
    S_new = gamma_C[:, None] * S_prev + S_delta
    
    return O, S_new


def test_v3_correctness():
    """验证 v3 与 v2 结果一致"""
    torch.manual_seed(42)
    C, d = 64, 128
    
    Q = torch.randn(C, d)
    K = torch.randn(C, d)
    V = torch.randn(C, d)
    alpha = torch.sigmoid(torch.randn(C, d))
    beta = torch.sigmoid(torch.randn(C))
    S_prev = torch.randn(d, d)
    
    # v3 结果
    O_v3, S_v3 = kda_chunk_v3_reference(Q, K, V, alpha, beta, S_prev)
    
    # v2 结果 (直接复制 v2 逻辑)
    O_v2, S_v2 = kda_chunk_v2_reference(Q, K, V, alpha, beta, S_prev)
    
    # 比较
    O_diff = (O_v3 - O_v2).abs().max().item()
    S_diff = (S_v3 - S_v2).abs().max().item()
    
    print(f"O max diff: {O_diff:.2e}")
    print(f"S max diff: {S_diff:.2e}")
    
    assert O_diff < 1e-5, "O mismatch!"
    assert S_diff < 1e-5, "S mismatch!"
    print("✓ v3 与 v2 结果一致")


def kda_chunk_v2_reference(Q, K, V, alpha, beta, S_prev, eps=1e-6):
    """v2 参考实现 (用于对比)"""
    C, d_k = K.shape
    d_v = V.shape[1]
    
    log_alpha = torch.log(alpha + eps)
    gc = torch.cumsum(log_alpha, dim=0)
    Gamma = torch.exp(gc)
    Gamma_inv = torch.exp(-gc)
    
    K_scaled = K * Gamma
    K_div = K * Gamma_inv
    Q_scaled = Q * Gamma
    
    A_raw = K_scaled @ K_div.T
    A = A_raw * beta[:, None]
    
    L = torch.tril(A, diagonal=-1)
    R = torch.eye(C, device=A.device, dtype=A.dtype)
    for i in range(1, C):
        for j in range(i):
            R[i, j] = -torch.sum(L[i, j:i] * R[j:i, j])
    M = R * beta[None, :]
    
    W = M @ K_scaled
    U = M @ V
    WS = W @ S_prev
    V_pseudo = U - WS
    
    O_inter = Q_scaled @ S_prev
    A_qk = torch.tril(Q_scaled @ K_div.T)
    O_intra = A_qk @ V_pseudo
    O = O_inter + O_intra
    
    gc_C = gc[-1:, :]
    decay_to_end = torch.exp(gc_C - gc)
    K_decay = K * decay_to_end
    S_delta = K_decay.T @ V_pseudo
    
    gamma_C = torch.exp(gc[-1, :])
    S_new = gamma_C[:, None] * S_prev + S_delta
    
    return O, S_new


if __name__ == "__main__":
    test_v3_correctness()
```

---

## 9. 总结

### 9.1 v3 核心优化

| 优化策略 | 效果 | 实现方式 |
|---------|------|---------|
| Vec 阶段合并 | -3 次通信 | Phase A 预计算 decay, Phase E 合并 tril |
| Cube 中间结果复用 | -1 次通信 | W 不传回 Vec |
| 常驻数据 | -1 次通信 | gamma_C 保留在寄存器 |

### 9.2 版本对比

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| Cube-Vec 通信 | 11 次 | 11 次 | **6 次** |
| 状态衰减 | ❌ 错误 | ✓ 正确 | ✓ 正确 |
| 预计算利用 | 无 | 无 | **K_decay, gamma_C** |
| 估计延迟 | ~17μs | ~17μs | **~12μs** |

### 9.3 进一步优化方向

1. **二级 Chunk 分块**：对于更长序列，引入 mini-chunk 减少前向替换的串行依赖
2. **混合精度**：A 矩阵和 M 矩阵可用 FP16，累积衰减用 FP32
3. **异步预取**：利用空闲组预取下一 chunk 的数据
4. **算子融合到 Cube**：如果硬件支持 fused multiply-add with mask

---

## 附录: 昇腾 AscendC 伪代码框架

```cpp
// kda_chunk_v3.cpp
#include "kernel_operator.h"

using namespace AscendC;

class KDAChunkV3 {
public:
    __aicore__ inline KDAChunkV3() {}
    
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                 GM_ADDR alpha, GM_ADDR beta,
                                 GM_ADDR s_prev, GM_ADDR o, GM_ADDR s_new) {
        // 初始化 Global Memory 指针
        // 分配 L2 Buffer
    }
    
    __aicore__ inline void Process() {
        // Phase A: Vec 超级预处理
        PhaseA_VecPreprocess();
        SyncL2();  // 通信 ①
        
        // Phase B: Cube A_raw
        PhaseB_CubeAttention();
        SyncL2();  // 通信 ②
        
        // Phase C: Vec UT 变换
        PhaseC_VecUTTransform();
        SyncL2();  // 通信 ③
        
        // Phase D: Cube 超级 matmul
        PhaseD_CubeSuperMatmul();
        SyncL2();  // 通信 ④
        
        // Phase E: Vec 超级融合
        PhaseE_VecSuperFusion();
        SyncL2();  // 通信 ⑤
        
        // Phase F: Cube 最终 matmul
        PhaseF_CubeFinalMatmul();
        SyncL2();  // 通信 ⑥
        
        // Phase G: Vec 最终融合
        PhaseG_VecFinalFusion();
    }
    
private:
    __aicore__ inline void PhaseA_VecPreprocess() {
        // Vec Kernel
        LocalTensor<float> log_alpha = ...;
        LocalTensor<float> gc = ...;
        
        // log + cumsum
        Ln(log_alpha, alpha_local);
        // cumsum 需要手动实现或调用库函数
        
        // exp
        Exp(Gamma, gc);
        Exp(Gamma_inv, Muls(gc, -1.0f));
        
        // scaled 矩阵
        Mul(K_scaled, K_local, Gamma);
        Mul(K_div, K_local, Gamma_inv);
        Mul(Q_scaled, Q_local, Gamma);
        
        // 预计算
        // decay_to_end = exp(gc[-1] - gc)
        // K_decay = K * decay_to_end
        // gamma_C = exp(gc[-1])  保留在寄存器
    }
    
    __aicore__ inline void PhaseB_CubeAttention() {
        // Cube Kernel
        // A_raw = K_scaled @ K_div.T
        Matmul(A_raw, K_scaled, K_div_T);
    }
    
    // ... 其他 Phase 实现
};

extern "C" __global__ __aicore__ void kda_chunk_v3_kernel(...) {
    KDAChunkV3 op;
    op.Init(...);
    op.Process();
}
```