## 高效 Online Softmax：局部抽样策略——从“连续性”到“极值簇”

> **核心洞察**：在长序列 Transformer 推理中，Softmax 的最大值计算是瓶颈。  
> **解法思路**：利用 Key 向量的**局部连续性** + **最大值聚集（Clustered Maxima）**，用「粗抽样 + 局部精化」近似求 max，降计算量 10~20 倍，保持数值稳定。

---

### 1. 问题：Softmax 的“max” 为何昂贵？

Transformer 注意力：

$$
\text{Attention} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

每行 $s_i = q_i K^\top \in \mathbb{R}^n$，需计算：

$$
\text{softmax}(s_i) = \frac{\exp(s_i - \max(s_i))}{\sum \exp(s_i(j) - \max(s_i))}
$$

**推理痛点**：
- $n$ 可达 64k+；
- 每行都要全扫描找 $\max(s_i)$；
- 总复杂度 $O(n)$ / 行 → 随序列长度线性爆炸。

> **目标**：用 $<O(n)$ 的代价，**近似但安全地** 找到一个“够大”的 $\tilde{m}$ 替代 $\max(s_i)$。

---

### 2. 为什么可以“跳着算”？——两大数学性质

#### ① 局部连续性（Locality）
训练后模型的 Key 向量：
- 相邻 token 语义连续；
- 位置编码平滑。

⇒ 注意力分数变化缓慢：

$$
|s_i(j) - s_i(j+1)| \leq L \quad (\text{Lipschitz 条件})
$$

**意义**：分数不会“跳崖”，允许大步长抽样。

---

#### ② 极值聚集性（Clustered Maxima）
最大值不是孤立尖峰，而是一段**高值平台**：

```
分数序列示例：
[0.8, 0.9, 1.0, 0.97, 0.93, 0.4, ...]
       ↑↑↑↑↑  ← 连续高分区（Cluster）
```

⇒ 真实最大值 $j^*$ 附近 $[j^*-r, j^*+r]$ 内，分数都接近最大。

**意义**：找到“粗略峰值”后，只需局部再扫一遍。

---

### 3. 两阶段算法：粗抽样 + 局部精化

| 阶段 | 操作 | 复杂度 |
|------|------|--------|
| **Step 1: 粗抽样** | 每隔 $\Delta$ 个 key 计算 $q_i \cdot k_{j\Delta}$ | $O(n/\Delta)$ |
| | 找到粗略最大值 $\tilde{m}_0$ 和位置 $j^*$ | |
| **Step 2: 局部精化** | 在 $[j^*-r, j^*+r]$ 内精确计算 | $O(r)$ |
| | 得到近似最大值 $\tilde{m}$ | |

**总复杂度**：$O(n/\Delta + r)$  
推荐：$\Delta=8\sim16$，$r=32\sim64$ → 仅原 1/10~1/20 计算量。

---

### 4. 误差分析与安全保障

- **误差上界**（因 Lipschitz）：
  $$
  |\max(s_i) - \tilde{m}| \leq L \cdot \Delta
  $$
- **安全修正**（防 $\tilde{m}$ 略小）：
  $$
  \tilde{m}_{\text{safe}} = \tilde{m} + \delta, \quad \delta \approx 1.0 \sim 2.0
  $$
  ⇒ 保证 $\exp(s_i - \tilde{m}_{\text{safe}}) \leq 1$，避免溢出。

---

### 5. 华为昇腾（Ascend）硬件适配

| 硬件特性 | 优化策略 |
|---------|---------|
| 双向量单元 | Unit1：粗抽样；Unit2：局部精化 |
| 顺序访存快 | $\Delta$ 抽样走连续内存 |
| 带宽受限 | 用双缓冲隐藏延迟 |

**效果**：在 Ascend 上实现高效 pipeline，推理吞吐显著提升。

---

### 6. 理论总结表

| 性质 | 数学表达 | 工程价值 |
|------|----------|----------|
| 局部连续性 | $|s_i(j)-s_i(j+1)| \le L$ | 支持大步长抽样 |
| 极值聚集 | $\exists r, s_i(j) \approx \max \text{ for } |j-j^*|\le r$ | 局部精化即可 |
| 误差控制 | $|m - \tilde{m}| \le L\Delta$ | 可调 $\Delta$ 平衡精度/速度 |
| 安全边界 | $\tilde{m} + \delta$ | 保证数值稳定 |

---

### 7. 结论：一句话讲透

> **用“粗抽样锁定峰值区域 + 局部精化找真 max”，结合安全边界，即可在 1/10 计算量下，实现数值稳定的 Online Softmax。**

---

### 8. 未来扩展方向

- **Top-k 近似**：只在簇内选 k 个；
- **层次化注意力**：多尺度抽样；
- **Token 聚类预筛选**：先分组再抽样；
- **FlashAttention 混合版**：融入本策略做“智能 max”。

---

> **系列续篇预告**：下一文将探讨如何将「局部抽样 max」嵌入 FlashAttention，构建 **Hierarchical FlashAttention**，实现端到端长序列加速。