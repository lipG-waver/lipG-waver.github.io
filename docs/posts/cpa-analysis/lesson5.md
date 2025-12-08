# 截面与面板数据分析问题集4：知识点讲解

---

## 第1题：正态分布的尺度变换

### 题目

设正态分布的概率密度函数（均值 $\mu$，标准差 $\sigma$）为 $\phi(x; \mu, \sigma)$，对应的累积分布函数为 $\Phi(x; \mu, \sigma)$。设 $c$ 为正常数。

**(a)** $\phi(cx; \mu, c\sigma)$ 的形式是什么？

**(b)** $\Phi(cx; \mu, c\sigma)$ 的形式是什么？

### 知识点

**正态分布的密度函数**：
$$\phi(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**核心性质**：正态分布在线性变换下保持正态。若 $X \sim N(\mu, \sigma^2)$，则 $aX + b \sim N(a\mu + b, a^2\sigma^2)$。

**解题思路**：将 $cx$ 代入密度函数，观察标准化变量 $\frac{cx - \mu}{c\sigma} = \frac{x - \mu/c}{\sigma}$ 的形式，从而建立与原分布的联系。

---

## 第2题：离散分布的MLE与MOM

### 题目

随机变量 $X$ 有如下概率质量函数：
$$P(X = x) = \begin{cases} \frac{\theta}{2}, & x = 1 \\ 2\theta(1-\theta), & x = 2 \\ (1-\theta)^2, & x = 3 \end{cases}$$

其中 $0 < \theta < 1$。假设有i.i.d.样本 $X_1 = 1, X_2 = 2, X_3 = 1$。

**(a)** 求 $\theta$ 的最大似然估计量。

**(b)** 基于期望建立矩条件，求矩估计量。

**(c)** (a)和(b)的结果是否相同？

### 知识点

**最大似然估计（MLE）**

似然函数（样本中：两个1，一个2）：
$$L(\theta) = \left(\frac{\theta}{2}\right)^2 \cdot 2\theta(1-\theta) = \frac{\theta^3(1-\theta)}{2}$$

对数似然：
$$\ell(\theta) = 3\log\theta + \log(1-\theta) + \text{const}$$

一阶条件：
$$\frac{d\ell}{d\theta} = \frac{3}{\theta} - \frac{1}{1-\theta} = 0$$

解得：$3(1-\theta) = \theta \implies \hat{\theta}_{ML} = \frac{3}{4}$

**一般形式**：设样本中有 $n_1$ 个1，$n_2$ 个2，$n_3$ 个3，则：
$$\hat{\theta}_{ML} = \frac{n_1 + n_2}{n_1 + 2n_2 + 2n_3}$$

**矩估计（MOM）**

总体期望：
$$E[X] = 1 \cdot \frac{\theta}{2} + 2 \cdot 2\theta(1-\theta) + 3 \cdot (1-\theta)^2$$

展开计算：
$$E[X] = \frac{\theta}{2} + 4\theta - 4\theta^2 + 3 - 6\theta + 3\theta^2 = 3 - \frac{3\theta}{2} - \theta^2$$

样本均值：$\bar{X} = \frac{1+2+1}{3} = \frac{4}{3}$

令 $E[X] = \bar{X}$，解二次方程得 $\hat{\theta}_{MOM} \approx 0.743$

**MLE与MOM的比较**：本题中 $\hat{\theta}_{ML} = 0.75$ 与 $\hat{\theta}_{MOM} \approx 0.743$ 非常接近（差异<1%），但不完全相等。

> **注**：具体结果请以课堂讲解为准。

### MLE与MOM何时相等？

两者给出相同结果的条件：

1. **指数族分布**：当分布属于指数族且使用充分统计量对应的矩条件时，两者一致
2. **矩条件等价于Score方程**：如果 $E[g(X)] = 0$ 恰好等价于 $E[\partial \log f / \partial \theta] = 0$

### MLE与MOM的效率比较

| 性质 | MLE | MOM |
|------|-----|-----|
| 一致性 | ✓ | ✓ |
| 渐近正态性 | ✓ | ✓ |
| **渐近有效性** | ✓ 达到Cramér-Rao下界 | ✗ 一般不具备 |

**MOM为何通常不是渐近有效的？**

MOM的渐近方差取决于所选矩条件的形式。只有当矩条件 $g(X)$ 恰好与Score函数成比例时，MOM才能达到Cramér-Rao下界。

---

## 第3题：Laplace分布的参数估计

### 题目

设 $X_1, X_2, \ldots, X_N$ 是i.i.d.样本，概率密度函数为：
$$f(x; \sigma) = \frac{1}{2\sigma} \exp\left(-\frac{|x|}{\sigma}\right)$$

**(a)** 求 $\sigma$ 的最大似然估计量。

**(b)** 基于期望建立矩条件，求矩估计量。

**(c)** (a)和(b)的结果是否相同？

### 知识点

**Laplace分布的性质**：
- $E[X] = 0$（关于0对称）
- $E[|X|] = \sigma$
- $\text{Var}(X) = 2\sigma^2$

**MLE**：$\hat{\sigma}_{ML} = \frac{1}{N}\sum_{i=1}^N |x_i|$

**MOM**（使用 $E[|X|] = \sigma$）：$\hat{\sigma}_{MOM} = \frac{1}{N}\sum_{i=1}^N |x_i|$

**结论**：MLE = MOM，因为 $E[|X|]$ 的矩条件恰好与MLE的一阶条件等价。

---

## 第4题：均匀分布MLE的局限性

### 题目

设 $X_i, i = 1, \ldots, N$ 是i.i.d.样本，服从 $[0, \theta]$ 上的均匀分布，其中 $\theta > 0$。概率密度为：
$$f_X(x) = \begin{cases} 1/\theta & 0 \leq x \leq \theta \\ 0 & \text{otherwise} \end{cases}$$

求 $\theta$ 的最大似然估计量，并讨论为什么它在这种情况下可能不是一个好的估计量。

### MLE的推导

似然函数：
$$L(\theta) = \frac{1}{\theta^N} \cdot \mathbf{1}_{\theta \geq X_{(N)}}$$

在 $\theta \geq X_{(N)}$ 的约束下，$L(\theta)$ 关于 $\theta$ 单调递减，因此：
$$\hat{\theta}_{ML} = X_{(N)} = \max\{X_1, \ldots, X_N\}$$

### MLE渐近理论的正则条件

MLE具有良好渐近性质需要满足以下**正则条件（Regularity Conditions）**：

**条件1：参数空间**
- 真实参数 $\theta_0$ 位于参数空间 $\Theta$ 的**内点**（interior point）
- 参数空间 $\Theta$ 是 $\mathbb{R}^k$ 的开子集

**条件2：支撑独立于参数**
- 分布的支撑 $\{x: f(x;\theta) > 0\}$ **不依赖于参数** $\theta$
- 对所有 $\theta \in \Theta$，密度函数在相同的 $x$ 集合上为正

**条件3：可微性**
- 对数似然函数 $\log f(x;\theta)$ 关于 $\theta$ **三次可微**
- 导数可以在积分号下交换

**条件4：Fisher信息的正则性**
- Fisher信息矩阵 $\mathcal{I}(\theta)$ 存在且**正定**
- $0 < \mathcal{I}(\theta) < \infty$

**条件5：识别性**
- 不同参数值对应不同的分布
- $\theta_1 \neq \theta_2 \implies f(x;\theta_1) \neq f(x;\theta_2)$

### 均匀分布违反了哪些条件？

**违反条件2**：支撑 $[0, \theta]$ 依赖于参数 $\theta$

这导致：
- 似然函数在 $\hat{\theta} = X_{(N)}$ 处不可微
- 无法对似然函数做Taylor展开
- Score函数的期望不为零
- Fisher信息的常规定义失效

### 均匀分布MLE的问题

| 问题 | 说明 |
|------|------|
| **有偏性** | $E[X_{(N)}] = \frac{N}{N+1}\theta < \theta$ |
| **非渐近正态** | 极限分布不是正态分布 |
| **异常收敛速度** | 以 $O(1/N)$ 而非 $O(1/\sqrt{N})$ 收敛 |

**修正**：无偏估计量 $\tilde{\theta} = \frac{N+1}{N} X_{(N)}$

---

## 第5题：线性回归的MLE

### 题目

考虑线性回归模型：
$$y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$$

其中所有 $\varepsilon_i$ 在给定 $x_i$ 的条件下是i.i.d.正态分布，均值为0，方差为 $\sigma^2$。

**(a)** 样本的似然函数是什么？

**(b)** $\beta$ 的MLE是什么？

**(c)** 当 $x_i$ 是虚拟变量时，推导MLE的具体形式。

### 知识点

**似然函数**：
$$L(\beta_0, \beta_1, \sigma^2) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \beta_0 - \beta_1 x_i)^2}{2\sigma^2}\right)$$

**MLE与OLS的等价性**：正态误差假设下，最大化对数似然等价于最小化残差平方和，因此 MLE = OLS。

---

## 第6题：Fisher信息矩阵（核心重点）

### 题目

对于来自指数分布的随机样本 $D_1, \ldots, D_N \stackrel{iid}{\sim} \text{Exp}(\theta)$，密度函数为：
$$f_D(d; \theta) = \frac{1}{\theta} \exp\left(-\frac{d}{\theta}\right), \quad d \in \mathbb{R}^+$$

**(a)** 该样本的Score和Hessian是什么？

**(b)** Score在真实参数处的期望是什么？

**(c)** Hessian在真实参数处的期望是什么？它与Fisher信息的关系是什么？

**(d)** 不使用Fisher信息，推导MLE的渐近方差，并验证它们之间的关系。

---

### 基本概念

**Score函数**：对数似然关于参数的一阶导数

单个观测：$s(d; \theta) = \frac{\partial \log f}{\partial \theta} = -\frac{1}{\theta} + \frac{d}{\theta^2}$

样本Score：$S(\theta) = \sum_{i=1}^N s(D_i; \theta) = -\frac{N}{\theta} + \frac{\sum D_i}{\theta^2}$

**关键性质**：$E[S(\theta_0)] = 0$

**Hessian**：对数似然关于参数的二阶导数

单个观测：$h(d; \theta) = \frac{\partial^2 \log f}{\partial \theta^2} = \frac{1}{\theta^2} - \frac{2d}{\theta^3}$

---

### Fisher信息的三个等价定义

$$\mathcal{I}(\theta) = E\left[\left(\frac{\partial \log f}{\partial \theta}\right)^2\right] = -E\left[\frac{\partial^2 \log f}{\partial \theta^2}\right] = \text{Var}\left(\frac{\partial \log f}{\partial \theta}\right)$$

**本题计算**：
$$\mathcal{I}(\theta) = -E[h(D; \theta)] = -\frac{1}{\theta^2} + \frac{2\theta}{\theta^3} = \frac{1}{\theta^2}$$

---

### Fisher信息为什么重要？

#### 1. Cramér-Rao下界：估计精度的理论极限

任何无偏估计量 $\tilde{\theta}$ 的方差满足：
$$\text{Var}(\tilde{\theta}) \geq \frac{1}{N \cdot \mathcal{I}(\theta)}$$

Fisher信息决定了我们能够多精确地估计参数——这是物理上的极限，与估计方法无关。

#### 2. MLE的最优性

$$\sqrt{N}(\hat{\theta}_{ML} - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{\mathcal{I}(\theta_0)}\right)$$

MLE渐近达到Cramér-Rao下界，是"最有效"的估计量。

#### 3. 似然曲率的几何解释

| 似然函数形状 | Fisher信息 | 估计精度 |
|-------------|-----------|---------|
| 尖锐的峰（曲率大） | 大 | 高 |
| 平坦的峰（曲率小） | 小 | 低 |

**直觉**：峰越尖锐，偏离真实参数时似然下降越快，参数越容易被精确定位。

---

### Fisher信息的广泛应用

#### 应用1：实验设计

选择最大化Fisher信息的实验条件。例如在回归中，$x$ 值应尽量分散以最大化 $\sum(x_i - \bar{x})^2$。

#### 应用2：假设检验的三大检验

| 检验 | 基本思想 |
|------|---------|
| Wald检验 | $\hat{\theta}$ 离 $\theta_0$ 有多远？ |
| 似然比检验 | 约束/无约束似然之比 |
| Score检验 | 在 $\theta_0$ 处Score是否为0？ |

Score检验统计量：$LM = \frac{S(\theta_0)^2}{N \cdot \mathcal{I}(\theta_0)} \xrightarrow{d} \chi^2(1)$

#### 应用3：贝叶斯推断中的Jeffreys先验

$$\pi(\theta) \propto \sqrt{\mathcal{I}(\theta)}$$

具有参数变换不变性。

#### 应用4：信息几何

Fisher信息定义了统计流形上的黎曼度量：
$$D_{KL}(p_\theta \| p_{\theta+d\theta}) \approx \frac{1}{2} d\theta^T \mathcal{I}(\theta) d\theta$$

---

### Fisher信息与梯度下降的深层联系

#### 形式上的相似性

两者都涉及：梯度（Score）、二阶结构（Fisher/Hessian）、寻找最优点。

但**本质不同**：

| | Fisher信息 | 梯度下降 |
|---|-----------|---------|
| **本质** | 参数空间的几何度量 | 迭代优化方法 |
| **关注对象** | 距离、信息量、方差下界 | 找到最小值 |
| **数学对象** | 二阶期望矩阵 | 一阶梯度方向 |

#### 自然梯度下降：两者的完美结合

普通梯度下降：
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

自然梯度下降：
$$\theta_{t+1} = \theta_t - \eta \cdot \mathcal{I}(\theta)^{-1} \nabla_\theta L(\theta_t)$$

**直觉理解：在地图上走路 vs 在真实地球上走路**

- **普通梯度下降** = 在平面地图上走，不知道地形实际弯曲程度
- **Fisher信息** = 告诉你地形实际上是怎样的（哪里陡、哪里平）
- **自然梯度** = 拿着真实三维地图走最短路径

Fisher信息描述的是：
- 某些参数方向很"敏感"（信息大）→ 步长要小
- 某些参数方向"不敏感"（信息小）→ 步长可以大

#### 与Ridge回归的联系

**关键洞察**：Ridge回归的惩罚矩阵是Fisher信息的粗糙近似。

| 方法 | 校正矩阵 |
|------|---------|
| Ridge回归 | $\lambda I$（各方向相同惩罚） |
| 广义Ridge | $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$（各方向不同） |
| Fisher信息近似 | $\Lambda$ 与数据敏感性一致 |
| 自然梯度 | $\mathcal{I}(\theta)$（完整矩阵，包含方向相关性） |

**如果Ridge的 $\lambda_i$ 能对每个方向独立调节，并且与该方向的敏感度成比例，它就趋近于Fisher信息。**

这解释了为什么Ridge的超参数要与特征值在同一数量级——本质上都是在做"几何校正"。

#### 在各类模型中的应用

| 模型 | Fisher信息的作用 |
|------|-----------------|
| **Logistic回归** | 自动调整：$p \approx 0.5$ 时步长大，$p \approx 0,1$ 时步长小 |
| **Softmax** | 压缩冗余方向，只在有效方向更新 |
| **深度网络** | K-FAC算法：Fisher信息的近似，大幅加速训练 |
| **强化学习** | TRPO、PPO：在策略空间做自然梯度，保证稳定更新 |
| **变分推断** | KL散度的二阶近似系数就是Fisher信息 |

---

### 本题完整计算

**(a) Score和Hessian**

$$S(\theta) = -\frac{N}{\theta} + \frac{\sum D_i}{\theta^2}, \quad H(\theta) = \frac{N}{\theta^2} - \frac{2\sum D_i}{\theta^3}$$

**(b) Score期望**

$$E[S(\theta_0)] = N \cdot \left(-\frac{1}{\theta_0} + \frac{\theta_0}{\theta_0^2}\right) = 0$$

**(c) Hessian期望与Fisher信息**

$$E[H(\theta_0)] = N \cdot \left(\frac{1}{\theta_0^2} - \frac{2\theta_0}{\theta_0^3}\right) = -\frac{N}{\theta_0^2}$$

关系：$\mathcal{I}(\theta_0) = -\frac{1}{N}E[H(\theta_0)] = \frac{1}{\theta_0^2}$

**(d) 渐近方差推导**

由Taylor展开和中心极限定理：
$$\sqrt{N}(\hat{\theta} - \theta_0) \xrightarrow{d} N\left(0, \frac{1}{\mathcal{I}(\theta_0)}\right) = N(0, \theta_0^2)$$

---

## 总结

| 概念 | 核心作用 |
|------|---------|
| **Score** | 指示参数调整方向，期望为0 |
| **Hessian** | 描述似然曲率 |
| **Fisher信息** | 度量参数可识别程度，定义参数空间几何 |
| **Cramér-Rao下界** | 无偏估计方差的理论极限 |
| **自然梯度** | 在概率流形上的最优下降方向 |
| **正则条件** | MLE渐近理论成立的前提 |

**核心洞察**：Fisher信息不仅是统计学中的理论工具，更是连接统计推断、优化算法、信息几何的桥梁。从Ridge回归到深度学习，从假设检验到强化学习，"考虑参数空间的几何结构"这一思想贯穿始终。

---

# 第二部分：在线学习导论

---

## 引言：从滑雪租赁问题说起

### 一个简单的决策困境

假设你刚搬到一个滑雪胜地，面临一个决策：

- **租雪板**：每天花费 ¥100
- **买雪板**：一次性花费 ¥1000

问题是：**你不知道自己会滑几天**。可能滑几天就厌倦了，也可能爱上滑雪成为终身爱好。

如果你事先知道会滑多少天：
- 滑 ≤10 天：租划算（花费 ≤ ¥1000）
- 滑 >10 天：买划算（花费固定 ¥1000）

但现实是：**你必须在不知道未来的情况下做决策**。

这就是**在线学习（Online Learning）**的核心场景——在信息逐步揭示的过程中做出不可撤回的决策。

---

## 确定性策略的困境

### 策略1：一直租

如果你决定"永远只租不买"：
- 滑10天：花费 ¥1000 ✓
- 滑100天：花费 ¥10000 ✗（本可以只花 ¥1000）

最坏情况：你的花费是最优解的 **无穷倍**！

### 策略2：一直买

如果你决定"第一天就买"：
- 滑100天：花费 ¥1000 ✓
- 滑1天：花费 ¥1000 ✗（本可以只花 ¥100）

最坏情况：你的花费是最优解的 **10倍**。

### 策略3：租到第k天再买

假设你决定"租k-1天，第k天买"：
- 总花费 = $(k-1) \times 100 + 1000$

分析：
- 如果实际只滑了 $k-1$ 天：最优花费 = $(k-1) \times 100$，你的花费多了 ¥1000
- 如果实际滑了很多天：最优花费 = ¥1000，你多花了 $(k-1) \times 100$

**最优的确定性策略**：$k = 10$（租9天，第10天买）

此时：
- 总花费 = $900 + 1000 = ¥1900$
- 最优花费 = ¥1000（事后看应该第一天就买）
- **竞争比 = 1900/1000 = 1.9**

---

## 竞争比：在线算法的评价标准

### 定义

$$\text{竞争比} = \frac{\text{在线算法的花费}}{\text{事后最优的花费}}$$

我们希望找到一个策略，使得**无论未来发生什么**，竞争比都不超过某个常数 $c$。

### 滑雪问题的确定性下界

**定理**：对于滑雪租赁问题，任何确定性策略的竞争比至少为 $2 - 1/B$，其中 $B$ = 买价/租价 = 10。

即：最优确定性策略的竞争比 = $2 - 0.1 = 1.9$

**证明思路**：无论你选择哪一天买，对手（adversary）总可以选择让你刚好在最坏的时间点做决策。

---

## 对抗性环境与确定性策略的局限

### 为什么确定性策略有问题？

如果你的策略是**确定的**（比如"第10天买"），一个恶意的对手可以：
- 观察你的策略
- 选择让你损失最大的情况

例如：如果对手知道你第10天会买，他可以安排让你第9天就必须停止滑雪（比如滑雪场关闭）。这样你租了9天花了¥900，而最优策略只需花¥900，看起来没亏。但如果对手让你滑到第11天才停，你花了¥1900，最优只需¥1000。

**关键洞察**：确定性策略在对抗性环境（adversarial setting）中是脆弱的。

---

## 随机化策略：打破对抗性困境

### 核心思想

如果对手**无法预测**你的决策，他就无法针对性地设计最坏情况。

**随机化策略**：不是确定"第k天买"，而是按某个概率分布随机选择购买时间。

### 滑雪问题的随机化策略

设 $p_k$ 为"在第k天购买"的概率（假设之前一直在租）。

**最优随机策略**：

$$p_k = \frac{1}{B} \cdot \left(1 - \frac{1}{B}\right)^{-(k-1)} \quad \text{for } k = 1, 2, \ldots, B$$

这是一个**截断的几何分布**。

### 随机化策略的竞争比

**定理**：存在随机化策略，使得滑雪租赁问题的**期望竞争比**为：

$$\frac{e}{e-1} \approx 1.58$$

这比最优确定性策略的 1.9 更好！

**直觉**：
- 随机化让对手无法针对性地设计最坏情况
- 对手只能针对你的**期望行为**，而不是确切行为
- 这种"不可预测性"本身就是一种优势

---

## 从滑雪问题到一般框架

### 在线学习的基本设定

1. **时间步 $t = 1, 2, \ldots, T$**
2. 每一步，学习者选择一个动作 $a_t \in \mathcal{A}$
3. 环境揭示损失函数 $\ell_t: \mathcal{A} \to \mathbb{R}$
4. 学习者承受损失 $\ell_t(a_t)$

**目标**：最小化**遗憾（Regret）**

$$\text{Regret}_T = \sum_{t=1}^T \ell_t(a_t) - \min_{a \in \mathcal{A}} \sum_{t=1}^T \ell_t(a)$$

遗憾 = 实际累计损失 - 事后最优固定策略的累计损失

### 与竞争比的关系

- **竞争比**：乘法形式，$\text{ALG} \leq c \cdot \text{OPT}$
- **遗憾**：加法形式，$\text{ALG} \leq \text{OPT} + R_T$

好的在线算法追求 $R_T = o(T)$，即平均遗憾趋于零。

---

## 对抗性 vs 随机性环境

### 三种环境假设

| 环境类型 | 特点 | 难度 |
|---------|------|------|
| **随机环境** | 损失从固定分布i.i.d.采样 | 最容易 |
| **遗忘对手** | 对手事先固定所有损失序列 | 中等 |
| **自适应对手** | 对手可以根据你的历史决策调整 | 最难 |

**随机化策略的威力**：即使面对自适应对手，随机化也能提供保护，因为对手只能针对你的策略分布，而非具体实现。

---

## 经典算法：Multiplicative Weights

### 专家问题（Expert Problem）

- 有 $N$ 个"专家"，每天给出建议
- 你需要选择听从哪个专家
- 事后揭示每个专家建议的损失

### Multiplicative Weights Update (MWU)

**初始化**：每个专家权重 $w_i^{(1)} = 1$

**每轮 $t$**：
1. 按权重比例选择专家：$p_i^{(t)} = \frac{w_i^{(t)}}{\sum_j w_j^{(t)}}$
2. 观察损失 $\ell_i^{(t)}$ 
3. 更新权重：$w_i^{(t+1)} = w_i^{(t)} \cdot (1-\eta)^{\ell_i^{(t)}}$

**遗憾界**：
$$\text{Regret}_T \leq O(\sqrt{T \log N})$$

即：平均遗憾 $\to 0$，速度为 $O(\sqrt{\log N / T})$

---

## 在线学习的现实应用

### 应用1：动态定价

电商平台每天面临定价决策：
- 价格高：利润率高但可能卖不出
- 价格低：销量大但利润薄

消费者行为事先未知，需要在线学习最优价格。

### 应用2：广告投放

- 有多个广告位和多个广告
- 每次展示后观察点击/转化
- 需要在线分配广告到广告位

### 应用3：推荐系统

- 向用户推荐内容
- 观察用户反馈（点击、停留时间）
- 动态调整推荐策略

### 应用4：网络路由

- 数据包需要选择传输路径
- 网络拥塞状况动态变化
- 需要在线选择最优路由

---

## 与统计学习的对比

| 维度 | 统计学习 | 在线学习 |
|------|---------|---------|
| **数据假设** | i.i.d.从固定分布采样 | 可以是对抗性的 |
| **学习模式** | 批量学习（先收集再训练） | 逐步学习（边做边学） |
| **评价标准** | 泛化误差、样本复杂度 | 遗憾、竞争比 |
| **目标** | 学习"真实"规律 | 与最优策略竞争 |
| **典型问题** | 分类、回归、聚类 | 专家问题、在线凸优化 |

**Fisher信息的联系**：在随机环境的在线学习中，Fisher信息同样决定了参数估计的极限精度。自然梯度在在线凸优化中也有重要应用（Online Natural Gradient）。

---

## 总结：在线学习的核心思想

1. **不确定性是本质的**：未来不可预测，必须在信息不完全时做决策

2. **竞争比/遗憾是合理的评价标准**：不追求"最优"（不可能），而是追求"与最优相比不会太差"

3. **随机化是对抗对手的武器**：不可预测性本身就是价值

4. **在线学习 ≠ 统计学习**：不假设数据来自固定分布，允许对抗性环境

5. **理论与实践的桥梁**：从滑雪租赁到推荐系统，在线学习无处不在

---

## 延伸阅读

- **《Prediction, Learning, and Games》** by Cesa-Bianchi & Lugosi：在线学习的经典教材
- **《Introduction to Online Convex Optimization》** by Elad Hazan：现代在线凸优化
- **Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems**：多臂老虎机综述