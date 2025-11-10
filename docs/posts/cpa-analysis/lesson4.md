# 🎓 GLS / SUR 模型设定错误的思想逻辑讲解

## 一、问题背景

我们有一个系统方程：

$$
\begin{cases}
y_{1i} = \beta_1 x_{1i} + \varepsilon_{1i} \\
y_{2i} = \beta_2 x_{2i} + \beta_3 x_{3i} + \varepsilon_{2i}
\end{cases}
$$

分析者使用 **GLS（广义最小二乘）** 来估计整个系统，但错误地 **遗漏了第二个方程中的 \(x_3\)**。

我们关心的问题是：

> 这个错误会不会导致第一个方程的 $\hat{\beta}_1$ 不再一致？

---

## 二、直觉与思想核心

### 1️⃣ OLS：方程彼此独立

如果使用 OLS：
- 每个方程单独估计；
- 因此只要第一个方程设定正确，$\hat{\beta}_1$ 就是一致的；
- 第二个方程的错误与第一个方程无关。

👉 **OLS 不会被别的方程“拖下水”。**

---

### 2️⃣ GLS / SUR：方程之间共享信息

GLS 不同。  
它假设两个方程的误差项 $\varepsilon_1$ 和 $\varepsilon_2$ 有协方差：

$$
\operatorname{Cov}(\varepsilon_{1i}, \varepsilon_{2i}) = \sigma_{12} \neq 0
$$

因此，GLS 会用协方差结构来“联合估计”两个方程，从而提高效率。

但是——

如果第二个方程漏掉了 $x_3$，  
$\varepsilon_{2i}$ 中就混入了 $x_3$ 的信息。

一旦 GLS 把 $\varepsilon_{2i}$ 的信息“借给”第一个方程，
就等于把这个污染一并带入。于是：

> 即使第一个方程原本设定正确，$\hat{\beta}_1$ 也会偏误、不再一致。

---

## 三、污染的传播机制

GLS 的思想是：

$$
\hat{\beta} = (X' \Psi^{-1} X)^{-1} X' \Psi^{-1} y
$$

其中 $\Psi$ 是误差协方差矩阵。  
当 $\sigma_{12} \neq 0$ 时，两个方程的残差被“混合加权”。

若第二个方程误设：
- $\varepsilon_2$ 中含有遗漏变量 $x_3$；
- 若 $x_3$ 又与 $x_1$ 或 $x_2$ 有关；
- 则这一部分相关性会经由 $\sigma_{12}$ 传播到 $\hat{\beta}_1$。

这就是 GLS 的“传染效应”：

> 一处设错，整体受害。

---

## 四、什么时候不会被污染

根据推导结果，$\hat{\beta}_1$ 一致需要以下任意条件成立：

1. **误差不相关：** $\sigma_{12} = 0$  
   没有信息交换，GLS 退化为 OLS。

2. **遗漏变量与解释变量无关：**  
   $$
   \mathbb{E}[x_{1i}x_{3i}] = 0, \quad \mathbb{E}[x_{2i}x_{3i}] = 0
   $$
   即便方程误差相关，污染无法通过相关路径传递。

换句话说：

> 只要误差不相通，或者遗漏变量与现有变量独立，就安全。

---

## 五、形象比喻

- **OLS 像独立器官：**  
  每个方程自己运行，出错互不影响。

- **GLS 像血液循环系统：**  
  一处感染（设错）会通过血管（误差相关）传播到全身（其他方程）。

因此：

> GLS 的强大之处在于“共享信息”，  
> 弱点也正在于“共享错误”。

---

## 六、总结要点

| 方面 | OLS | GLS / SUR |
|------|------|------------|
| 方程估计方式 | 各自独立 | 联合估计（共享误差信息） |
| 误差协方差利用 | 否 | 是 |
| 对设定错误的敏感性 | 局部 | 全局（可传播） |
| $\hat{\beta}_1$ 一致的条件 | 自身模型正确即可 | 需他方模型也正确，或 $\sigma_{12}=0$，或 $x_3$ 独立 |

---

## ✳️ 教学启发

当学生问“为什么要学 SUR？”时：

> 因为它让我们看到：**效率与稳健性往往不能兼得。**  
> 共享信息的同时，也必须承担共享错误的风险。



# 🎓 一、直觉篇：GMM在干什么？

## 🧩 1. 出发点：矩条件 (Moment Conditions)

在统计学里，我们经常知道某些关于模型参数的**期望约束**应该成立，比如：

$$
\mathbb{E}[z_i \varepsilon_i(\theta)] = 0
$$

这叫作一个**矩条件**。
例如在回归模型中：
$$
y_i = x_i'\beta + \varepsilon_i, \quad \text{且 } \mathbb{E}[x_i \varepsilon_i] = 0
$$

这就是说，“解释变量和误差项不相关” 是我们对模型正确性的假设。
→ 于是我们就可以通过这些约束来“倒推出” $\beta$。

---

## 🧭 2. 核心思想：让样本矩尽量接近理论矩

理论上：
$$
\mathbb{E}[z_i \varepsilon_i(\theta)] = 0
$$

在样本中，用平均值近似期望：
$$
g_n(\theta) = \frac{1}{n} \sum_{i=1}^n z_i \varepsilon_i(\theta)
$$

GMM 的思想就是：

> 找一个 $\hat{\theta}$，让这些样本矩条件尽量接近 0。

即：
$$
\hat{\theta}*{GMM} = \arg\min*{\theta} ; g_n(\theta)' W g_n(\theta)
$$

其中 $W$ 是一个权矩阵，用来控制不同矩条件的重要性。

---

# 🧮 二、数学篇：GMM的形式化定义

## 1️⃣ 模型设定

我们有样本 ${y_i, x_i, z_i}_{i=1}^n$，
参数 $\theta$ 满足：
$$
\mathbb{E}[f(y_i, x_i, z_i, \theta)] = 0
$$

记：
$$
g_n(\theta) = \frac{1}{n}\sum_{i=1}^n f(y_i, x_i, z_i, \theta)
$$

---

## 2️⃣ 目标函数（加权平方距离）

GMM 定义为：
$$
\hat{\theta}*{GMM} = \arg\min*{\theta} Q_n(\theta)
$$

其中：
$$
Q_n(\theta) = g_n(\theta)' W g_n(\theta)
$$

* 当 $W = I$（单位矩阵）时，这就是**Method of Moments (MM)**。
* 当 $W = \text{Cov}[g_n(\theta)]^{-1}$ 时，就是**最优 GMM**。

---

## 3️⃣ 一阶条件（GMM 的核心方程）

求导后得：
$$
\frac{\partial Q_n(\theta)}{\partial \theta}
= 2 G_n' W g_n(\theta) = 0
$$

其中：
$$
G_n = \frac{\partial g_n(\theta)}{\partial \theta'}
$$

这组方程决定了 GMM 估计量的数值解。

---

# 🧱 三、应用篇：OLS 是 GMM 的一个特例

让我们看看熟悉的线性回归模型：

$$
y_i = x_i'\beta + \varepsilon_i, \quad \text{且 } \mathbb{E}[x_i \varepsilon_i] = 0
$$

对应的矩条件是：
$$
g_n(\beta) = \frac{1}{n}\sum_i x_i (y_i - x_i'\beta) = 0
$$

GMM 的目标函数是：
$$
Q(\beta) = g_n(\beta)' W g_n(\beta)
$$

如果选择：

* 工具变量 $z_i = x_i$
* 权矩阵 $W = I$

那么：
$$
Q(\beta) = (y - X\beta)'(y - X\beta)
$$

→ **这就是 OLS！**

---

# 🧠 四、GMM的意义

| 层次                  | 含义                                             |
| ------------------- | ---------------------------------------------- |
| **OLS**             | 一种特殊的 GMM（$z_i=x_i$, $W=I$）                    |
| **IV（工具变量）**        | GMM 的另一种特例（$z_i$ 是工具变量）                        |
| **最优GMM**           | 使用 $\text{Cov}[g_n(\theta)]^{-1}$ 作为权矩阵，使估计量有效 |
| **系统GMM / 动态面板GMM** | 在复杂场景下的扩展：动态模型、自相关、异方差等                        |

---

# 📘 五、总结一句话

> **GMM 是一个统一的框架：**
>
> * 把“模型成立”转化为“矩条件为零”；
> * 把“估计”转化为“让样本矩尽量接近理论矩”；
> * 所有的 OLS、IV、2SLS、SUR……
>   都只是它的不同特例。

---

# ✅ 六、快速复盘

| 名称            | Moment Conditions                    | 权矩阵 $W$                | 特点     |
| ------------- | ------------------------------------ | ---------------------- | ------ |
| OLS           | $\mathbb{E}[x_i(y_i-x_i'\beta)] = 0$ | $I$                    | 简单、有效  |
| IV / 2SLS     | $\mathbb{E}[z_i(y_i-x_i'\beta)] = 0$ | $(Z'Z)^{-1}$           | 工具变量   |
| Efficient GMM | 同上                                   | $\text{Cov}[g_i]^{-1}$ | 最优有效估计 |
| System GMM    | 多方程矩条件                               | Block 权矩阵              | 面板动态模型 |

# Problem 3：线性 IV-GMM 的权矩阵选择与效率

## 模型与矩条件

线性模型：
$$
y_i = x_i' \beta + \varepsilon_i, \qquad \mathbb{E}[z_i \varepsilon_i]=0,
$$
其中 $z_i$ 为工具变量（$n\times M$，$M\ge K=\dim(\beta)$）。

样本矩向量：
$$
g_n(\beta)=\frac1n\sum_{i=1}^n z_i\big(y_i-x_i'\beta\big).
$$

一般 GMM 估计量：
$$
\widehat\beta(W)=\arg\min_\beta\; Q_n(\beta)=g_n(\beta)'Wg_n(\beta)
\quad\Rightarrow\quad
\widehat\beta(W)=\big(X'ZWZ'X\big)^{-1}X'ZWZ' y.
$$

### 记号
- $A \equiv \mathbb{E}[z_i x_i']\;(K\times M)$  
- $S \equiv \mathbb{E}[\varepsilon_i^2 z_i z_i']\;(M\times M)$ —— “矩条件的协方差”  
- 渐近方差（一般权矩阵 $W$）：
  $$
  \text{AVAR}\big(\widehat\beta(W)\big)
  = (A'WA)^{-1}\,A'WSWA\,(A'WA)^{-1}.
  $$

---

## (a) 什么时候需要权矩阵？若用单位权会得到什么？

- **只在过识别时需要权矩阵：**  
  当 $M=K$（恰好识别）时，$g_n(\beta)=0$ 就能直接解出 $\beta$，$W$ 不影响结果；  
  当 **$M>K$（过识别）** 时，矩条件多于未知参数，需要 $W$ 来“加权”各条件的重要性。

- **单位权矩阵 $W=I$**（可用但通常不是最优）时：
  $$
  \widehat\beta(I)=\big(X'ZZ'X\big)^{-1}X'ZZ' y .
  $$
  > 注意：它与 2SLS 不同（2SLS 等价于取 $W=(Z'Z)^{-1}$）。

---

## (b) 在 (a) 的同方差、无序列相关设定下的最优权矩阵？

若 $\varepsilon_i$ **同方差且不相关**（$\operatorname{Var}(\varepsilon_i|z_i)=\sigma^2$）：
$$
S=\mathbb{E}[\varepsilon_i^2 z_i z_i']=\sigma^2\,\mathbb{E}[z_i z_i'] \quad\Rightarrow\quad
W_{\text{opt}}=S^{-1}\propto\big(\mathbb{E}[z_i z_i']\big)^{-1}.
$$
样本实现可以取
$$
\widehat W_{\text{opt}} \propto (Z'Z/n)^{-1}.
$$
代回闭式解得
$$
\widehat\beta\!\left((Z'Z)^{-1}\right)
=\big(X'Z(Z'Z)^{-1}Z'X\big)^{-1}X'Z(Z'Z)^{-1}Z' y,
$$
这**正是 2SLS**。

---

## (c) 若存在异方差，最优权矩阵是什么？

异方差时 $S=\mathbb{E}[\varepsilon_i^2 z_i z_i']$ 不再与 $\mathbb{E}[z_i z_i']$ 成比例。  
**最优权矩阵始终是**：
$$
W_{\text{opt}}=S^{-1}=\big(\mathbb{E}[\varepsilon_i^2 z_i z_i']\big)^{-1}.
$$
样本中用两步 GMM 估计：  
1) 先用某个初始 $W$（如 $(Z'Z)^{-1}$ 或 $I$）得到 $\widehat\beta^{(1)}$；  
2) 构造 $\widehat S=\frac1n\sum_i \widehat\varepsilon_i^{\,2} z_i z_i'$，取 $W=\widehat S^{-1}$，再算 $\widehat\beta^{(2)}$（即**有效 GMM**）。

---

## (d) 三种权矩阵的比较与影响

- **单位权 $W=I$**：一致但**低效**。  
  渐近方差
  $$
  \text{AVAR}\big(\widehat\beta(I)\big)
  =(A'A)^{-1}A'SA(A'A)^{-1}.
  $$

- **同方差最优权 $W\propto(Z'Z)^{-1}$（2SLS）**：  
  在同方差下达到有效性；若实际存在异方差，则一致但低效。

- **异方差稳健最优权 $W=S^{-1}$（两步/迭代 GMM）**：  
  **无论是否异方差**都不劣于其他权矩阵；其渐近方差达到
  $$
  \text{AVAR}\big(\widehat\beta(S^{-1})\big)
  = (A'S^{-1}A)^{-1},
  $$
  且对任意对称正定 $W$ 有
  $$
  \text{AVAR}\big(\widehat\beta(W)\big)
  \succeq (A'S^{-1}A)^{-1}\quad\text{(半正定序)}.
  $$

**要点总结：**
1. $M=K$ 时不需要 $W$；$M>K$ 时必须选 $W$（过识别）。  
2. $W=I$ 可行但不高效；$W=(Z'Z)^{-1}$ 给出 2SLS（在同方差下有效）。  
3. **有效 GMM** 一律取 $W=S^{-1}$；异方差情形必须用它（或其稳健估计）才能效率最优。

---

## 🧩 一、GMM 的目标函数

我们最小化

$$
Q_n(\theta) = g_n(\theta)' W g_n(\theta),
$$

其中
$$
g_n(\theta) = \frac{1}{n}\sum_{i=1}^n g_i(\theta), \quad g_i(\theta) = z_i \varepsilon_i(\theta).
$$

矩条件假设：
$$
\mathbb{E}[g_i(\theta_0)] = 0,
$$
其中 (\theta_0) 是真参数。

---

## 🧮 二、一阶条件（First Order Condition）

对 (Q_n(\theta)) 关于 (\theta) 求导并在估计点 (\hat{\theta}) 处置零：

$$
\frac{\partial Q_n}{\partial \theta}
= 2 G_n' W g_n(\hat{\theta}) = 0,
$$

其中
$$
G_n = \frac{\partial g_n(\theta)}{\partial \theta'}.
$$

由此可得近似：
$$
\hat{\theta} - \theta_0 \approx - (G' W G)^{-1} G' W g_n(\theta_0).
$$

---

## 📈 三、渐近方差（Asymptotic Variance）

由中心极限定理：
$$
\sqrt{n}, g_n(\theta_0) \xrightarrow{d} \mathcal{N}(0, S),
$$

其中
$$
S = \mathrm{Var}(g_i(\theta_0)) = \mathbb{E}[g_i g_i'].
$$

代入前式：

$$
\sqrt{n}(\hat{\theta} - \theta_0)
= -(G' W G)^{-1} G' W \sqrt{n} g_n(\theta_0)
$$

于是：
$$
\mathrm{Avar}(\hat{\theta})
= (G' W G)^{-1} G' W S W G (G' W G)^{-1}.
$$

---

## 💡 四、选择最优 (W) 的原则

我们希望找到使方差最小的 (W)。

显然 (S) 是由数据决定的（即真实矩条件的方差），
唯一可控的是 (W)。
我们要最小化矩阵意义下的：

$$
A(W) = (G' W G)^{-1} G' W S W G (G' W G)^{-1}.
$$

---

## 🔍 五、最优 (W) 的推导结果

Hansen（1982）证明（矩阵微积分可验证）：

> **当 (W = S^{-1}) 时，上式取最小值。**

即：
$$
W_{\text{opt}} = S^{-1} = [\mathrm{Var}(g_i(\theta_0))]^{-1}.
$$

此时渐近方差退化为最小值形式：
$$
\mathrm{Avar}(\hat{\theta}_{\text{eff}})
= (G' S^{-1} G)^{-1}.
$$

---

## 🧠 六、直觉解释

1. (S = \mathrm{Var}(g_i)) 表示不同矩条件的波动和相关性；
2. 方差大的矩条件“噪声多”，应降低权重；
3. 相关的矩条件“信息重复”，应下调其组合权；
4. 取 (S^{-1}) 就相当于“去相关 + 方差标准化”，
   即最优加权，使所有条件在目标函数中等权但无冗余。

> 换句话说：**最优 GMM 就是用矩条件的协方差逆来白化（whiten）矩向量。**

---

## ✅ 七、总结一句话

| 层次   | 含义                                                           |
| ---- | ------------------------------------------------------------ |
| 一致性  | 任意正定 (W) 都能得到一致估计                                            |
| 效率性  | 使 (\mathrm{Avar}(\hat{\theta})) 最小化的 (W) 是 (S^{-1})          |
| 几何意义 | (S^{-1}) 相当于在矩空间中“去除噪声的形状”                                   |
| 实际操作 | 先用任意 (W_0) 初估，再用残差估计 (\hat{S})，取 (\hat{W}=\hat{S}^{-1}) 二步迭代 |


---
# Problem 4：当误差协方差矩阵为对角时，System 2SLS = 3SLS

## 1. 模型设定与记号

有 \(G\) 个方程、每个方程有 \(n\) 个观测，堆叠后写为

$$
y_i = X_i \beta + \varepsilon_i,\qquad
y_i=\begin{bmatrix}y_{1i}\\ \vdots\\ y_{Gi}\end{bmatrix},\ 
X_i=\operatorname{diag}(x_{1i},\dots,x_{Gi}),\ 
\beta=\begin{bmatrix}\beta_1\\ \vdots\\ \beta_G\end{bmatrix}.
$$

工具变量按方程块对角给出
$$
Z_i=\operatorname{diag}(z_{1i},\dots,z_{Gi}),
$$
满足相关性与外生性。
把所有样本堆叠：
$$
Y = [y_1;\dots;y_n], \quad
X = [X_1;\dots;X_n], \quad
Z = [Z_1;\dots;Z_n].
$$

误差协方差（跨方程）设为：
$$
\Sigma = \mathrm{Var}(\varepsilon_i).
$$

矩条件（系统 IV-GMM）为：
$$
\mathbb{E}[Z'\varepsilon] = 0.
$$

---

## 2. 两个系统估计量的通式

### 2.1 System 2SLS（等价于 GMM 权矩阵 \(W=(Z'Z)^{-1}\)）
$$
\widehat\beta_{\text{sys-2SLS}}
= \big(X'Z(Z'Z)^{-1}Z'X\big)^{-1}\,X'Z(Z'Z)^{-1}Z'Y .
$$

### 2.2 3SLS（等价于“有效 GMM”，权矩阵为最优 \(W=\Omega^{-1}\)）
先记
$$
\Omega \equiv \mathrm{Var}(Z'\varepsilon) = Z'(\Sigma\otimes I_n)Z .
$$
则
$$
\widehat\beta_{\text{3SLS}}
= \big(X'Z\,\Omega^{-1} Z'X\big)^{-1}\,X'Z\,\Omega^{-1} Z'Y .
$$

> 直觉：3SLS 之所以优于 2SLS，是因为 \(\Omega^{-1}\) 会利用**跨方程误差相关性**来提高效率。

---

## 3. 关键情形：\(\Sigma\) 为对角

若
$$
\Sigma=\operatorname{diag}(\sigma_{11},\dots,\sigma_{GG}),
$$
则
$$
\Omega = Z'(\Sigma\otimes I_n)Z
       = \operatorname{diag}\!\big(\sigma_{11}Z_1'Z_1,\ \dots,\ \sigma_{GG}Z_G'Z_G\big),
$$
是按方程的**块对角矩阵**。因此
$$
\Omega^{-1}
= \operatorname{diag}\!\Big(\tfrac{1}{\sigma_{11}}(Z_1'Z_1)^{-1},\ \dots,\ \tfrac{1}{\sigma_{GG}}(Z_G'Z_G)^{-1}\Big).
$$

又因为 \(X=\operatorname{diag}(X_1,\dots,X_G)\)、\(Z=\operatorname{diag}(Z_1,\dots,Z_G)\) 亦为块对角，  
于是 3SLS 的矩阵乘法**在方程层面完全分块**，第 \(g\) 个方程的 3SLS 估计化为

$$
\widehat\beta^{\,(3SLS)}_{g}
= \big(X_g'Z_g (Z_g'Z_g)^{-1} Z_g'X_g\big)^{-1}\,
   X_g'Z_g (Z_g'Z_g)^{-1} Z_g'Y_g .
$$

这正是**方程逐个的 2SLS 公式**，而 system 2SLS 对块对角 \(Z\) 也等价于对每个方程做 2SLS。  
因此
$$
\widehat\beta_{\text{3SLS}}
\equiv \widehat\beta_{\text{sys-2SLS}}
\quad\text{（数值相同）}.
$$

该结论与模型**是否过识别**无关：在过识别时只是 \(Z_g\) 列多于 \(X_g\)，上式仍成立。

---

## 4. 一句话直觉

**3SLS 的优势来自“利用跨方程误差相关性”。**  
当 \(\Sigma\) 为对角时 **没有跨方程相关性**，3SLS 的最优加权退化为对每个方程用 \((Z_g'Z_g)^{-1}\) 的独立加权，  
于是与（system）2SLS 完全一致。

第四题的**核心含义**是：

> **当系统中不同方程的误差项互不相关（即协方差矩阵 Σ 是对角矩阵）时，三阶段最小二乘法 (3SLS) 和系统两阶段最小二乘法 (system 2SLS) 给出的估计结果是完全相同的。**

下面我帮你拆解这句话的**理论含义、背后逻辑和教学重点**👇

---

## 🎯 一、题目想考什么

这题不是在考你推公式，而是在考你理解：

> **3SLS 比 2SLS 强在哪里？**
> **当这种“强”失效时，它们为什么会变成一样？**

也就是说：
**3SLS 的优势来自利用跨方程误差的相关性。**
一旦这些误差互不相关（Σ 为对角），3SLS 的“系统性”信息优势就消失，它就退化成了 system 2SLS。

---

## 🧩 二、背景知识对比

| 方法              | 估计方式           | 信息利用          | 有效性               |
| --------------- | -------------- | ------------- | ----------------- |
| **2SLS（单方程）**   | 每个方程独立做两阶段最小二乘 | 只用该方程的工具变量    | 一致，但不利用系统信息       |
| **system 2SLS** | 多方程同时估计（矩阵堆叠）  | 仍假设方程误差独立     | 与单方程 2SLS 数值相同    |
| **3SLS**        | 系统性三阶段估计       | 利用方程间误差相关性（Σ） | 若 Σ 有非零协方差项，则效率更高 |

---

## ⚙️ 三、逻辑推理

在一般的多方程系统中：

$$
\varepsilon =
\begin{bmatrix}
\varepsilon_1\
\varepsilon_2\
\vdots\
\varepsilon_G
\end{bmatrix}
\quad\text{且}\quad
\mathrm{Var}(\varepsilon_i) = \Sigma \otimes I_n
$$

* 如果不同方程间的误差**相关**，即 Σ 有非零协方差项，
  那么知道一个方程的残差信息，可以改进另一个方程的估计。
  ⇒ 3SLS 可以用 GLS 框架综合利用所有方程的信息，提高效率。

* 如果 Σ **对角**（即误差不相关），
  那系统误差的协方差矩阵就是块对角矩阵，
  每个方程的估计完全独立，系统估计的加权也会分块进行。
  ⇒ 3SLS 就退化为逐方程的 2SLS。

---

## 💡 四、直觉比喻

你可以这样对学生解释：

> 想象你要同时预测多个学生的考试分数，每个学生的方程是一个回归模型。
>
> * 如果学生之间成绩高度相关（比如都受相同老师影响），那你可以利用这些相关性——这是 3SLS。
> * 但如果每个学生完全独立（互不影响），那你没法通过别人来改进预测——此时 3SLS 退化成 2SLS。

---

## 🧮 五、数学形式简化（背后的核心结果）

3SLS 的估计式是：

$$
\hat{\beta}_{3SLS}
= (X'Z,\Omega^{-1},Z'X)^{-1} X'Z,\Omega^{-1},Z'Y,
$$

其中：
$$
\Omega = Z' (\Sigma \otimes I_n) Z
$$

当 Σ 对角时：
$$
\Omega = \text{blockdiag}\big(\sigma_{11} Z_1'Z_1, \dots, \sigma_{GG}Z_G'Z_G \big)
$$

于是矩阵乘法在每个方程上都分块计算，
每个方程的结果就变成：

$$
\hat{\beta}_g = (X_g'Z_g (Z_g'Z_g)^{-1} Z_g'X_g)^{-1} X_g'Z_g (Z_g'Z_g)^{-1} Z_g'Y_g,
$$

——这正是 2SLS 的公式。
所以 system 2SLS 与 3SLS 数值相同。

---

## ✅ 六、结论与启发

> 当误差项跨方程无关时，3SLS 不再比 system 2SLS 更有效。
>
> 二者的估计值与渐近方差完全相同。

**启发：**

* 3SLS 的效率来自“跨方程误差相关性”。
* 如果没有相关性，就不要额外复杂化建模。
* 在应用中可以先检验方程误差是否相关（比如用 Breusch-Pagan 系统相关性检验），决定是否采用 3SLS。

---

是否希望我把这段总结写成一份**课堂讲义风格的 Markdown 版本**？
包括公式、比喻、结论三部分，方便直接展示。
