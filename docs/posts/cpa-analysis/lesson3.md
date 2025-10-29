
# 📘 广义最小二乘法（Generalized Least Squares, GLS）详解

是的！您完全抓住了 **广义最小二乘法（GLS）** 的核心思想。👏
让我们系统地梳理一下这个关键步骤。

---

## ✅ 一、问题设定

我们有线性模型：

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon},
\quad \text{其中 } \mathbb{E}[\boldsymbol{\varepsilon}] = \mathbf{0},
\quad \text{Var}(\boldsymbol{\varepsilon}) = \boldsymbol{\Sigma}
$$

假设：

* $\mathbf{Y}$ 是 $n \times 1$；
* $\mathbf{X}$ 是 $n \times k$（满列秩）；
* $\boldsymbol{\Sigma}$ 是 **已知的、对称正定** 的 $n \times n$ 协方差矩阵。

在经典 OLS 中，我们假设 $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$，此时 OLS 是 BLUE（最佳线性无偏估计）。
但当 $\boldsymbol{\Sigma} \neq \sigma^2 \mathbf{I}$（存在异方差、自相关或跨方程相关等），OLS 虽然仍无偏一致，但**不再有效**（方差不是最小）。

---

## 🔑 二、GLS 的核心思想：寻找变换矩阵 $\mathbf{P}$

目标：找到一个可逆矩阵 $\mathbf{P}$，使得变换后的误差项满足：

$$
\text{Var}(\mathbf{P}\boldsymbol{\varepsilon}) = \mathbf{I}
$$

因为：

$$
\text{Var}(\mathbf{P}\boldsymbol{\varepsilon})
= \mathbf{P} , \text{Var}(\boldsymbol{\varepsilon}) , \mathbf{P}'
= \mathbf{P} \boldsymbol{\Sigma} \mathbf{P}'
$$

所以我们希望：

$$
\mathbf{P} \boldsymbol{\Sigma} \mathbf{P}' = \mathbf{I}
$$

---

## 🧮 三、如何构造这样的 $\mathbf{P}$？

由于 $\boldsymbol{\Sigma}$ 是对称正定矩阵，它有**唯一的对称正定平方根**，记为 $\boldsymbol{\Sigma}^{1/2}$，满足：

$$
\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^{1/2} \boldsymbol{\Sigma}^{1/2}
$$

于是，我们可以取：

$$
\mathbf{P} = \boldsymbol{\Sigma}^{-1/2}
$$

验证：

$$
\mathbf{P} \boldsymbol{\Sigma} \mathbf{P}'
= \boldsymbol{\Sigma}^{-1/2} \boldsymbol{\Sigma} \boldsymbol{\Sigma}^{-1/2}
= \mathbf{I}
$$

✅ 成功！

> 💡 注：$\boldsymbol{\Sigma}^{-1/2}$ 可通过特征值分解或 Cholesky 分解实现。
> 若 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}'$（Cholesky 分解），则可取 $\mathbf{P} = \mathbf{L}^{-1}$，因为：
> $$
> \mathbf{L}^{-1} \boldsymbol{\Sigma} (\mathbf{L}^{-1})'
> = \mathbf{L}^{-1} \mathbf{L} \mathbf{L}' (\mathbf{L}^{-1})' = \mathbf{I}
> $$

---

## 🔄 四、变换后的模型（GLS 模型）

将原模型两边左乘 $\mathbf{P} = \boldsymbol{\Sigma}^{-1/2}$：

$$
\mathbf{P} \mathbf{Y} = \mathbf{P} \mathbf{X} \boldsymbol{\beta} + \mathbf{P} \boldsymbol{\varepsilon}
$$

记：

$$
\mathbf{Y}^* = \mathbf{P} \mathbf{Y},
\quad \mathbf{X}^* = \mathbf{P} \mathbf{X},
\quad \boldsymbol{\varepsilon}^* = \mathbf{P} \boldsymbol{\varepsilon}
$$

则新模型为：

$$
\mathbf{Y}^* = \mathbf{X}^* \boldsymbol{\beta} + \boldsymbol{\varepsilon}^*,
\quad \text{其中 } \text{Var}(\boldsymbol{\varepsilon}^*) = \mathbf{I}
$$

此时满足经典 Gauss-Markov 假设，对变换后的模型做 OLS，即可得到最优估计。

---

## 📌 五、GLS 估计量的显式形式

对变换后模型做 OLS：

$$
\hat{\boldsymbol{\beta}}^{\text{GLS}}
= (\mathbf{X}^{*'} \mathbf{X}^*)^{-1} \mathbf{X}^{*'} \mathbf{Y}^*
= (\mathbf{X}' \mathbf{P}' \mathbf{P} \mathbf{X})^{-1} \mathbf{X}' \mathbf{P}' \mathbf{P} \mathbf{Y}
$$

由于 $\mathbf{P}' \mathbf{P} = \boldsymbol{\Sigma}^{-1}$，所以：

$$
\boxed{
\hat{\boldsymbol{\beta}}^{\text{GLS}}
= (\mathbf{X}' \boldsymbol{\Sigma}^{-1} \mathbf{X})^{-1}
\mathbf{X}' \boldsymbol{\Sigma}^{-1} \mathbf{Y}
}
$$

这就是**广义最小二乘估计量（GLS Estimator）** 的标准形式。

---

## 🌟 六、直觉理解

你说得非常准确：

> “用一个 $P$ 乘上两边，使得 $P\varepsilon$ 的方差回到单位矩阵”

这正是 GLS 的**几何本质**：
通过线性变换将“扭曲”的误差空间（协方差为 $\boldsymbol{\Sigma}$）**球形化**（协方差为 $\mathbf{I}$），从而恢复 OLS 的最优性。

---

## 🔁 七、FGLS（可行广义最小二乘法）

当 $\boldsymbol{\Sigma}$ **未知** 时，我们无法直接应用 GLS。
于是使用以下步骤实现 **可行版本（Feasible GLS, FGLS）**：

1. 用 OLS 得到残差 $\hat{\varepsilon}$；
2. 用 $\hat{\varepsilon}$ 建模估计 $\boldsymbol{\Sigma}$（例如 $\log(\hat{e}_i^2) \sim X$）；
3. 得到 $\hat{\boldsymbol{\Sigma}}$；
4. 构造 $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1/2}$；
5. 对变换后的模型做 OLS。

于是得到：

$$
\hat{\boldsymbol{\beta}}^{\text{FGLS}}
= (\mathbf{X}' \hat{\boldsymbol{\Sigma}}^{-1} \mathbf{X})^{-1}
\mathbf{X}' \hat{\boldsymbol{\Sigma}}^{-1} \mathbf{Y}
$$

> ✅ **GLS**：$\boldsymbol{\Sigma}$ 已知 → 理论最优；
> ✅ **FGLS**：$\boldsymbol{\Sigma}$ 未知 → 用样本估计后近似 GLS。

---

## 🧠 八、总结

| 方法   | 协方差结构                      | 是否已知   | 是否有效（BLUE） | 是否可行   |
| ---- | -------------------------- | ------ | ---------- | ------ |
| OLS  | $\sigma^2 I$               | ✅ 已知   | ✅          | ✅      |
| GLS  | $\boldsymbol{\Sigma}$ 一般形式 | ✅ 已知   | ✅          | ❌ 一般未知 |
| FGLS | $\boldsymbol{\Sigma}$ 一般形式 | ❌ 估计得到 | ✅ 渐近有效     | ✅ 可行   |

---

## ✅ 九、核心结论

> **GLS 的本质**：通过线性变换将误差协方差标准化（球形化），
> 从而在变换后的模型中使用 OLS，恢复最优性。

> **FGLS 的思想**：当协方差未知时，用数据估计其结构并逼近 GLS。

---

**总结一句话：**

> GLS 通过“变换让误差回到同方差”的方式，使我们在一般协方差条件下仍能获得最佳线性无偏估计。
> FGLS 则让这个过程在现实中变得可行。
