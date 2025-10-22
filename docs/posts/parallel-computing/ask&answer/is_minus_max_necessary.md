# Flash Attention中是否必须减去最大值？

## 研究问题

在Flash Attention的softmax计算中，标准做法是先减去最大值以保证数值稳定性。但这个操作是否真的必要？

在大多数实际场景中，注意力分数 QK^T/√d_k 的值不会达到导致float32溢出的阈值（x > 88.7）。假设输入值都不超过溢出的阈值，不存在溢出问题，计算的结果精度会不会受到影响？

## 直觉

从直觉上来说，浮点数在越大的时候误差越大，累计计算中误差会越来越大。但是对于FlashAttention中Softmax计算，这一放大的误差反而会在除法中获得抵消，从而违反我们的直觉。


## 数学推导

### 假设：浮点运算的相对舍入误差模型

设浮点运算的相对舍入误差上界为 μ（machine epsilon），即：

fl(x op y) = (x op y)(1 + δ)，其中 |δ| ≤ μ

### 两元素情况的推导

suppose a > b，v₁，v₂为精确值。

设 eᵃ，eᵇ 为精确的 exp(a)，exp(b)，

fl(·) 表示浮点数表示

#### 方法 1：直接计算（不减最大值）
```
fl(eᵃ)v₁ + fl(eᵇ)v₂
───────────────────
fl(eᵃ) + fl(eᵇ)

= (eᵃ + eᵃ·μ)v₁ + (eᵇ + eᵇ·μ)·v₂
  ───────────────────────────────
  (eᵃ + eᵃ·μ) + (eᵇ + eᵇ·μ)

= eᵃv₁ + eᵇ·v₂ + μ(eᵃ·v₁ + eᵇ·v₂)
  ──────────────────────────────
  eᵃ + eᵇ + μ(eᵃ + eᵇ)
```

#### 方法 2：减去最大值后计算
```
fl(eᵃ⁻ᵃ)v₁ + fl(eᵇ⁻ᵃ)v₂
────────────────────────
fl(eᵃ⁻ᵃ) + fl(eᵇ⁻ᵃ)

= eᵃ⁻ᵃ(1+μ)v₁ + eᵇ⁻ᵃ(1+μ)v₂
  ─────────────────────────  化简得
  eᵃ⁻ᵃ(1+μ) + eᵇ⁻ᵃ(1+μ)

= eᵃ(1+μ)v₁ + eᵇ(1+μ)v₂
  ─────────────────────
  eᵃ + eᵇ + (eᵃ + eᵇ)μ

= eᵃv₁ + eᵇv₂ + μ(eᵃv₁ + eᵇv₂)
  ──────────────────────────────
  eᵃ + eᵇ + μ(eᵃ + eᵇ)
```

**两种方法的舍入误差相同！**

### 推广到 n 个元素

设 x₁, x₂, ..., xₙ 为 n 个输入，对应的 value 向量为 v₁, v₂, ..., vₙ，记 m = max(x₁, x₂, ..., xₙ)。

#### 方法 1：直接计算
```
Σᵢ fl(exp(xᵢ))·vᵢ
─────────────────
Σᵢ fl(exp(xᵢ))

≈ Σᵢ exp(xᵢ)(1+μ)·vᵢ
  ──────────────────
  Σᵢ exp(xᵢ)(1+μ)

= Σᵢ exp(xᵢ)·vᵢ + μ·Σᵢ exp(xᵢ)·vᵢ
  ─────────────────────────────
  Σᵢ exp(xᵢ) + μ·Σᵢ exp(xᵢ)
```

#### 方法 2：减去最大值
```
Σᵢ fl(exp(xᵢ-m))·vᵢ
───────────────────
Σᵢ fl(exp(xᵢ-m))

≈ Σᵢ exp(xᵢ-m)(1+μ)·vᵢ
  ────────────────────
  Σᵢ exp(xᵢ-m)(1+μ)

= Σᵢ exp(xᵢ)·vᵢ + μ·Σᵢ exp(xᵢ)·vᵢ
  ─────────────────────────────
  Σᵢ exp(xᵢ) + μ·Σᵢ exp(xᵢ)
```

**结论：n 个元素情况下，两种方法的一阶舍入误差仍然相同！**

## 数值稳定性的额外考虑

虽然单个 exp(x) 不会溢出（当 x < 88.7），但在求和过程中仍需注意：

### 累加溢出问题

对于序列长度为 n 的softmax计算：
```
sum = exp(x₁) + exp(x₂) + ... + exp(xₙ)
```

即使每个 exp(xᵢ) < 溢出阈值，当 n 很大时，累加和可能溢出。

### 解决方案：归一化技巧

在累加前对每个指数项进行缩放：
```
exp(xᵢ) / n
```

这样可以保证：
- **单项计算**：exp(xᵢ) 不溢出（通过 xᵢ < 88.7 保证）
- **累加过程**：Σ(exp(xᵢ)/n) 的上界降低了 n 倍，避免累加溢出
- **最终结果**：分子分母都除以 n，约去后结果不变

**关键性质**：这个归一化技巧对任意 n 都适用，因为：
```
Σᵢ (exp(xᵢ)/n)·vᵢ     Σᵢ exp(xᵢ)·vᵢ
─────────────────── = ───────────────
Σᵢ (exp(xᵢ)/n)        Σᵢ exp(xᵢ)
```

## 研究结论

1. **数值等价性（可推广性）**：在一阶舍入误差分析下，对于任意 n 个元素，减去最大值和不减去最大值的softmax计算产生相同量级的舍入误差 O(μ)。

2. **溢出阈值分析**：IEEE 754 float32格式下，exp(x)在x > 88.7时发生上溢。在实际的注意力机制中，QK^T/√d_k产生>88的值的概率极小。

3. **累加溢出防护（通用方案）**：通过预先除以序列长度 n，可以在保证单项不溢出的前提下，防止任意长度序列累加过程中的溢出。该方法对 n 的大小没有限制。

4. **实践意义**：在确保输入范围可控的情况下，结合归一化技巧的简化softmax实现在数值精度上与标准实现等价，且可减少计算开销（避免寻找最大值的 O(n) 遍历）。

5. **工程权衡**：Flash Attention采用减最大值的做法主要是出于：
   - 算法鲁棒性和可移植性
   - 防止训练初期参数未收敛时的数值异常
   - 避免下溢出（underflow）导致的精度损失
   - 处理极端输入分布的保守策略

**结论**：对于Flash Attention，在大多数实际场景中，不减去最大值的softmax实现是可行的。通过输入范围检查（xᵢ < 88.7）和归一化技巧（除以序列长度 n），可以在舍入误差量级上等价于标准实现，同时避免寻找最大值的计算开销。**此结论对任意序列长度 n 都成立。**

基于此结论，提出softmax新的算法。

## 异常驱动的Softmax伪代码

```
Algorithm: Exception-Driven Softmax (No Max Subtraction)

// 初始化状态
State:
  o = 0                    // 分子累加器
  s = 0                    // 分母累加器  
  current_max = 0         // 当前最大值
  recovery_mode = false    // 是否在恢复模式

Function ProcessChunk(x_chunk, v_chunk, n):
  if recovery_mode:
    // 恢复模式：使用当前最大值
    exp_x = exp(x_chunk - current_max) / n
  else:
    // 快速路径：直接计算（可能溢出！）
    exp_x = exp(x_chunk) / n
  
  try:
    o_update = sum(exp_x * v_chunk)
    s_update = sum(exp_x)
    
    o = o + o_update
    s = s + s_update
    
  catch OverflowException:
    if not recovery_mode:
      // 第一次溢出：进入恢复模式
      recovery_mode = true
      new_max = max(x_chunk)
      
      // 重新缩放之前的所有累加
      if s > 0:
        rescale_factor = exp(previous_implicit_max - new_max)
        o = o * rescale_factor
        s = s * rescale_factor
      
      // 重新处理当前chunk
      exp_x = exp(x_chunk - current_max) / n
      o = o + sum(exp_x * v_chunk)
      s = s + sum(exp_x)
    else:
      // 已经在恢复模式但再次溢出：更新最大值
      new_max = max(x_chunk, current_max)
      if new_max > current_max:
        // 重新缩放所有历史
        rescale_factor = exp(current_max - new_max)
        o = o * rescale_factor
        s = s * rescale_factor
        current_max = new_max
      
      // 用新最大值处理当前chunk
      exp_x = exp(x_chunk - current_max) / n
      o = o + sum(exp_x * v_chunk)
      s = s + sum(exp_x)

Function Finalize():
  return o / s
```

## 关键特性

1. **零检查开销**：正常路径完全没有条件判断
2. **异常驱动**：只在硬件检测到溢出时才处理
3. **精确恢复**：溢出时重新缩放历史状态，保持数学正确性
4. **状态维护**：`current_max` 跟踪全局缩放因子

## 工作流程

```
输入流: [x1, x2, x3, x4, ...]
状态:   [正常, 正常, 溢出!, 恢复模式, ...]

时间线:
t0: x1 → 直接计算exp(x1) ✓
t1: x2 → 直接计算exp(x2) ✓  
t2: x3 → 直接计算exp(x3) → 溢出! → 进入恢复模式
t3: x4 → 使用exp(x4 - max) ✓
...
```

## 附录 
### 代码验证不减最大值时的数值稳定性

```python
import numpy as np


test_cases = [
    # 基础测试：正常范围
    {
        'name': '正常范围',
        'a': np.array([2.0, 1.0, 0.0], dtype=np.float32),
        'v': np.array([1.0, 2.0, 3.0], dtype=np.float32)
    },
    
    # 边界测试：接近溢出
    {
        'name': '接近溢出边界',
        'a': np.array([87.0, 85.0, 83.0], dtype=np.float32),
        'v': np.array([1.5, 2.5, 3.5], dtype=np.float32)
    },
    
    
    # 数值稳定性测试
    {
        'name': '大值差异大',
        'a': np.array([50.0, 10.0, 1.0], dtype=np.float32),  # 差异很大
        'v': np.array([1.0, 2.0, 3.0], dtype=np.float32)
    },
    
    # 长序列测试
    {
        'name': '长序列',
        'a': np.array([80.0] * 100, dtype=np.float32),  # 很多大值
        'v': np.array([1.0] * 100, dtype=np.float32)
    },
    
    # 实际注意力分布
    {
        'name': '注意力分布',
        'a': np.array([10.0, 8.0, 5.0, 2.0, -1.0], dtype=np.float32),
        'v': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    }
]

def not_minus_max(a,v):
    # 方法1：不减最大值
    exp_a = np.exp(a)
    o1 = np.sum(exp_a * v/len(a)) 
    s1 = np.sum(exp_a) / len(a)
    result1 = o1 / s1
    return result1

def minus_max(a,v):
    # 方法2:减去最大值
    m = np.max(a)
    exp_a_stable = np.exp(a - m)
    o2 = np.sum(exp_a_stable * v) / len(a)
    s2 = np.sum(exp_a_stable) / len(a)
    result2 = o2 / s2
    return result2

def run_comparison(a, v, case_name):
    print(f"\n=== {case_name} ===")
    print(f"输入 a: {a}")
    print(f"最大值: {np.max(a):.1f}")
    
    try:
        # 方法1：不减最大值
        result1 = not_minus_max(a, v)
        method1_status = "成功"
    except:
        result1 = None
        method1_status = "溢出"
    
    # 方法2：减去最大值
    result2 = minus_max(a, v)
    
    # 真实值（float64参考）
    a64 = a.astype(np.float64)
    exp_a64 = np.exp(a64 - np.max(a64))
    weights = exp_a64 / np.sum(exp_a64)
    true_result = np.sum(weights * v.astype(np.float64))
    
    print(f"不减最大值: {result1} [{method1_status}]")
    print(f"减去最大值: {result2}")
    print(f"真实值:     {true_result}")
    
    if result1 is not None:
        error1 = abs(result1 - true_result)
        error2 = abs(result2 - true_result)
        print(f"方法1误差: {error1:.30e}")
        print(f"方法2误差: {error2:.30e}")

# 运行所有测试
for case in test_cases:
    run_comparison(case['a'], case['v'], case['name'])

```

Test Result:
```
=== 正常范围 ===
输入 a: [2. 1. 0.]
最大值: 2.0
不减最大值: 1.4247896671295166 [成功]
减去最大值: 1.424789547920227
真实值:     1.4247896173955585
方法1误差: 4.973395806295854981726733967662e-08
方法2误差: 6.947533148782270018273266032338e-08

=== 接近溢出边界 ===
输入 a: [87. 85. 83.]
最大值: 87.0
不减最大值: 1.6490628719329834 [成功]
减去最大值: 1.6490627527236938
真实值:     1.6490629077791317
方法1误差: 3.584614827190080177388153970242e-08
方法2误差: 1.550554378226820517738815397024e-07

=== 大值差异大 ===
输入 a: [50. 10.  1.]
最大值: 50.0
不减最大值: 1.0 [成功]
减去最大值: 1.0
真实值:     1.0
方法1误差: 0.000000000000000000000000000000e+00
方法2误差: 0.000000000000000000000000000000e+00

=== 长序列 ===
输入 a: [80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80.
 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80.
 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80.
 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80.
 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80. 80.
 80. 80. 80. 80. 80. 80. 80. 80. 80. 80.]
最大值: 80.0
不减最大值: 0.9999998211860657 [成功]
减去最大值: 1.0
真实值:     0.9999999999999999
方法1误差: 1.788139342151495725374843459576e-07
方法2误差: 1.110223024625156540423631668091e-16

=== 注意力分布 ===
输入 a: [10.  8.  5.  2. -1.]
最大值: 10.0
不减最大值: 1.1311984062194824 [成功]
减去最大值: 1.1311984062194824
真实值:     1.1311983895270497
方法1误差: 1.669243276936072106764186173677e-08
方法2误差: 1.669243276936072106764186173677e-08
```

可以看到，在大多数情况下，方法1的误差（不减最大值）反而更低。这可能得因于减少了b-a的误差。在极特殊的所有输入都相等的情况下，才会出现方法2，即传统方法的精度更高。