# Rethinking Softmax: Why Subtracting the Maximum Doesn't Reduce Numerical Error

## Key Finding

**Contrary to conventional wisdom, subtracting the maximum value in softmax computation does not reduce first-order rounding error.** 

This article proves mathematically that both methods—with and without max subtraction—produce identical first-order error of O(μ), where μ is the machine epsilon. The traditional justification that "subtracting max improves precision" is a misconception; the actual benefit is solely overflow prevention.

---

## The Conventional Belief

In FlashAttention and virtually all softmax implementations, the standard practice is:

$$
\text{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max_j x_j
$$

When asked why, most practitioners—including LLMs and experienced researchers—cite two reasons:
1. Prevents overflow when $x_i > 88.7$ (float32 limit)
2. **Reduces numerical error** ← *This is incorrect*

The first reason is valid. The second is not.

---

## Mathematical Proof

### Setup

Let the relative rounding error bound be μ (machine epsilon):

$$
\text{fl}(x \circ y) = (x \circ y)(1 + \delta), \quad |\delta| \le \mu
$$

### Two-Element Case

Consider inputs $a > b$ with corresponding values $v_1, v_2$.

**Method 1: Direct computation (no max subtraction)**

$$
\frac{\text{fl}(e^a)v_1 + \text{fl}(e^b)v_2}{\text{fl}(e^a) + \text{fl}(e^b)} = \frac{e^a(1+\mu)v_1 + e^b(1+\mu)v_2}{e^a(1+\mu) + e^b(1+\mu)}
$$

The $(1+\mu)$ factor appears in both numerator and denominator, yielding:

$$
= \frac{e^a v_1 + e^b v_2}{e^a + e^b} \cdot \frac{1+\mu}{1+\mu} + O(\mu^2)
$$

**Method 2: With max subtraction**

After subtracting $a$:

$$
\frac{\text{fl}(e^0)v_1 + \text{fl}(e^{b-a})v_2}{\text{fl}(e^0) + \text{fl}(e^{b-a})}
$$

Following the same expansion yields the identical expression to first order in μ.

### General Case (n Elements)

For any sequence $x_1, \ldots, x_n$:

$$
\frac{\sum_i \text{fl}(\exp(x_i)) v_i}{\sum_i \text{fl}(\exp(x_i))} \approx \frac{\sum_i e^{x_i}(1+\mu) v_i}{\sum_i e^{x_i}(1+\mu)} = \frac{\sum_i e^{x_i} v_i}{\sum_i e^{x_i}} + O(\mu)
$$

The same cancellation occurs regardless of whether we subtract the maximum.

**Conclusion: Both methods produce identical first-order rounding error O(μ).**

---

## Why the Cancellation Happens

The key insight is that softmax is a **ratio**. Any multiplicative error $(1+\mu)$ in the exponentials affects both numerator and denominator equally, causing the errors to cancel during division.

This is analogous to why $\frac{2.001}{4.002} \approx \frac{2}{4}$ even though both values have absolute errors—the relative structure is preserved.

---

## Experimental Verification

### Test Cases

```python
test_cases = [
    {'name': 'Normal Range',           'a': [2.0, 1.0, 0.0]},
    {'name': 'Near Overflow (87)',     'a': [87.0, 85.0, 83.0]},
    {'name': 'Large Difference',       'a': [50.0, 10.0, 1.0]},
    {'name': 'Attention Distribution', 'a': [10.0, 8.0, 5.0, 2.0, -1.0]},
]
```

### Results

| Test Case | Error (No Max) | Error (With Max) |
|-----------|----------------|------------------|
| Normal Range | 4.97e-08 | 6.95e-08 |
| Near Overflow | 3.58e-08 | 1.55e-07 |
| Large Difference | 0.00e+00 | 0.00e+00 |
| Attention Distribution | 1.67e-08 | 1.67e-08 |

**Observation**: In most cases, the method without max subtraction actually produces *lower* error, likely because it avoids the additional rounding error from the subtraction operation $(x_i - m)$.

### Edge Case

When all inputs are equal (e.g., all 80.0), the max-subtraction method shows better precision (error: 1.11e-16 vs 1.79e-07). This is expected: when $m = x_i$ for all $i$, we compute $e^0 = 1$ exactly, eliminating exponential error entirely.

---

## Practical Implications

### The Real Purpose of Max Subtraction

Max subtraction serves **one purpose only**: preventing overflow when $x_i > 88.7$.

For typical attention scores, values rarely approach this threshold. The subtraction operation adds O(n) computational overhead to find the maximum—overhead that provides no precision benefit.

### Alternative: Sequence-Length Normalization

A simpler approach to prevent overflow:

$$
\text{softmax}(x_i) = \frac{\exp(x_i)/n}{\sum_j \exp(x_j)/n}
$$

Dividing by sequence length $n$ ensures:
- Individual exponentials stay bounded
- Sum accumulation doesn't overflow
- Result is mathematically identical
- No O(n) max-finding overhead

This works because the $1/n$ factors cancel in the ratio.

---

## Why FlashAttention Still Uses Max Subtraction

Despite the above analysis, max subtraction remains standard practice for valid engineering reasons:

1. **Robustness**: Handles extreme outliers during early training
2. **Underflow prevention**: When inputs are very negative, $e^{x_i}$ approaches zero; subtracting max keeps at least one term at $e^0 = 1$
3. **Defensive design**: Works correctly even if assumptions are violated
4. **Historical inertia**: "Everyone does it this way"

These are legitimate reasons for production code. But they're different from claiming it improves numerical precision—it doesn't.

---

## Conclusion

The belief that subtracting the maximum reduces numerical error in softmax is a widespread misconception. Mathematically, the first-order rounding error is identical with or without the subtraction.

Max subtraction is useful for preventing overflow, but alternative methods (like sequence-length normalization) can achieve the same goal. For applications where the input range is bounded and overflow is not a concern, omitting the max subtraction saves computation without sacrificing precision.

---

## Appendix: Verification Code

```python
import numpy as np

def softmax_no_max(a, v):
    """Method 1: Direct computation without max subtraction"""
    exp_a = np.exp(a)
    return np.sum(exp_a * v) / np.sum(exp_a)

def softmax_with_max(a, v):
    """Method 2: Standard computation with max subtraction"""
    m = np.max(a)
    exp_a = np.exp(a - m)
    return np.sum(exp_a * v) / np.sum(exp_a)

def true_value(a, v):
    """Ground truth using float64"""
    a64 = a.astype(np.float64)
    exp_a = np.exp(a64 - np.max(a64))
    weights = exp_a / np.sum(exp_a)
    return np.sum(weights * v.astype(np.float64))

# Example test
a = np.array([87.0, 85.0, 83.0], dtype=np.float32)
v = np.array([1.5, 2.5, 3.5], dtype=np.float32)

result_no_max = softmax_no_max(a, v)
result_with_max = softmax_with_max(a, v)
truth = true_value(a, v)

print(f"No max subtraction:   {result_no_max:.10f}")
print(f"With max subtraction: {result_with_max:.10f}")
print(f"True value:           {truth:.10f}")
print(f"Error (no max):       {abs(result_no_max - truth):.2e}")
print(f"Error (with max):     {abs(result_with_max - truth):.2e}")
```

Output:
```
No max subtraction:   1.6490628719
With max subtraction: 1.6490627527
True value:           1.6490629078
Error (no max):       3.58e-08
Error (with max):     1.55e-07
```