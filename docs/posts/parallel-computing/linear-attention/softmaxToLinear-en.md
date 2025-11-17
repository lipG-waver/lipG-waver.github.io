# From Standard Attention to Linear Attention


---

## 0. The Fundamental Question: **What Are We Trying to Compute?**

We receive a time-ordered sequence of key–value pairs:

$$
(k_1, v_1), (k_2, v_2), \dots, (k_t, v_t)
$$

and at time (t), a query vector:

$$
q_t.
$$

We want a function:

$$
y_t = f(q_t;{k_i, v_i}_{i\le t})
$$

such that:

1. **Content-based retrieval**: keys similar to (q_t) should give their values more weight.
2. **Linearity in values**: the output should be a linear combination of (v_i).
3. **Online computability**: ideally we should not revisit all past tokens for each new step.

Self-attention is one way to implement this.

---

## 1. Classical Softmax Attention

Similarity scores:

$$
s_{ti} = q_t^\top k_i,\quad i\le t.
$$

Softmax weights:

$$
\alpha_{ti}
= \frac{\exp(q_t^\top k_i)}{\sum_{j\le t} \exp(q_t^\top k_j)}.
$$

Output:

$$
y_t = \sum_{i\le t} \alpha_{ti} v_i.
$$

Combined:

$$
\boxed{
y_t
===

\frac{
\sum_{i\le t} \exp(q_t^\top k_i), v_i
}{
\sum_{j\le t} \exp(q_t^\top k_j)
}}
\tag{1}
$$

---

## 2. Softmax as “Numerator / Denominator”

Define:

**Numerator:**

$$
N_t(q_t) = \sum_{i\le t} \exp(q_t^\top k_i), v_i
$$

**Denominator:**

$$
D_t(q_t) = \sum_{j\le t} \exp(q_t^\top k_j)
$$

Then:

$$
\boxed{
y_t = \frac{N_t(q_t)}{D_t(q_t)}
}
\tag{2}
$$

This separation becomes crucial for linear attention.

---

## 3. Kernel Trick: Approximating exp Inner Products with Feature Maps

*(Note: formula not placed in the heading to ensure correct rendering.)*

The key idea of linear attention is to approximate the exponential kernel using a feature map (\phi):

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k).
$$

This converts a nonlinear exponential into a linear inner product, enabling efficient accumulation.

---

## 4. Substitute exp with the Kernel Approximation

### 4.1 Numerator

Replace (\exp) with the kernel approximation:

$$
N_t(q_t)
\approx
\sum_{i\le t}
\phi(q_t)^\top \phi(k_i), v_i.
$$

Use linearity of the inner product:

$$
N_t(q_t) =
\phi(q_t)^\top
\left(
\sum_{i\le t} \phi(k_i), v_i^\top
\right).
$$

Define the numerator state:

$$
\boxed{
S_{v,t} := \sum_{i\le t} \phi(k_i), v_i^\top
}
\tag{4}
$$

Thus:

$$
\boxed{
N_t(q_t)\approx \phi(q_t)^\top S_{v,t}
}
\tag{3}
$$

---

### 4.2 Denominator

Likewise:

$$
D_t(q_t)
\approx
\phi(q_t)^\top
\left(
\sum_{j\le t} \phi(k_j)
\right).
$$

Define the denominator state:

$$
\boxed{
S_{k,t} := \sum_{j\le t} \phi(k_j)
}
\tag{6}
$$

Thus:

$$
\boxed{
D_t(q_t)\approx \phi(q_t)^\top S_{k,t}
}
\tag{5}
$$

---

## 5. The Linear Attention Formula

Plugging the approximated numerator and denominator into the Softmax structure:

$$
\boxed{
y_t \approx
\frac{
\phi(q_t)^\top S_{v,t}
}{
\phi(q_t)^\top S_{k,t}
}
}
\tag{7}
$$

Here:

* (S_{v,t}): “numerator memory” (value-weighted)
* (S_{k,t}): “denominator memory” (key-only)

This is the core structure of kernelized linear attention.

---

## 6. Online Updates

Both states update incrementally:

$$
S_{v,t} = S_{v,t-1} + \phi(k_t)v_t^\top,
$$

$$
S_{k,t} = S_{k,t-1} + \phi(k_t).
$$

Output requires only:

$$
y_t \approx \frac{\phi(q_t)^\top S_{v,t}}{\phi(q_t)^\top S_{k,t}}.
$$

Thus computational complexity drops from **O(t²)** to **O(t)**.

---

## 7. Interpreting the One-State Version: Only (S_t), No Normalization

Some simplified papers use only a single state:

State update:

$$
S_t = S_{t-1} + k_t v_t^\top
$$

Read:

$$
\hat v_t = S_t^\top k_t
$$

Loss:

$$
\mathcal{L}_t(S) = -\langle S_t^\top k_t,\ v_t\rangle.
$$

This corresponds to:

* (q_t=k_t)
* (\phi(x)=x)
* **no denominator**, i.e., no Softmax normalization.

Thus:

$$
S_t^\top k_t
============

\sum_{i\le t} (k_i^\top k_t), v_i.
\tag{8}
$$

Weights become:

$$
\tilde{\alpha}_{ti}=k_t^\top k_i
$$

instead of the softmax-normalized:

$$
\alpha_{ti}\propto \exp(q_t^\top k_i).
$$

This is essentially a **“numerator-only linear attention”**.

---

## 8. Full Logical Chain Summary

1. Softmax attention = **numerator / denominator**.
2. Exponential kernel approximated via feature map:
   (\exp(q^\top k)\approx \phi(q)^\top \phi(k)).
3. This makes both numerator and denominator **linear in accumulated history**.
4. Linear attention arises from maintaining:

   * (S_{v,t} = \sum \phi(k_i)v_i^\top)
   * (S_{k,t} = \sum \phi(k_i))
5. Output is:
   $$
   y_t\approx \frac{\phi(q_t)^\top S_{v,t}}{\phi(q_t)^\top S_{k,t}}.
   $$
6. Complexity becomes **O(t)**.
7. Single-state versions correspond to “unnormalized linear attention.”

