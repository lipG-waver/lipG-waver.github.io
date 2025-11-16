# The Evolution of Linear Attention: From Unbounded Accumulation to Fine-grained Gating

## Introduction

Linear Attention, as an efficient alternative to standard Softmax Attention, has undergone a fascinating evolutionary process in the field of sequence modeling. This article traces the complete development trajectory from Linear Attention in 2020 to Kimi Delta Attention in 2025, revealing the theoretical foundations and engineering wisdom behind it.

## I. The Logical Chain of Evolution

### Phase 1: Problem Identification (2020)

**Linear Attention** proposed an efficient attention mechanism based on cumulative state:

**Core Mechanism**:
$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^\top$$

**Optimization Objective**:
$$\mathcal{L}_t(\mathbf{S}) = -\langle\mathbf{S}^\top \mathbf{k}_t, \mathbf{v}_t\rangle$$

From the fast weights perspective, $\mathbf{S}_t$ serves as associative memory, storing transient mappings from keys to values.

**Core Problems**:
- ❌ Unbounded growth of accumulated state
- ❌ Inability to forget outdated information
- ❌ Memory interference in long sequences
- ❌ Unbounded optimization objective

### Phase 2: Introduction of Forgetting Mechanism (2023)

**DeltaNet** fundamentally changed the learning paradigm of linear attention by redefining the optimization objective.

**Key Insight**: Redefining the problem from correlation maximization to **reconstruction error minimization**:

$$\mathcal{L}_t(\mathbf{S}) = \frac{1}{2}\|\mathbf{S}^\top\mathbf{k}_t - \mathbf{v}_t\|^2$$

**Update Rule** (derived through gradient descent):
$$\mathbf{S}_t = (\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top)\mathbf{S}_{t-1} + \beta_t\mathbf{k}_t\mathbf{v}_t^\top$$

**Innovations**:
- ✅ Delta Rule: self-correcting associative memory
- ✅ Equivalent to generalized Householder transformation
- ✅ Rank-1 update structure supports efficient parallelization
- ✅ Bounded optimization objective

**Remaining Issues**:
Although introducing a structured correction mechanism, it still retains outdated associations indefinitely, lacking active forgetting.

### Phase 3: Scalar Gated Forgetting (2024)

**Gated DeltaNet (GDN)** introduced an explicit forgetting mechanism.

**Core Innovation**: Introducing scalar forget gate $\alpha_t \in [0,1]$:

$$\mathbf{S}_t = \alpha_t(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top)\mathbf{S}_{t-1} + \beta_t\mathbf{k}_t\mathbf{v}_t^\top$$

**Theoretical Interpretation**:
- $\alpha_t$ implements **weight decay**
- Similar to data-dependent $L_2$ regularization
- Provides a principled way to control memory lifespan

**Limitations**:
- Scalar gate applies **uniform decay** to the entire state matrix
- Cannot differentiate across different dimensions
- Lacks **fine-grained positional awareness**

### Phase 4: Fine-grained Diagonal Gating (2025)

**Kimi Delta Attention (KDA)** extends scalar gating to diagonal matrix.

**Core Innovation**: Diagonalized gating $\text{Diag}(\boldsymbol{\alpha}_t)$:

$$\mathbf{S}_t = \left(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top\right) \text{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}_{t-1} + \beta_t\mathbf{k}_t\mathbf{v}_t^\top$$

**Key Advantages**:
- ✅ **Fine-grained decay control**: independent forgetting rate per dimension
- ✅ **Positional awareness**: learnable positional encoding characteristics
- ✅ **Computational efficiency**: diagonal matrix maintains $O(d)$ complexity
- ✅ **Expressive power**: relaxes RoPE's orthogonality constraint

**Engineering Value**:
Although theoretically just a natural extension from scalar to vector, it finds the optimal balance (sweet spot) between parameter efficiency and expressive power.

## II. Summary of Core Evolution Logic

### Problem Level
```
Unbounded growth → Need for forgetting → Need for fine-grained forgetting
```

### Method Level
```
No forgetting (LA) → Structured forgetting (DN) → Scalar forgetting (GDN) → Diagonal forgetting (KDA)
```

### Theory Level
```
Gradient descent → Reconstruction loss → Weight decay → Learnable positional encoding
```

### Efficiency Level
```
Associative memory → Rank-1 update → Maintain parallelization → Diagonalization speedup
```

## III. Key Insight of Reconstruction Loss

### Essential Difference Between Two Loss Functions

#### Linear Attention: Correlation Maximization
$$\mathcal{L}_t(\mathbf{S}) = -\mathbf{k}_t^\top \mathbf{S} \mathbf{v}_t$$

**Characteristics**:
- Unbounded objective (can be infinitely large)
- Gradient direction: $\nabla_\mathbf{S} \mathcal{L}_t = -\mathbf{v}_t \mathbf{k}_t^\top$
- Update: $\mathbf{S}_t = \mathbf{S}_{t-1} + \eta \mathbf{k}_t \mathbf{v}_t^\top$
- Result: **Pure accumulation, never decreases**

#### DeltaNet: Reconstruction Error Minimization
$$\mathcal{L}_t(\mathbf{S}) = \frac{1}{2}\|\mathbf{S}^\top\mathbf{k}_t - \mathbf{v}_t\|^2$$

**Characteristics**:
- Bounded objective (minimum value is 0)
- Clear "correct answer": $\mathbf{S}^\top\mathbf{k}_t = \mathbf{v}_t$
- Gradient direction: $\nabla_\mathbf{S} \mathcal{L}_t = \mathbf{k}_t(\mathbf{S}^\top\mathbf{k}_t - \mathbf{v}_t)$
- Update: **Can both increase and decrease** elements of $\mathbf{S}$

### Self-correction Mechanism

Expanding DeltaNet's gradient descent update:

$$\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \mathbf{k}_t(\mathbf{S}_{t-1}^\top\mathbf{k}_t - \mathbf{v}_t)^\top$$

The meaning of the key term $-\beta_t \mathbf{k}_t\mathbf{k}_t^\top\mathbf{S}_{t-1}$:

- If $\mathbf{S}_{t-1}^\top\mathbf{k}_t > \mathbf{v}_t$ (prediction too large) → **decrease** weights in corresponding direction
- If $\mathbf{S}_{t-1}^\top\mathbf{k}_t < \mathbf{v}_t$ (prediction too small) → **increase** weights in corresponding direction

This implements a **bidirectional correction** mechanism, rather than unidirectional accumulation.

### Intuitive Analogy

**Linear Attention (Correlation Maximization)**:
> Like a "write-only" notebook, continuously adding new content without ever modifying old content. Result: the notebook gets thicker and thicker, information becomes increasingly messy.

**DeltaNet (Reconstruction Error Minimization)**:
> Like an "erasable" whiteboard, actively erasing old content that conflicts with current keys when writing new content. Result: maintains information consistency and finite capacity.

### Theoretical Guarantees

| Dimension | Linear Attention | DeltaNet |
|------|-----------------|----------|
| Objective Function | Unbounded (maximize correlation) | Bounded (minimize MSE) |
| Optimal Solution | Does not exist (can be infinitely large) | Exists and is explicit |
| Update Direction | Unidirectional accumulation | Bidirectional correction |
| Long-term Behavior | Divergence | Convergence (with gating) |
| Capacity Control | None | Implicit (through rank-1 projection) |

## IV. Historical Origins of Reconstruction Loss

DeltaNet's reconstruction loss is not a completely new invention, but a revival and innovative application of classical ideas.

### 1. Most Direct Source: Classical Delta Rule (1960)

**Widrow-Hoff Delta Rule** (the original Delta Rule):

$$\Delta \mathbf{w} = \eta (y - \hat{y}) \mathbf{x}$$

Where $\hat{y} = \mathbf{w}^\top \mathbf{x}$, optimization objective:

$$\mathcal{L} = \frac{1}{2}(y - \mathbf{w}^\top \mathbf{x})^2$$

**This is the prototype of reconstruction loss!**

**Comparison**:
```
Delta Rule:  w ← w - η(w^T x - y)x
DeltaNet:    S ← S - β(S^T k - v)k
```

Completely parallel structure! DeltaNet is named after this.

### 2. Associative Memory: Hopfield Network (1982)

**Hopfield Network**'s Hebbian learning rule:

$$\mathbf{W} = \sum_{\mu} \mathbf{v}^{(\mu)} (\mathbf{v}^{(\mu)})^\top$$

**Problem**: Limited capacity, severe memory interference

**DeltaNet's Contribution**: Transforming offline batch learning Hopfield networks into **online learning** associative memory.

### 3. Online Learning: LMS Algorithm (1960s)

**Least Mean Squares (LMS) Algorithm**:

$$\mathbf{w}_{t+1} = \mathbf{w}_t + \mu (d_t - \mathbf{w}_t^\top \mathbf{x}_t) \mathbf{x}_t$$

**This is completely isomorphic to DeltaNet's update rule!**

Linear Attention is equivalent to a degraded version without the correction term:
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \mu d_t \mathbf{x}_t$$

### 4. Recursive Least Squares (RLS)

**Recursive Least Squares** covariance matrix update:

$$\mathbf{P}_t = \mathbf{P}_{t-1} - \frac{\mathbf{P}_{t-1}\mathbf{x}_t\mathbf{x}_t^\top\mathbf{P}_{t-1}}{1 + \mathbf{x}_t^\top\mathbf{P}_{t-1}\mathbf{x}_t}$$

Has **structural similarity** with DeltaNet's $(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^\top)$, both are **rank-1 corrections**.

### 5. Neuroscience: Rescorla-Wagner Model (1972)

Learning rule for classical conditioning:

$$\Delta V = \alpha \beta (\lambda - V)$$

Where $(\lambda - V)$ is **prediction error** — error-driven mechanism in biological learning!

### Timeline of Idea Evolution

```
1960  Widrow-Hoff Delta Rule (single-layer perceptron)
  ↓
1972  Rescorla-Wagner (neuroscience)
  ↓
1982  Hopfield Network (associative memory)
  ↓
1986  Backpropagation (Delta Rule extension to multi-layer networks)
  ↓
2016  Modern Hopfield (energy perspective)
  ↓
2020  Linear Attention (unbounded accumulation)
  ↓
2023  DeltaNet (return to Delta Rule, introduce reconstruction loss)
  ↓
2024  Gated DeltaNet (scalar gating)
  ↓
2025  Kimi Delta Attention (diagonal gating)
```

### DeltaNet's True Innovation

**Not inventing reconstruction loss** (which is a classical idea), but:

1. Applying classical Delta Rule to modern Transformer architecture
2. Discovering compatibility between rank-1 updates and chunkwise parallelization
3. Implementing online learning in matrix-valued state space
4. Connecting **online learning** and **attention mechanism** domains

This is a successful **cross-domain knowledge transfer**:

```
Adaptive Signal Processing (LMS)
        ↓
Neural Network Training (Delta Rule)
        ↓
Sequence Modeling (Linear Attention)
```

### Why Didn't Linear Attention Use This Idea?

Possible reasons:

1. **Fast Weights Tradition**: Inherited Schmidhuber's fast weights idea, emphasizing accumulation
2. **Softmax Attention Analogy**: Attempting to mimic standard attention, which lacks explicit reconstruction objective
3. **Lack of Theoretical Analysis**: Not until DeltaNet was the optimization objective systematically analyzed
4. **Paradigm Inertia**: Transformer community focused more on architecture design than optimization theory

## V. Value of Engineering Optimization

### KDA's "Micro-innovation"

From a theoretical perspective, KDA's innovation is indeed limited:

```
GDN:  α_t * (I - β_t k_t k_t^T) S_{t-1}     // scalar
KDA:  Diag(α_t) * (I - β_t k_t k_t^T) S_{t-1}  // vector
```

Essentially:
- Natural extension from scalar → vector
- Direct generalization from uniform decay → non-uniform decay

### But Significant Engineering Value

Although theoretically "trivial", it finds the optimal balance in engineering:

**Trade-off Between Parameter Efficiency and Expressive Power**:
- Scalar: 1 parameter, **too weak**
- Full matrix: $d^2$ parameters, **too expensive**
- Diagonal matrix: $d$ parameters, **just right (sweet spot)**

**Computational Efficiency**:
- Diagonal matrix multiplication: $O(d)$
- Full matrix multiplication: $O(d^2)$
- Maintains memory access continuity
- Perfectly compatible with chunkwise parallelization

**Interpretability Enhancement**:
- Visualizable forgetting rate per dimension
- Capability to implicitly learn positional encoding
- Bypasses RoPE's orthogonality constraint

### Common Pattern in Academia

**True Breakthroughs** (few):
- Linear Attention (2020): core paradigm shift
- DeltaNet (2023): Delta Rule reinterpretation

**Incremental Optimizations** (many):
- GDN: add a scalar gate
- KDA: change scalar to vector
- Next paper possibly: low-rank matrix gating?

### Why Incremental Optimizations Still Matter?

1. **Engineering Practice Needs These Optimizations**
   - Theoretically "obvious" improvements may have significant practical effects
   - Expansion of hyperparameter space brings better adaptability

2. **Value of Combinatorial Innovation**
   - Adaptation with parallelization strategies
   - Integration with numerical stability optimizations
   - Overall system performance improvement

3. **Reality of Academic Publishing**
   - Need for continuous paper output
   - Incremental improvements are also contributions
   - Paving the way for subsequent breakthroughs

## VI. Summary of Core Points

### Evolution Logic

The evolution of linear attention embodies a progressive optimization process from **coarse-grained to fine-grained**, from **fixed mechanisms to learnable mechanisms**, from **global control to dimension-specific control**:

```
Problem Identification → Theory Reconstruction → Explicit Forgetting → Fine-grained Control
       LA           →         DN          →        GDN         →        KDA
```

### Key Insight

**Reconstruction loss** is the core turning point of the entire evolution chain:

- From unbounded optimization to bounded optimization
- From unidirectional accumulation to bidirectional correction
- From divergent behavior to convergent properties
- From engineering trick to theoretically analyzable

### Historical Enlightenment

DeltaNet's value lies in **revival of classical wisdom in new domains**:

- Delta Rule from the 1960s
- Revitalized in the Transformer era of 2023
- Connecting online learning and attention mechanism domains

Just as ResNet revived highway networks, DeltaNet revived Delta Rule.

### Engineering Wisdom

KDA reminds us:

- Theoretical innovation is certainly important
- Engineering optimization is equally valuable
- Finding the optimal balance of **parameters-performance-efficiency**
- Accumulated incremental improvements can bring significant differences

## Conclusion

The evolutionary journey from Linear Attention to Kimi Delta Attention demonstrates two important aspects of machine learning research:

1. **Theoretical Innovation**: DeltaNet fundamentally changed the optimization paradigm by introducing reconstruction loss
2. **Engineering Optimization**: GDN and KDA achieved practical improvements through incremental refinement of gating mechanisms

This evolutionary process also reminds us:

- New breakthroughs often come from **re-examining basic assumptions** (Why maximize correlation?)
- Classical theories still have vitality in new scenarios (Revival of Delta Rule)
- Theoretically "obvious" extensions may have tremendous engineering value (scalar→diagonal)

Future linear attention may continue to evolve, but the two core ideas of reconstruction loss and fine-grained control have already laid a solid foundation for this field.

---

**References**

- Linear Attention (2020): Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- DeltaNet (2023): "DeltaNet: Online Gradient Descent on Reconstruction Loss"
- Gated DeltaNet (2024): "Gated DeltaNet as Weight Decay"
- Kimi Delta Attention (2025): "Kimi Delta Attention: Improving Delta Rule with Fine-grained Gating"
- Widrow-Hoff (1960): "Adaptive Switching Circuits"
- Hopfield (1982): "Neural networks and physical systems with emergent collective computational abilities"