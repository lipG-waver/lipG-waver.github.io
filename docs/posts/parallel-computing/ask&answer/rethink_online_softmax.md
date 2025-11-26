# **Rethinking FlashAttention on Huawei Ascend NPUs: Why GPU-Optimized Designs Don’t Directly Transfer**

FlashAttention has become the de facto standard for efficient attention computation on modern GPUs. Its design—especially the **Online Softmax algorithm**—beautifully matches the architectural strengths of NVIDIA hardware: large register files, fast shared memory, and cheap warp-level reductions.

But when I began porting FlashAttention-style kernels onto the **Huawei Ascend NPU**, something felt misaligned. The runtime profile looked different. Cube kernels idled. L2 traffic spiked. The performance curve no longer resembled the familiar GPU behavior.

That tension raised a question:

> **Does GPU-optimal FlashAttention remain optimal on NPUs?
> Or does the hardware difference fundamentally change what "optimal" means?**

This blog is my attempt to answer that question—not by trial-and-error benchmarking, but through **first-principles theoretical analysis** of the Ascend hardware.

What I found is this:

> **Online Softmax is not universally optimal.
> Its design depends on NVIDIA GPU assumptions that do not hold on Huawei NPUs.**

And once we account for how the Ascend architecture actually works, a different solution emerges.

---

# 1. The Core Architectural Mismatch

The key insight is simple:

> **On NVIDIA GPUs, almost all intermediate values stay on-chip.
> On Huawei Ascend NPUs, intermediate values must round-trip through L2.**

Let’s look at the essential hardware pathways.

---

## 1.1 How NVIDIA GPUs Execute FlashAttention

NVIDIA GPUs are built around:

* large register files (tens of KB per warp)
* warp-level shuffle instructions
* extremely fast shared memory / L1
* very low-latency L2

FlashAttention exploits all of this:

* Loads a block of Q and K into registers
* Computes partial QKᵀ locally
* Applies Online Softmax inside registers
* Streams through V with minimal global memory traffic

**Data hardly ever touches L2**, and certainly does not need to be written repeatedly.

This is why block-wise Online Softmax is optimal.

---

## 1.2 How Huawei Ascend NPUs Work

Ascend NPUs have a very different micro-architecture:

* **Cube Kernel**: high-throughput matrix multiplication unit
* **Vec Kernel**: vector unit for Softmax, scaling, activation
* **L2 buffer is the only communication channel between Cube and Vec**

The critical fact:

> **Every intermediate produced by Cube must be written to L2 before Vec can read it.
> Every Vec output must be written back to L2 before Cube can consume it.**

No shared register file.
No warp shuffle.
No SM local communication.

Just L2.

And on Ascend, L2 is:

* wide, but
* **latency-sensitive**
* and all writes incur noticeable **fixed per-operation overhead**

This alone is enough to change the optimal algorithm design.

---

# 2. Why Online Softmax Becomes Expensive on Ascend

On GPUs, Online Softmax is a win because:

* it saves DRAM bandwidth
* it controls numerical stability
* it avoids storing full rows

But on Ascend:

> **Online Softmax multiplies L2 write frequency—
> and frequent L2 writes are exactly what the hardware is slow at.**

Let’s compare two approaches for a block of 16 attention rows (my unit of analysis):

---

## 2.1 Full-Row (Non-Online) Softmax

Compute the full 16×8192 QKᵀ row at once:

* Write to L2: **3 times total**
* Synchronize Cube↔Vec: **2 times**
* Run Softmax once per row
* Multiply with V

Total L2 writes:
**≈ 3**

---

## 2.2 Online Softmax (FlashAttention style)

Split 8192 into 16 chunks of 512:

* Each chunk writes QKᵀ to L2
* Applies local Softmax using Vec
* Multiplies with V
* Rescales previously accumulated partial result
* Writes updated results to L2

Total L2 writes:
**≈ 64**

Total Cube↔Vec synchronizations:
**≈ 32**

---

## 2.3 Why this matters: the theory

For Ascend, L2 write time is well modeled as:

[
T_{\text{write}}
= N_{\text{write}} \cdot L_{\text{fixed}}

* \frac{\text{Data}}{B_{L2}}
  ]

If fixed latency ( L_{\text{fixed}} ) is non-trivial (and on Ascend it is), then:

* Full-Row Softmax:
  [
  T \propto 3 L_{\text{fixed}}
  ]

* Online Softmax:
  [
  T \propto 64 L_{\text{fixed}}
  ]

Thus even if bandwidth is huge (500 GB/s),

**Online Softmax can be 4× to 10× slower purely due to fixed latency**.

This is not a software bug.
This is the predictable outcome of the hardware communication model.

---

# 3. A Theoretical Re-evaluation of “Optimality”

Given these constraints, I conducted a systematic theoretical analysis:

### ✔ Memory Model

Evaluating L2 read/write behavior and fixed-latency impact.

### ✔ Kernel Pipeline Model

Analyzing the Cube–Vec scheduling timeline.

### ✔ Block-size Tradeoff Model

Understanding how Online Softmax scales with block granularity.

### ✔ Numerical Stability Considerations

Ensuring no loss of correctness when removing Online updates.

From this, a clear picture emerged:

> **On Ascend, the bottleneck is not arithmetic throughput.
> The bottleneck is L2 synchronization frequency.**

So an algorithm that:

* minimizes L2 writes
* minimizes Cube↔Vec synchronization
* minimizes block fragmentation
  is likely to outperform the GPU-optimal design.

That algorithm is:

> **Full-Row Softmax (or large-block Softmax).**

---

# 4. A Better Strategy for Ascend: Large-Block Softmax

Based on the theoretical modeling, I propose the following:

## **Instead of 512-sized blocks (GPU style), use much larger blocks on Ascend.**

For example:

| Block Size          | Rescale Ops | Synchronizations | L2 Peak Usage |
| ------------------- | ----------- | ---------------- | ------------- |
| 512                 | 15          | 32               | 256 KB        |
| 1024                | 7           | 16               | 512 KB        |
| 2048                | 3           | 8                | 1 MB          |
| **4096**            | **1**       | **4**            | **2 MB**      |
| **8192 (full row)** | **0**       | **2**            | **4 MB**      |

The result:

* **Fewer L2 writes**
* **Fewer synchronizations**
* **Less rescaling overhead**
* **Higher Cube utilization**

And Ascend has **192 MB of L2**, so even 4 MB per row-block is entirely feasible.

Thus:

> **The full-row Softmax—which is suboptimal on GPUs—becomes the optimal strategy on NPUs.**

---

# 5. What This Means for Real Attention Kernels

This theoretical analysis leads to a practical conclusion:

---

## 🎯 **FlashAttention’s Online Softmax is not universally optimal.**

It is GPU-optimal, but not hardware-agnostic.

On Huawei Ascend NPUs:

* the memory hierarchy is different
* Cube/Vec communication is L2-bounded
* L2 has fixed-latency writes
* synchronizations are costly
* memory is not the limiting factor

Therefore:

> **A redesigned FlashAttention with large-block or full-row Softmax
> yields higher performance than the standard Online Softmax.**

---

# 6. Why This Matters (For Real-World Systems)

This is not just a niche optimization.
It’s a broader systems lesson:

### **Deep learning kernels must be co-designed with hardware.**

Algorithms optimized for NVIDIA GPUs cannot be assumed to be optimal for:

* NPUs
* TPUs
* custom ASICs
* edge processors
* FPGA-like architectures

The gap between theory and performance is often a *memory-system phenomenon*, not an arithmetic one.

And bridging that gap—
understanding it, modeling it, optimizing it—
is exactly what I believe “real systems research” is about.

---

# 7. Closing Thoughts

This project started as a practical engineering task:
**port FlashAttention to Huawei Ascend**.

But along the way, it turned into something deeper:

* a study of architectural differences
* a theoretical performance model
* a rethinking of attention algorithm design
* and ultimately, a concrete optimization strategy tailored to NPUs

It taught me that:

> **The fastest algorithm is always hardware-dependent.
> Understanding the hardware enables designing fundamentally better algorithms.**

This is the kind of problem I hope to keep pursuing in graduate school:
the intersection of **algorithms, hardware systems, and real performance engineering**—
where theory meets silicon, and where a new perspective can unlock real gains.
