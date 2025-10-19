# 从CUDA到AscendC：编程视角的转变

> 理解昇腾AI处理器的编程范式

## CUDA：线程视角编程

### CUDA Kernel 示例

```c
__global__ void vectorAdd(const float *A, const float *B, 
                          float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}
```

### CUDA层次结构解析

**核心公式：** `i = blockDim.x × blockIdx.x + threadIdx.x`

这是一个索引方式，用于计算线程在全局中的唯一索引。什么是线程？什么是块？不妨比喻为小弟和大哥。出现一个计算任务以后，先想到的是把任务布置给小弟（线程）。然而我们名单上管理的都是大哥（块），每个大哥手下都有小弟，所以我们计算哪个大哥的哪个小弟做哪个任务，就是上面这一行式子。

- **Block（块，也是大哥）**：管理一组线程的单元
- **Thread（线程，也是小弟）**：实际执行任务的最小单元
- **blockDim**：每个Block中的线程数（每个大哥带多少小弟）
- **blockIdx**：Block的索引（第几个大哥）
- **threadIdx**：线程在Block中的索引（小弟在大哥名册上的序号）

**示例：** 假设有10个Block，每个Block有10个线程（blockDim.x=10）  
当 `i=66` 时 → blockIdx.x=6（第7个Block），threadIdx.x=6（第7个线程）

> **注意：** 程序员按线程视角编写，但实际执行按Warp（线程束，32个线程）调度。这是后话，目前从线程视角理解即可。

---

## AscendC：数据块视角编程

### 设计哲学

写这些内容的时候，突然想到了华为的一个管理理念：华为很少开除最底层员工，一般优化的都是领导。因为华为认为要干不好，都是领导的锅。没有一个好领导，再好的员工都发挥不出威力。

**昇腾AscendC是按照数据块视角编写，从领导的编程视角来看的。**

这是什么含义呢？

Ascend芯片内部对于几个数据的同时操作（同一时钟周期）提供了统一的简单算子。比如，加法、减法、乘法、除法这些。内部如何实现的，不得而知。

只要你能将这些你希望处理数据按照昇腾需要的形状填入，那昇腾就能为你做高效计算。相比之下，英伟达的Cuda编程就不需要你有什么通用的形状，因为Cuda处理的往往是通用计算，而非形状在大多时候都是可预测的神经网络计算。

让我们通过实例来进行理解。

### AscendC 初始化示例

```c
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    xGm.SetGlobalBuffer((__gm__ half *)x + 
          BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    
    pipe.InitBuffer(inQueueX, BUFFER_NUM, 
          TILE_LENGTH * sizeof(half));
}
```

**数据搬运流程：**

- 一开始的时候，数据在内存条(DDR)上。经过了SetGlobalBuffer这一句，数据被搬运到了NPU内部的全局缓存上。
- 还是用大哥和小弟的思路进行理解。每位大哥会被分配到要处理一定的数据，也就是BLOCK_LENGTH的数据。每个人从哪里开始处理数据呢？就是初始位置加上每个大哥处理的位置×这个大哥的序号。比如初始位置是100，每个大哥处理100个，这个大哥的序号是8，那第0个大哥从100处理到199，8号的处理序号就是900-999。当然了，现在这一步是声明位置，还没涉及到直接的搬运处理。

**关键参数：**

- **GetBlockIdx()**：告诉你这是第几个Core（第几位大哥）
- **BLOCK_LENGTH**：每个Core处理多少数据
- 在pipe.InitBuffer这一行为什么又出现了TILE_LENGTH呢？每个大哥也不可能一下子处理所有数据，具体原因在下面会写到。如果先接受这个观念，那你就会接受要先分块。每一块再加载到队列中，这时候就有直接搬运了。

**数据分配公式：**

```
Core数量 × 每个Core处理的数据量 = 总数据量
```

示例：USE_CORE_NUM = 8, BLOCK_LENGTH = 2048 → 总数据量 = 16,384

---

## 为什么要分批处理？流水线优化

### ⚠️ 核心问题：能否一次性处理2048个数据？

**答案：硬件上可以，但效率极低！**

### 原因1：Local Memory装不下

- 昇腾Local Memory只有256-512KB
- 2048个half数据 ≈ 4KB
- 双缓冲：4KB × 3(x,y,z) × 2 = 24KB（还能接受）
- 但是很多复杂算子（如卷积）可能需要几百KB → 装不下
- 一次处理一定量的数据效率比较高

### 原因2：流水线效率低

**一次性处理的时间线：**

- 前160μs：搬运工干活，AI Core **空转**
- 中间80μs：AI Core干活，搬运工**空转**
- 后160μs：搬运工干活，AI Core **空转**

**总耗时 = 400μs**

### ✓ 优化方案：分批+流水线

**分成8批，每批256个（TILE_LENGTH=256），配合双缓冲：**

- 搬入引擎、AI Core、搬出引擎可以"三管齐下"
- 搬入第2批时，第1批在计算
- 计算第2批时，第1批在搬出、第3批在搬入

**流水线时间线：**

```
批次1: [搬入] → [计算] → [搬出]
批次2:       [搬入] → [计算] → [搬出]
批次3:             [搬入] → [计算] → [搬出]
...
```

**🚀 总耗时 ≈ 100μs（省了75%的时间！）**

> **关键点：** 搬入/计算/搬出是三个独立的硬件单元，互不影响！所以分成小批次能让三个单元都"满负荷运转"

### 为什么是双缓冲而非三缓冲？

由于只有一个AI Core，计算步骤注定只能串行。所以只需把操作分为两类：**加载（搬入/搬出）** 与 **计算**

**→ TILE_NUM=8、BUFFER_NUM=2 是最优配置**

要理解BUFFER_NUM = 2，是因为只有一个计算单元（我们是给每个计算单元进行编程，所以每个计算单元中只有一个），所以BUFFER_NUM 不可能是3，只能是2。

---

## 缓冲区管理机制

### InitBuffer

```c
pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
```

生成BUFFER_NUM个队列（2个），每次长度为TILE_LENGTH，供数据反复填入移出

### 队列容量限制

`inQueueX` 这个队列的容量是由 `BUFFER_NUM` 决定的。在这里 `BUFFER_NUM = 2`，意味着这个队列**最多只能同时容纳2个数据块**。

**双缓冲工作机制：**

- Buffer 0：正在被AI Core计算
- Buffer 1：正在被搬入数据

**流程控制：**

- 当队列已满（2个buffer都被占用）时，`EnQue` 操作会**阻塞等待**
- 只有当 `DeQue` 取出一个数据后，才能继续 `EnQue` 新数据
- 这样确保了生产者（CopyIn）和消费者（Compute）的同步

**处理总量：** Process需完成 `TILE_NUM × BUFFER_NUM` 次操作  
2个buffer × 8个tile = 16块数据

### 数据处理流程

**1. 分配本地张量**

```c
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
```

**2. 数据拷贝**

```c
DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
```

**3. 入队**

```c
inQueueX.EnQue(xLocal);
```

*CopyIn函数结束，xLocal销毁*

**4. 计算中出队**

```c
xLocal = inQueueX.DeQue();
```

*在Compute中使用，获取新的xLocal*

**5. 释放张量**

```c
inQueueX.FreeTensor(xLocal);
```

*Add操作完成后及时释放*

### 为什么先入队再在Compute中出队？

因为CopyIn和Compute中的xLocal不是同一个实例！CopyIn中入队后xLocal销毁，Compute中DeQue获得新的xLocal。队列内容在整个函数周期都存在，计算后需要释放资源，并将输出结果Z入队。

---

## 核心要点总结

- **CUDA范式：** 线程视角，程序员管理每个线程的执行逻辑
- **AscendC范式：** 数据块视角，从整体数据流的角度组织计算
- **流水线优化：** 分批处理+双缓冲，让硬件单元满负荷运转
- **队列管理：** 精细的入队/出队/释放机制确保资源高效利用

---

*理解编程范式的转变，掌握AI芯片编程的核心思想*