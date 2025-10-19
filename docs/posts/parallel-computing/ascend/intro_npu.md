# 回到AscendC Kernel函数一侧
## AscendC 初始化示例
### 定义
```c
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
```
回归一下我们一开始的函数签名。由于我们刚刚已经清楚了其在CPU上的整个过程，接下来我们就要来到CPU上讲解这一过程。
第一点，这个GM_ADDR是在NPU上的位置。因为已经有过一个Memcpy的操作了。
我们清晰知道，要把x,y地址上的东西进行一个加法，然后写到z上面去。
于是我们写到：
```
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z){
      KernelAdd op;
      op.Init(x,y,z);
      op.Process();
}
```
很显然，这里定义了一个类叫做KernelAdd.
作为一个类，KernelAdd会有一个本体`KernelAdd`，两个对应的方法分别是`Init`和`Process`.
```
class KernelAdd {
public:
    __aicore__ inline KernelAdd(){}
    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z){}
    // 核心处理函数，实现算子逻辑，调用私有成员函数CopyIn、Compute、CopyOut完成矢量算子的三级流水操作
    __aicore__ inline void Process(){}
```
这个Process比较值得玩味。一方面，数据已经被我们拷入了NPU的大存储池，但还没有拷入小存储池子。在小存储池的操作才涉及到真正的拿来就用的计算。在小存储池，以及对应的ALU单元完成运算以后，还要搬回大存储池。

也就是说，我们干的事情和我们在CPU一侧干的十分类似。只是一级一级封装。在CPU的时候，我们看到的是数据进入了NPU，然后调用了一个黑箱函数add_custom_do, 数据从NPU里出来了。

现在还是同样的道理。依旧是我们看到数据从GM_ADDR进到了LM_ADDR,也就是从全局存储到了局部的存储。在华为昇腾的设备角度看来，GM_ADDR是所有核共享的内存，而LM_ADDR则是每个核内部独享的内存。

那怎么实现这一操作，一定有一种类似的搬运函数？

```
__aicore__ inline void CopyIn( int32_t progress)
{
    // alloc tensor from queue memory
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    // copy progress_th tile from global tensor to local tensor
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
    AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
    // enque input tensors to VECIN queue
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
```
答案就在这里。

喂喂，这边的inQueueX是什么东西？有脏东西？

要理解这个`inQueue`,关键还是要理解最开始的`Init`过程。


```c
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    // get start index for current core, core parallel
    xGm.SetGlobalBuffer((__gm__ half*)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    yGm.SetGlobalBuffer((__gm__ half*)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    zGm.SetGlobalBuffer((__gm__ half*)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
}
```
这里只列出了x来进行一个全局缓冲和队列的作用，实际上源代码肯定包括x,y,z。
数据到底是怎么搬运的？
**数据搬运流程：**

- 一开始的时候，数据在NPU的内存条(DDR)上。经过了SetGlobalBuffer这一句，每一个 **AI CORE** 就理解了自己的职责范围所在。这个时候其实还没有真正地分配空间，虽然看着像。但这里确实没有从HBM到SRAM的搬运。

- 还是用大哥和小弟的思路进行理解。每位大哥会被分配到要处理一定的数据，也就是BLOCK_LENGTH的数据。每个人从哪里开始处理数据呢？就是初始位置加上每个大哥处理的位置×这个大哥的序号。比如初始位置是100，每个大哥处理100个，这个大哥的序号是8，那第0个大哥从100处理到199，8号的处理序号就是900-999。

**关键参数：**

- **GetBlockIdx()**：告诉你这是第几个Core（第几位大哥）
- **BLOCK_LENGTH**：每个Core处理多少数据

**数据分配公式：**

```
Core数量 × 每个Core处理的数据量 = 总数据量
```

示例：USE_CORE_NUM = 8, BLOCK_LENGTH = 2048 → 总数据量 = 16,384

```c
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
```

- 在pipe.InitBuffer这一行为什么又出现了TILE_LENGTH呢？这个TILE_LENGTH和BLOCK_LENGTH有什么区别，有什么联系？这就好像每个包工头收到了每人要完成一栋楼的任务，那包工头有的包工头可能一天的工作量是一层楼，有的是两层楼。一栋楼如果是16层的话，那有的人花8天就能搞定，有的人花16天才能搞定。TILE_LENGTH就是一层楼或者两层楼，也就是一次的任务量。

- 那TILE_LENGTH应当设定为多少？这就是个和经验相关的问题了。一般来说，能每次多运行一点数据是刚好的，但SRAM的大小就那么大，太多了又会溢出，所以你不能设置太多的数据。

- 文中给出的数据是128. 为什么是这样？考虑到总共有`8*2048`个数据等待计算。将这些数据分配给8个`AICORE`进行运算，每一个分配到了2048个。然后觉得一次128个数据比较好，再加上DOUBLE_BUFFER, 就可以在8个TILE处理完数据。


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


### 讲完了这么多，再回到Process函数。
由于我们每个AI_CORE有`2048`个数据要处理。我们分为了双缓冲，所以大概的流程就是第一个数据被拷贝进来以后计算的时候，再有第二组数据被拷贝进来。每一次处理`128`个数据，所以总共要写`2048/128=16`次循环，来CopyIn,然后计算，然后CopyOut.
这也就是`Process()`函数。
```
__aicore__ inline void Process()
{
    // loop count need to be doubled, due to double buffer
    constexpr int32_t loopCount = TILE_NUM * BUFFER_NUM;
    // tiling strategy, pipeline parallel
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

```
其实我们还是能看到很多结构的抽象。比如CopyIn是如何实现的？Compute是如何实现的？
### 数据处理流程

```
__aicore__ inline void CopyIn( int32_t progress)
{
    // alloc tensor from queue memory
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    // copy progress_th tile from global tensor to local tensor
    AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
    AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
    // enque input tensors to VECIN queue
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
}
```


**第一步、分配本地张量**

```c
LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
```
inQueueX的长度在这里就是128个的长度了。这里给128个长度分配了一个空间，就是xLocal作为起点。当然了，这个空间并不是空穴来潮的。而是在设置这个Class的时候就已经有一个

```c
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;  //输入数据Queue队列管理对象，TPosition为VECIN
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;  //输出数据Queue队列管理对象，TPosition为VECOUT
```

**第二步、数据拷贝**

```c
DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
```
xLocal现在是有数据在身的。

**第三步、入队**

```c
inQueueX.EnQue(xLocal);
```
这个队列进入到计算过程中了。


**第四步、计算中出队**

```c
    // deque input tensors from VECIN queue
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>()
```

**在Compute中使用，获取新的xLocal**
这到底是不是一个好的操作，通过队列形式加入呢？没什么不好的，因为出队、入队用不了多少时间，但是队列就是一个方便管理的东西，否则世界上很多稀缺物品的抢购就不会通过排队进行了。

**第五步、计算,并且将z的计算结果作为z的队列放过去**
```c
    AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
    outQueueZ.EnQue<half>(zLocal);

```
这像是世界上最简单的事情，然而为了达到这一步，不知道绕了多少圈。

**第六步. 释放张量**

```c
inQueueX.FreeTensor(xLocal);
```

*Add操作完成后及时释放*

### 为什么先入队再在Compute中出队？
队列内容在整个函数周期都存在，计算后需要释放资源，并将输出结果Z入队。

**CopyOut** 可能是最不值得多讲的，这里不再赘述了。

## 核心要点总结

- **CUDA范式：** 线程视角，程序员管理每个线程的执行逻辑
- **AscendC范式：** 数据块视角，从整体数据流的角度组织计算
- **流水线优化：** 分批处理+双缓冲，让硬件单元满负荷运转
- **队列管理：** 精细的入队/出队/释放机制确保资源高效利用

---

*理解编程范式的转变，掌握AI芯片编程的核心思想*