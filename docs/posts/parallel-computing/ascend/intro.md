# 从CUDA到AscendC：编程视角的转变——CPU一侧介绍

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

## 如果我就是昇腾的设计师，我会怎么设计？
```c
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
```
以上为最常见的实现矩阵加法的函数定义。
这里有一个略显怪异的东西， **GM_ADDR** .

这是什么呢？在这之后给出了解释，
```c
#define GM_ADDR __gm__ uint8_t*
```
这代表了之后代码所有的GM_ADDR都会被编译器替换为 `__gm__ uint8_t*`. `__gm__` 则是一个变量类型限定符，表明该指针指向的 **Global Memory(全局内存)** 上的地址。那这个全局内存是CPU上的还是NPU上的呢？这是共享内存，数据是存放在CPU这一侧的设备上的，只是NPU能通过其DMA搬运单元来将共享内存的数据搬运到 **Local Memory** ，这又是后话。现在还是让我们看一下CPU这一侧是如何生成共享内存，如何为共享内存分配要计算的数据。


### 起手式
```c
    // AscendCL初始化
    CHECK_ACL(aclInit(nullptr));
    // 运行管理资源申请
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
```

### 1️⃣ **CHECK_ACL 宏**

```c
CHECK_ACL(aclInit(nullptr));
```

**作用**：包装错误检察宏

**工作原理**：
- 每个 `acl` 开头的函数都会返回一个错误码（`aclError`类型）
- `CHECK_ACL` 宏会检查返回值，如果出错就报告错误信息并退出
- 类似于断言（assert），但更友好

**官方实现**：
```c
#define CHECK_ACL(x)                                                                     
    do {                                                                                    
        aclError __ret = x;                                                                 
        if (__ret != ACL_ERROR_NONE) {                                                      
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; 
        }                           
    while (0);                                                        
```

**为什么需要**：
- ✓ 及早发现问题（哪一步出错了？）
- ✓ 提供错误位置（第几行？）
- ✓ 简化代码（不用每次都写 `if` 判断）


### 2️⃣ **aclInit - 初始化运行环境**

```c
aclInit(nullptr);
```

**作用**：初始化整个 AscendCL 运行时环境

**具体做什么**：
- 加载驱动程序
- 初始化底层硬件接口
- 准备运行时库
- 建立与NPU的通信通道

**类比**：就像给汽车打火，发动引擎才能开始工作

**只需调用一次**：通常在 `main()` 函数开头调用

---

### 3️⃣ **deviceId - 指定使用哪块NPU芯片**

```c
int32_t deviceId = 0;
```

**含义**：NPU设备的编号（从0开始）

**实际场景**：

```
服务器上可能有多块NPU卡：

┌─────────────┐
│ deviceId=0  │  ← 第1块NPU卡
├─────────────┤
│ deviceId=1  │  ← 第2块NPU卡
├─────────────┤
│ deviceId=2  │  ← 第3块NPU卡
└─────────────┘

deviceId=0 表示使用第1块卡
```

**如何选择**：
- 单卡场景：固定用 `deviceId = 0`
- 多卡场景：可以指定不同的卡号实现负载均衡

---

### 4️⃣ **aclrtSetDevice - 绑定设备到当前线程**

```c
aclrtSetDevice(deviceId);
```

**作用**：告诉系统"我这个线程要用这块NPU卡"

**实际效果**：
- 将当前CPU线程与指定的NPU设备绑定
- 后续所有操作（内存分配、kernel启动等）都在这块卡上执行
- 自动创建默认的 Context（上下文环境）

**类比**：
```
就像去银行办业务：
1. 你走进银行大厅（aclInit）
2. 选择某个柜台办理（aclrtSetDevice）
3. 之后的存款、取款都在这个柜台进行
```

**线程绑定示例**：
```c
// 线程A使用设备0
void* threadA(void* arg) {
    aclrtSetDevice(0);  // 线程A绑定到设备0
    // ... 这个线程的所有操作都在设备0上
}

// 线程B使用设备1
void* threadB(void* arg) {
    aclrtSetDevice(1);  // 线程B绑定到设备1
    // ... 这个线程的所有操作都在设备1上
}
```

---

### 5️⃣ **aclrtCreateStream - 创建任务流水线**

```c
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);
```

**作用**：创建一个任务执行的"流水线通道"

**Stream是什么**：
```
Stream = 任务队列 = 流水线

想象一条传送带：
  [任务1] → [任务2] → [任务3] → ...
  
在这条传送带上的任务按顺序执行
```

**为什么需要Stream**：

| 用途 | 说明 |
|------|------|
| **顺序保证** | 同一Stream中的任务按提交顺序执行 |
| **异步执行** | CPU提交任务后立即返回，NPU在后台执行 |
| **并行能力** | 多个Stream可以并行工作 |
| **资源隔离** | 不同任务流可以独立管理 |

### 🎯 完整流程类比

把整个起手式比作开餐厅：

```c
// 1. 开门营业（初始化系统）
CHECK_ACL(aclInit(nullptr));

// 2. 选择使用哪个厨房（选择NPU设备）
int32_t deviceId = 0;  // 使用1号厨房
CHECK_ACL(aclrtSetDevice(deviceId));

// 3. 准备一条上菜的流水线（创建任务队列）
aclrtStream stream = nullptr;
CHECK_ACL(aclrtCreateStream(&stream));

// 现在可以开始接单做菜了！
```

---

### 📊 资源关系图

```
┌─────────────────────────────────────────┐
│          AscendCL运行时                  │
│         (aclInit初始化)                  │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        │             │
   ┌────▼───┐    ┌───▼────┐
   │Device 0│    │Device 1│  ← aclrtSetDevice选择
   └────┬───┘    └───┬────┘
        │            │
   ┌────▼────┐  ┌───▼─────┐
   │Context  │  │Context  │  ← 自动创建
   └────┬────┘  └───┬─────┘
        │            │
   ┌────▼────┐  ┌───▼─────┐
   │Stream 1 │  │Stream A │  ← aclrtCreateStream创建
   │Stream 2 │  │Stream B │
   └─────────┘  └─────────┘
```

---

### ✅ 短暂的小结

| 步骤 | API | 作用 | 类比 |
|------|-----|------|------|
| **1** | `aclInit()` | 初始化整个运行环境 | 餐厅开门营业 |
| **2** | `aclrtSetDevice()` | 选择使用哪块NPU卡 | 选择在哪个厨房工作 |
| **3** | `aclrtCreateStream()` | 创建任务执行队列 | 准备一条传菜流水线 |

### ACL RUNTIME API
让我们粗略扫描一遍都有哪些aclrt(acl runtime)的API吧：
```c
// Runtime模块的Device相关API
aclrtSetDevice(deviceId)        // 设置设备
aclrtResetDevice(deviceId)      // 重置设备
aclrtGetDevice(&deviceId)       // 获取当前设备
aclrtSynchronizeDevice()        // 同步设备

// Runtime模块的Stream相关API
aclrtCreateStream(&stream)      // 创建流
aclrtDestroyStream(stream)      // 销毁流
aclrtSynchronizeStream(stream)  // 同步流


// Runtime模块的Memory相关API
aclrtMalloc(&ptr, size, policy) // 分配内存
aclrtFree(ptr)                  // 释放内存
aclrtMemcpy(dst, src, size)     // 拷贝内存
```
<!-- // // Runtime模块的Event相关API （我不确定是不是真有这些还是说Claude的ai幻觉）
// aclrtCreateEvent(&event)        // 创建事件
// aclrtRecordEvent(event, stream) // 记录事件
// aclrtSynchronizeEvent(event)    // 同步事件 -->

这些内容扫一眼就好。接下来是分配内存的核心环节：
```c
// 分配Host内存
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;
    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
    // 分配Device内存
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // Host内存初始化
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
```
只要看注释，首先是分配了CPU这一侧的内存，用的是`aclrtMallocHost`，然后分配了NPU这一侧的内存，用的是`aclrtMalloc`。首先先用`ReadFile`将输入文件的内容读到CPU这一侧分配出来的内存中，然后再用`aclrtMemcpy`将CPU的内容复制到NPU这一侧的内存。由此我们看到了，NPU也能一口气将这么多的东西吃进自己的Global Memory（也称作DDR/HBM）。

有时候有人会说，HBM读写慢，SRAM读写快，所以要将更多内容复制到SRAM上。为什么会有HBM和SRAM这一说法呢？这是因为它们是两种不同的物理存储器件。**HBM（High Bandwidth Memory，高带宽内存）**或**DDR**是一种大容量的存储芯片，通常焊接在NPU芯片附近，容量可以达到几GB到几十GB，但访问速度相对较慢，延迟在几百纳秒级别。而 **SRAM（Static Random Access Memory，静态随机存取存储器）** 则是直接集成在AI Core内部的高速缓存，容量只有几MB，但访问速度极快，延迟只有几纳秒。

这就像是仓库和工作台的关系：HBM是大仓库，能存很多东西但走过去要时间；SRAM是工作台，空间有限但伸手就能拿到。所以在Ascend C编程中，我们要做的就是把需要计算的数据从大仓库（Global Memory/HBM）搬到工作台（Local Memory/SRAM）上，在工作台上快速完成计算，然后再把结果搬回仓库。这就是为什么文档中要做Tiling（数据切块）的原因——工作台太小，一次放不下所有数据，只能分批处理。

接下来开始计算：
```c
    // 用内核调用符<<<>>>调用核函数完成指定的运算,add_custom_do中封装了<<<>>>调用
    add_custom_do(blockDim, nullptr, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
```
这一个add_custom就是大多数人要编写的Kernel函数。你洋洋洒洒在NPU一侧写了很多很多东西，在CPU一侧的调用就体现为这样一句简简单单的话。

这边弄完了以后，将结果拷贝回去，将NPU一侧的资源进行释放即可。

```
CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);
    // 释放申请的资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    // AscendCL去初始化
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
```
刚刚申请的资源包括CPU侧和NPU侧的内存（回忆一下，他们是通过Memcpy的方式进行的搬运）。然后还申请了一个`Stream`,一个`Device`资源，都要还回去（一个是`DestroyStream`,一个是`ResetDevice`）。最后，`aclFinalize`,代表着终结整个过程。
