# aclrtSynchronizeStream 实战示例

让我通过几个从简单到复杂的例子，展示 `aclrtSynchronizeStream` 的实际应用场景。

---

## 📚 示例1：基础使用 - 确保结果可用

### 场景：计算完成后立即访问结果

```c
#include "acl/acl.h"

int main() {
    // ===== 起手式 =====
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // ===== 准备数据 =====
    size_t size = 1024 * sizeof(float);
    
    // 分配设备内存
    float *d_input, *d_output;
    CHECK_ACL(aclrtMalloc((void**)&d_input, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_output, size, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 主机内存
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    // 初始化输入数据
    for (int i = 0; i < 1024; i++) {
        h_input[i] = (float)i;
    }
    
    // ===== 异步操作序列 =====
    // 1. 拷贝输入数据到设备（异步）
    CHECK_ACL(aclrtMemcpyAsync(d_input, h_input, size, 
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    
    // 2. 启动kernel计算（异步）
    my_kernel<<<1024, stream>>>(d_input, d_output);
    
    // 3. 拷贝结果回主机（异步）
    CHECK_ACL(aclrtMemcpyAsync(h_output, d_output, size, 
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    
    // ⚠️ 关键点：此时所有操作都在后台进行，CPU已经继续执行了
    
    // ===== 同步等待 =====
    // 🔑 必须调用 SynchronizeStream，否则 h_output 中的数据可能是垃圾
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // ✅ 现在可以安全访问结果了
    printf("结果: h_output[0] = %f\n", h_output[0]);
    printf("结果: h_output[100] = %f\n", h_output[100]);
    
    // ===== 清理资源 =====
    free(h_input);
    free(h_output);
    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    
    return 0;
}
```

### 🔍 时序分析

```
时间轴 →

CPU线程:  │提交拷贝│提交kernel│提交拷贝│ ─── 继续执行 ─── │同步等待│访问结果│
          ↓        ↓         ↓                            ↓       ↓
          下发任务  下发任务   下发任务                      阻塞    继续

Stream:            │拷贝H→D│──│计算│──│拷贝D→H│──完成
                   └─────────────┬────────────┘
                                 │
                   如果没有SynchronizeStream，
                   CPU可能在这里就读取h_output了！
                   → 结果：垃圾数据！
```

---

## 📚 示例2：多Stream并行 - 选择性同步

### 场景：两个独立任务并行执行，分别等待

```c
int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    
    // 创建两个Stream
    aclrtStream stream1, stream2;
    CHECK_ACL(aclrtCreateStream(&stream1));
    CHECK_ACL(aclrtCreateStream(&stream2));
    
    // ===== 准备数据 =====
    size_t size = 1024 * sizeof(float);
    
    // Stream1的数据
    float *d_input1, *d_output1, *h_input1, *h_output1;
    CHECK_ACL(aclrtMalloc((void**)&d_input1, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_output1, size, ACL_MEM_MALLOC_HUGE_FIRST));
    h_input1 = (float*)malloc(size);
    h_output1 = (float*)malloc(size);
    
    // Stream2的数据
    float *d_input2, *d_output2, *h_input2, *h_output2;
    CHECK_ACL(aclrtMalloc((void**)&d_input2, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_output2, size, ACL_MEM_MALLOC_HUGE_FIRST));
    h_input2 = (float*)malloc(size);
    h_output2 = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < 1024; i++) {
        h_input1[i] = (float)i;
        h_input2[i] = (float)(i * 2);
    }
    
    // ===== Stream1：处理任务A =====
    printf("提交Stream1任务（任务A）\n");
    CHECK_ACL(aclrtMemcpyAsync(d_input1, h_input1, size, 
                               ACL_MEMCPY_HOST_TO_DEVICE, stream1));
    taskA_kernel<<<1024, stream1>>>(d_input1, d_output1);
    CHECK_ACL(aclrtMemcpyAsync(h_output1, d_output1, size, 
                               ACL_MEMCPY_DEVICE_TO_HOST, stream1));
    
    // ===== Stream2：处理任务B =====
    printf("提交Stream2任务（任务B）\n");
    CHECK_ACL(aclrtMemcpyAsync(d_input2, h_input2, size, 
                               ACL_MEMCPY_HOST_TO_DEVICE, stream2));
    taskB_kernel<<<1024, stream2>>>(d_input2, d_output2);
    CHECK_ACL(aclrtMemcpyAsync(h_output2, d_output2, size, 
                               ACL_MEMCPY_DEVICE_TO_HOST, stream2));
    
    printf("两个任务已提交，正在后台并行执行...\n");
    
    // ===== 场景1：只需要任务A的结果 =====
    printf("等待任务A完成...\n");
    CHECK_ACL(aclrtSynchronizeStream(stream1));  // 只等待Stream1
    printf("任务A完成！结果: %f\n", h_output1[0]);
    
    // 注意：此时Stream2可能还在执行！
    
    // CPU可以做其他工作...
    printf("CPU在做其他工作...\n");
    for (int i = 0; i < 100000000; i++) {
        // 模拟CPU工作
    }
    
    // ===== 场景2：现在需要任务B的结果 =====
    printf("等待任务B完成...\n");
    CHECK_ACL(aclrtSynchronizeStream(stream2));  // 等待Stream2
    printf("任务B完成！结果: %f\n", h_output2[0]);
    
    // ===== 清理 =====
    // ... 释放资源的代码 ...
    
    return 0;
}
```

### 🔍 时序分析

```
时间轴 →

CPU:     │提交A│提交B│─做其他工作─│等A│用A│─做更多工作─│等B│用B│
         ↓    ↓                  ↓       ↓          ↓   ↓
         
Stream1: │──拷贝──│──计算A──│──拷贝──│完成
                                      ↑
                              SynchronizeStream(stream1)阻塞到这里

Stream2:       │──拷贝──│────计算B(较慢)────│──拷贝──│完成
                                                      ↑
                                     SynchronizeStream(stream2)阻塞到这里
                                     
优势：Stream1完成后CPU不用等Stream2，可以继续做其他事
```

---

## 📚 示例3：流水线处理 - 批量数据处理

### 场景：处理多批数据，每批完成后立即处理下一批

```c
#define BATCH_COUNT 10
#define BATCH_SIZE 1024

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // ===== 准备数据 =====
    size_t size = BATCH_SIZE * sizeof(float);
    
    float *d_input, *d_output;
    CHECK_ACL(aclrtMalloc((void**)&d_input, size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&d_output, size, ACL_MEM_MALLOC_HUGE_FIRST));
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    // ===== 批量处理 =====
    for (int batch = 0; batch < BATCH_COUNT; batch++) {
        printf("\n===== 处理第 %d 批数据 =====\n", batch);
        
        // 准备当前批次的输入数据
        for (int i = 0; i < BATCH_SIZE; i++) {
            h_input[i] = (float)(batch * BATCH_SIZE + i);
        }
        
        // 异步处理当前批次
        CHECK_ACL(aclrtMemcpyAsync(d_input, h_input, size, 
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
        
        process_kernel<<<BATCH_SIZE, stream>>>(d_input, d_output);
        
        CHECK_ACL(aclrtMemcpyAsync(h_output, d_output, size, 
                                   ACL_MEMCPY_DEVICE_TO_HOST, stream));
        
        // 🔑 等待当前批次完成
        CHECK_ACL(aclrtSynchronizeStream(stream));
        
        // ✅ 立即处理结果（比如写入文件、累加统计等）
        printf("第 %d 批完成，结果样本: %f, %f, %f\n", 
               batch, h_output[0], h_output[100], h_output[500]);
        
        // 可以立即保存结果
        char filename[100];
        sprintf(filename, "output_batch_%d.bin", batch);
        WriteFile(filename, size, h_output, size);
        
        // 处理完后，立即处理下一批
    }
    
    printf("\n所有批次处理完成！\n");
    
    // ===== 清理 =====
    free(h_input);
    free(h_output);
    CHECK_ACL(aclrtFree(d_input));
    CHECK_ACL(aclrtFree(d_output));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    
    return 0;
}
```

---

## 📚 示例4：错误示例 - 忘记同步导致的问题

### ❌ 错误代码

```c
int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    size_t size = 1024 * sizeof(float);
    float *d_data, *h_data;
    CHECK_ACL(aclrtMalloc((void**)&d_data, size, ACL_MEM_MALLOC_HUGE_FIRST));
    h_data = (float*)malloc(size);
    
    // 初始化
    for (int i = 0; i < 1024; i++) {
        h_data[i] = (float)i;
    }
    
    // 异步拷贝
    CHECK_ACL(aclrtMemcpyAsync(d_data, h_data, size, 
                               ACL_MEMCPY_HOST_TO_DEVICE, stream));
    
    // 异步计算
    compute_kernel<<<1024, stream>>>(d_data);
    
    // 异步拷贝回来
    CHECK_ACL(aclrtMemcpyAsync(h_data, d_data, size, 
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    
    // ❌ 致命错误：忘记同步！
    // CHECK_ACL(aclrtSynchronizeStream(stream));  // 这行被注释掉了
    
    // ❌ 直接访问结果
    printf("结果: %f\n", h_data[0]);  // 可能输出垃圾数据！
    
    // ❌ 直接释放资源
    CHECK_ACL(aclrtFree(d_data));  // kernel可能还在使用d_data！
    
    // 程序可能崩溃或产生不可预测的结果
    
    return 0;
}
```

### ✅ 正确代码

```c
int main() {
    // ... 前面相同 ...
    
    CHECK_ACL(aclrtMemcpyAsync(h_data, d_data, size, 
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    
    // ✅ 正确：同步等待
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // ✅ 现在可以安全访问
    printf("结果: %f\n", h_data[0]);
    
    // ✅ 现在可以安全释放
    CHECK_ACL(aclrtFree(d_data));
    
    return 0;
}
```

---

## 📚 示例5：性能优化 - 双缓冲技术

### 场景：利用异步特性实现计算和数据传输的重叠

```c
#define NUM_BUFFERS 2
#define BATCH_COUNT 100
#define BATCH_SIZE 1024

int main() {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    
    // 创建两个Stream实现双缓冲
    aclrtStream compute_stream, copy_stream;
    CHECK_ACL(aclrtCreateStream(&compute_stream));
    CHECK_ACL(aclrtCreateStream(&copy_stream));
    
    size_t size = BATCH_SIZE * sizeof(float);
    
    // 双缓冲：两套设备内存
    float *d_input[NUM_BUFFERS], *d_output[NUM_BUFFERS];
    float *h_input[NUM_BUFFERS], *h_output[NUM_BUFFERS];
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        CHECK_ACL(aclrtMalloc((void**)&d_input[i], size, 
                              ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc((void**)&d_output[i], size, 
                              ACL_MEM_MALLOC_HUGE_FIRST));
        h_input[i] = (float*)malloc(size);
        h_output[i] = (float*)malloc(size);
    }
    
    // ===== 流水线处理 =====
    for (int batch = 0; batch < BATCH_COUNT; batch++) {
        int buf_idx = batch % NUM_BUFFERS;  // 轮流使用两个缓冲区
        
        printf("处理批次 %d (使用缓冲区 %d)\n", batch, buf_idx);
        
        // 准备输入数据
        for (int i = 0; i < BATCH_SIZE; i++) {
            h_input[buf_idx][i] = (float)(batch * BATCH_SIZE + i);
        }
        
        // 🔑 关键：确保当前缓冲区可用（上上次的计算已完成）
        if (batch >= NUM_BUFFERS) {
            CHECK_ACL(aclrtSynchronizeStream(compute_stream));
        }
        
        // 异步拷贝输入
        CHECK_ACL(aclrtMemcpyAsync(d_input[buf_idx], h_input[buf_idx], size, 
                                   ACL_MEMCPY_HOST_TO_DEVICE, copy_stream));
        
        // 等待拷贝完成再计算
        CHECK_ACL(aclrtSynchronizeStream(copy_stream));
        
        // 异步计算
        compute_kernel<<<BATCH_SIZE, compute_stream>>>(
            d_input[buf_idx], d_output[buf_idx]);
        
        // 异步拷贝输出（与下一批次的输入拷贝可以重叠）
        CHECK_ACL(aclrtMemcpyAsync(h_output[buf_idx], d_output[buf_idx], size, 
                                   ACL_MEMCPY_DEVICE_TO_HOST, copy_stream));
        
        // 🎯 优势：当前批次在计算时，可以准备下一批次的数据
    }
    
    // 🔑 最后：等待所有任务完成
    CHECK_ACL(aclrtSynchronizeStream(compute_stream));
    CHECK_ACL(aclrtSynchronizeStream(copy_stream));
    
    printf("所有批次处理完成！\n");
    
    // ===== 清理 =====
    for (int i = 0; i < NUM_BUFFERS; i++) {
        free(h_input[i]);
        free(h_output[i]);
        CHECK_ACL(aclrtFree(d_input[i]));
        CHECK_ACL(aclrtFree(d_output[i]));
    }
    
    CHECK_ACL(aclrtDestroyStream(compute_stream));
    CHECK_ACL(aclrtDestroyStream(copy_stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    
    return 0;
}
```

### 🔍 双缓冲时序图

```
批次:    0           1           2           3
       ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
拷入:  │Buf0   │   │Buf1   │   │Buf0   │   │Buf1   │
       └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘
           │           │           │           │
计算:      └───►┌───┐  └───►┌───┐  └───►┌───┐  └───►┌───┐
               │Buf0│      │Buf1│      │Buf0│      │Buf1│
               └─┬─┘      └─┬─┘      └─┬─┘      └─┬─┘
                 │          │          │          │
拷出:            └──►┌──┐   └──►┌──┐   └──►┌──┐   └──►
                     │B0│       │B1│       │B0│
                     └──┘       └──┘       └──┘
                     
同步点:                  ↑这里确保Buf0可重用
                                  ↑这里确保Buf1可重用
```

---

## ✅ 核心要点总结

### 何时必须调用 `aclrtSynchronizeStream`？

| 场景 | 是否需要 | 原因 |
|------|---------|------|
| **访问计算结果** | ✅ 必须 | 否则读到未完成的数据 |
| **释放设备内存** | ✅ 必须 | 否则释放正在使用的内存 |
| **修改输入数据** | ✅ 必须 | 否则影响正在进行的计算 |
| **切换到其他Stream** | ❌ 不需要 | 不同Stream独立 |
| **只是提交任务** | ❌ 不需要 | 异步提交即可 |
| **程序结束前** | ✅ 必须 | 确保所有工作完成 |

### 记忆口诀

```
异步提交快如风，
访问结果须同步。
释放资源先等待，
否则程序会出错！
```

希望这些例子能帮助你理解 `aclrtSynchronizeStream` 的实际应用！🚀