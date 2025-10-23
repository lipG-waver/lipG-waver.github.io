---
title: 并行计算第三课-前置知识
date: 2025-10-23
author: 离谱纪-Waver
---

# 什么是进程 (Process)

进程是操作系统中**正在执行的程序实例**，它包含了程序运行所需的所有资源和状态信息。

## 进程的内存布局

进程的虚拟内存空间通常按以下方式组织（从高地址到低地址）：

```
高地址
┌─────────────────┐
│     Kernel      │ ← 内核空间（用户程序不能直接访问）
├─────────────────┤
│      Stack      │ ← 栈区（向下增长）
│       ↓         │   - 存储局部变量、函数参数、返回地址
│                 │   - 由 rsp (栈指针) 和 rbp (基址指针) 管理
├─────────────────┤
│       ...       │ ← 未使用空间
│                 │
├─────────────────┤
│       ↑         │
│      Heap       │ ← 堆区（向上增长）
│                 │   - 动态分配的内存（malloc/new）
├─────────────────┤
│      BSS        │ ← 未初始化的全局变量
├─────────────────┤
│      Data       │ ← 已初始化的全局变量和静态变量
├─────────────────┤
│      Text       │ ← 代码段（程序的机器指令）
└─────────────────┘
低地址
```

## 重要的寄存器 (Registers)

寄存器是CPU内部的高速存储单元，用于存储程序执行的关键信息：

- **PC (Program Counter)** / **IP (Instruction Pointer)**: 指向下一条要执行的指令
- **SP (Stack Pointer)** / **rsp**: 指向栈顶的当前位置
- **BP (Base Pointer)** / **rbp**: 指向当前栈帧的基址，用于访问局部变量和参数
- **通用寄存器** (如 rbx, rax, rcx, rdx 等): 用于临时存储数据和计算结果
- **CC (Condition Codes)** / **Flags**: 存储运算结果的状态（如零标志、符号标志、进位标志等）

## 栈的管理

栈使用两个关键指针：
- **栈顶指针 (rsp)**: 总是指向栈的顶部（最后压入的数据）
- **栈底指针 (rbp)**: 指向当前函数栈帧的底部，作为访问局部变量的参考点

当函数调用时，栈会保存返回地址、参数和局部变量。

## I/O 和文件

每个进程都维护一个**文件描述符表 (File Descriptor Table)**：
- **标准输入 (stdin)**: 文件描述符 0
- **标准输出 (stdout)**: 文件描述符 1
- **标准错误 (stderr)**: 文件描述符 2
- 打开的文件、网络连接、管道等也会分配文件描述符

## 进程的其他重要组成部分

- **进程ID (PID)**: 唯一标识符
- **进程状态**: 运行、就绪、阻塞等
- **页表**: 虚拟地址到物理地址的映射
- **打开的文件表**: 管理进程打开的所有文件
- **信号处理器**: 处理异步事件

进程是操作系统资源分配和调度的基本单位，操作系统通过上下文切换在多个进程间切换，实现多任务并发执行。

# 什么是线程 (Thread)

线程是**进程内的执行单元**，是CPU调度和执行的最小单位。一个进程可以包含多个线程，这些线程共享进程的资源，但各自独立执行。

## 线程 vs 进程

### 进程的特点：
- 拥有独立的内存空间
- 进程间切换开销大
- 进程间通信复杂（需要IPC机制）
- 资源隔离性好，更安全

### 线程的特点：
- 共享进程的内存空间
- 线程间切换开销小
- 线程间通信简单（共享内存）
- 轻量级，创建和销毁更快

## 多线程进程的内存布局

```
高地址
┌─────────────────┐
│     Kernel      │ ← 内核空间
├─────────────────┤
│  Thread 1 Stack │ ← 线程1的栈（私有）
├─────────────────┤
│  Thread 2 Stack │ ← 线程2的栈（私有）
├─────────────────┤
│  Thread 3 Stack │ ← 线程3的栈（私有）
├─────────────────┤
│       ...       │
├─────────────────┤
│      Heap       │ ← 堆区（所有线程共享）
├─────────────────┤
│      BSS        │ ← 未初始化全局变量（共享）
├─────────────────┤
│      Data       │ ← 已初始化全局变量（共享）
├─────────────────┤
│      Text       │ ← 代码段（共享）
└─────────────────┘
低地址
```

## 线程的私有资源（每个线程独有）

每个线程都有自己的：

1. **线程栈 (Thread Stack)**
   - 存储该线程的局部变量
   - 函数调用栈
   - 返回地址

2. **寄存器上下文**
   - **PC (程序计数器)**: 每个线程有自己的执行位置
   - **rsp (栈指针)**: 指向该线程自己的栈顶
   - **rbp (基址指针)**: 该线程栈帧的基址
   - **通用寄存器** (rax, rbx, rcx, rdx 等)
   - **条件码寄存器 (CC/Flags)**

3. **线程ID (TID)**

4. **线程局部存储 (Thread Local Storage, TLS)**

5. **信号掩码**

6. **errno 变量**

## 线程的共享资源（同一进程内所有线程共享）

1. **代码段 (Text)**: 程序指令
2. **数据段 (Data/BSS)**: 全局变量和静态变量
3. **堆 (Heap)**: 动态分配的内存，堆顶用sbrk来表示
4. **文件描述符表**: 打开的文件、socket等
5. **进程ID (PID)**: 所有线程属于同一个进程
6. **内存映射区域**
7. **信号处理器**

## 线程上下文切换

当CPU从一个线程切换到另一个线程时：

```
线程A正在运行
    ↓
1. 保存线程A的寄存器状态
   - PC, rsp, rbp, rbx, rax 等
   
2. 保存线程A的栈指针
   
3. 选择下一个要运行的线程B
   
4. 恢复线程B的寄存器状态
   
5. 恢复线程B的栈指针
    ↓
线程B开始运行
```

**注意**: 由于线程共享内存空间，不需要切换页表，因此线程切换比进程切换快得多。

## 线程同步问题

由于线程共享内存，会产生并发问题：

### 竞态条件 (Race Condition)
```c
// 全局变量（所有线程共享）
int counter = 0;

// 线程1和线程2同时执行
counter++;  // 不是原子操作！
// 实际分为三步：
// 1. 从内存读取 counter 到寄存器
// 2. 寄存器值 +1
// 3. 写回内存
```

### 常用同步机制

1. **互斥锁 (Mutex)**
   ```c
   pthread_mutex_t lock;
   pthread_mutex_lock(&lock);
   counter++;  // 临界区
   pthread_mutex_unlock(&lock);
   ```

2. **信号量 (Semaphore)**
   - 控制对共享资源的访问数量

3. **条件变量 (Condition Variable)**
   - 线程间的等待/通知机制

4. **读写锁 (Read-Write Lock)**
   - 允许多个读者，单个写者

## 线程的状态

```
    创建
     ↓
  [就绪] ←──────┐
     ↓          │
  [运行] ───→ [阻塞]
     ↓          ↑
   结束      (等待I/O、锁等)
```

## 多线程的优势

1. **提高响应速度**: 一个线程阻塞时，其他线程继续执行
2. **资源共享**: 线程间通信简单，无需复杂的IPC
3. **经济性**: 创建和切换开销小
4. **多核利用**: 充分利用多核CPU的并行能力

## 多线程的挑战

1. **同步问题**: 需要仔细处理共享数据的访问
2. **死锁**: 多个线程相互等待对方释放资源
3. **调试困难**: 并发bug难以重现
4. **数据一致性**: 需要保证共享数据的正确性

## 示例：线程寄存器快照

假设两个线程在某一时刻的状态：

**线程1**:
- PC: 0x400500 (指向线程1当前执行的指令)
- rsp: 0x7fff1000 (线程1的栈顶)
- rbp: 0x7fff1100 (线程1的栈帧基址)

**线程2**:
- PC: 0x400800 (指向线程2当前执行的指令)
- rsp: 0x7fff2000 (线程2的栈顶)
- rbp: 0x7fff2100 (线程2的栈帧基址)

两个线程有各自的PC和栈，但共享相同的代码段、堆和全局变量。

# sbrk 详解

## 什么是 sbrk

**sbrk** (set break) 是一个**系统调用**，用于调整进程的**堆区 (heap) 大小**。

```
进程内存布局：
┌─────────────────┐
│      Stack      │
│       ↓         │
├─────────────────┤
│       ...       │
├─────────────────┤
│       ↑         │
│      Heap       │ ← sbrk 调整这里
├─────────────────┤ ← program break (brk)
│      BSS        │
├─────────────────┤
│      Data       │
├─────────────────┤
│      Text       │
└─────────────────┘
```

### Program Break

- **program break** (或简称 **brk**) 是堆区的**结束位置**
- 堆从低地址向高地址增长
- program break 标记了堆的当前边界

## sbrk 的作用

```c
#include <unistd.h>

void *sbrk(intptr_t increment);
```

- **参数 increment**:
  - 正数：扩展堆，program break 向上移动
  - 负数：收缩堆，program break 向下移动
  - 0：返回当前 program break 的位置（不改变大小）

- **返回值**: 
  - 成功：返回**调整前**的 program break 位置
  - 失败：返回 `(void *) -1`

### 示例

```c
// 获取当前 program break 位置
void *current_brk = sbrk(0);

// 申请 1024 字节的堆空间
void *new_memory = sbrk(1024);
// new_memory 指向新分配空间的起始地址

// 释放 512 字节
sbrk(-512);
```

## 为什么 sbrk 必须通过系统调用？

### 1. **内存管理由操作系统控制**

进程的虚拟内存布局由**操作系统内核**管理：

```
用户空间 (User Space)
├─ 用户程序只能"看到"虚拟地址
└─ 不知道物理内存在哪里

────────────────────────────────
系统调用边界
────────────────────────────────

内核空间 (Kernel Space)
├─ 维护页表 (Page Table)
├─ 管理物理内存分配
└─ 控制虚拟地址到物理地址的映射
```

**用户程序不能直接修改内存布局**，因为：
- 用户程序运行在非特权模式（用户态）
- 只有内核运行在特权模式（内核态）
- 内存管理需要修改页表等特权数据结构

### 2. **需要更新页表 (Page Table)**

当 sbrk 扩展堆时，内核需要：

```
1. 检查新地址范围是否合法
   - 不能与其他内存区域重叠
   - 不能超过进程的内存限制

2. 更新页表
   - 建立虚拟地址到物理页的映射
   - 设置页的权限（读/写/执行）

3. 可能需要分配物理内存
   - 使用按需分页 (demand paging)
   - 只有真正访问时才分配物理页

4. 更新进程控制块 (PCB)
   - 记录新的 program break 位置
   - 更新进程的内存使用统计
```

### 3. **资源保护和隔离**

```
进程 A               进程 B
  ↓                    ↓
虚拟地址空间       虚拟地址空间
(看起来都是       (看起来都是
 0x00000000...     0x00000000...
 到 0xFFFFFFFF)    到 0xFFFFFFFF)
  ↓                    ↓
  └────────┬───────────┘
           ↓
       内核管理
           ↓
    ┌──────┴──────┐
    │   物理内存   │
    └─────────────┘
```

**如果允许用户程序直接修改内存**：
- 进程可能访问其他进程的内存（安全漏洞）
- 进程可能破坏内核数据结构（系统崩溃）
- 无法实现内存隔离和保护

### 4. **需要进行安全检查**

内核在处理 sbrk 时会进行检查：

```c
// 内核中的 sbrk 处理逻辑（简化版）
sys_brk(new_brk) {
    // 1. 检查地址是否对齐
    if (new_brk % PAGE_SIZE != 0)
        return -EINVAL;
    
    // 2. 检查是否超出限制
    if (new_brk > process->rlimit[RLIMIT_DATA])
        return -ENOMEM;
    
    // 3. 检查是否与其他区域冲突
    if (conflicts_with_stack_or_mmap(new_brk))
        return -ENOMEM;
    
    // 4. 更新页表
    update_page_table(old_brk, new_brk);
    
    // 5. 更新 program break
    process->brk = new_brk;
    
    return old_brk;
}
```

## 系统调用的过程

```
用户程序调用 sbrk(1024)
         ↓
    用户态 (User Mode)
         ↓
    系统调用指令 (syscall / int 0x80)
         ↓
────────────────────────────────
    CPU 切换到内核态
────────────────────────────────
         ↓
    内核态 (Kernel Mode)
         ↓
    内核处理 sbrk 请求
    - 检查权限
    - 更新页表
    - 分配内存
         ↓
────────────────────────────────
    CPU 切换回用户态
────────────────────────────────
         ↓
    用户态 (User Mode)
         ↓
    返回分配的地址
```

## malloc 和 sbrk 的关系

用户程序通常不直接调用 sbrk，而是使用 **malloc**：

```c
// 用户程序
void *ptr = malloc(1024);

// malloc 的内部实现（简化）
void *malloc(size_t size) {
    // 小内存：从已有的堆中分配
    if (size < THRESHOLD) {
        return allocate_from_existing_heap();
    }
    
    // 大内存：通过 sbrk 扩展堆
    void *p = sbrk(size + METADATA_SIZE);
    if (p == (void *) -1)
        return NULL;
    
    // 添加元数据（记录块大小等）
    setup_metadata(p, size);
    
    return p + METADATA_SIZE;
}
```

**malloc 是库函数，sbrk 是系统调用**：
- malloc 在用户态管理内存池
- 当内存池不够时，调用 sbrk 向内核申请更多堆空间
- 现代系统中，malloc 也可能使用 mmap 系统调用

## 总结

**sbrk 必须是系统调用的核心原因**：

1. **特权操作**: 修改内存布局需要内核特权
2. **页表管理**: 需要更新虚拟地址到物理地址的映射
3. **安全隔离**: 防止进程访问非法内存区域
4. **资源控制**: 内核需要跟踪和限制进程的内存使用
5. **系统稳定**: 防止用户程序破坏系统数据结构

**关键点**: 用户程序运行在受限的环境中，任何涉及系统资源（内存、文件、设备等）的操作都必须通过系统调用请求内核代为执行，这是现代操作系统安全和稳定的基础。

# 进程的层次结构 (Process Hierarchy)

进程在操作系统中组织成**树形结构**，每个进程（除了根进程）都有一个父进程，可以有零个或多个子进程。

## 进程树的基本概念

```
            init/systemd (PID 1)
                   |
        ┌──────────┼──────────┐
        |          |          |
     sshd       cron      login
        |                    |
    ┌───┴───┐              bash (父进程)
  sshd    sshd               |
    |       |          ┌─────┼─────┐
  bash    bash         |     |     |
    |       |        vim   gcc   firefox
  python  ./app            |       |
                        ┌──┴──┐  ┌─┴─┐
                       cc1   as  子进程...
```

## 关键进程标识符

每个进程都有两个重要的ID：

1. **PID (Process ID)**: 进程自己的唯一标识符
2. **PPID (Parent Process ID)**: 父进程的PID

```c
#include <unistd.h>

pid_t getpid();   // 获取当前进程的PID
pid_t getppid();  // 获取父进程的PID
```

### 示例

```c
#include <stdio.h>
#include <unistd.h>

int main() {
    printf("My PID: %d\n", getpid());
    printf("My parent's PID: %d\n", getppid());
    return 0;
}

// 输出示例：
// My PID: 12345
// My parent's PID: 12340
```

## 进程的创建：fork()

进程通过 **fork()** 系统调用创建子进程：

```c
#include <unistd.h>

pid_t fork(void);
```

### fork() 的工作原理

```
调用 fork() 之前：
┌─────────────┐
│  父进程     │
│  PID: 100   │
│  PPID: 50   │
└─────────────┘

调用 fork() 之后：
┌─────────────┐          ┌─────────────┐
│  父进程     │          │  子进程     │
│  PID: 100   │ ────────>│  PID: 101   │ (新创建)
│  PPID: 50   │  创建    │  PPID: 100  │
│  fork()返回:│          │  fork()返回:│
│    101      │          │    0        │
└─────────────┘          └─────────────┘
```

**fork() 的特点**：
- 子进程是父进程的**副本**
- 子进程复制父进程的：内存空间、打开的文件、寄存器状态
- 在**父进程**中，fork() 返回**子进程的PID**
- 在**子进程**中，fork() 返回 **0**
- 失败时返回 **-1**

### fork() 示例

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main() {
    pid_t pid;
    int x = 10;
    
    printf("Before fork: PID=%d, x=%d\n", getpid(), x);
    
    pid = fork();  // 创建子进程
    
    if (pid < 0) {
        // fork 失败
        perror("fork failed");
        return 1;
    }
    else if (pid == 0) {
        // 子进程
        x = 20;
        printf("Child: PID=%d, PPID=%d, x=%d\n", 
               getpid(), getppid(), x);
    }
    else {
        // 父进程
        x = 30;
        printf("Parent: PID=%d, child PID=%d, x=%d\n", 
               getpid(), pid, x);
    }
    
    printf("Both: PID=%d, x=%d\n", getpid(), x);
    
    return 0;
}
```

**输出示例**：
```
Before fork: PID=1000, x=10
Parent: PID=1000, child PID=1001, x=30
Both: PID=1000, x=30
Child: PID=1001, PPID=1000, x=20
Both: PID=1001, x=20
```

## 进程树的根：init 进程

```
Linux/Unix 系统启动过程：

1. 内核启动
   ↓
2. 创建 init 进程 (PID 1)
   ↓
3. init 成为所有进程的祖先
```

### init 进程的特点：

- **PID 永远是 1**
- 系统中的**第一个用户空间进程**
- **所有其他进程的祖先**
- 负责系统初始化和服务管理
- 现代 Linux 通常使用 **systemd** 替代传统的 init

### 查看进程树

```bash
# 方法1: pstree 命令
pstree

# 输出示例：
systemd─┬─sshd───sshd───bash───vim
        ├─cron
        ├─dbus-daemon
        └─firefox─┬─{firefox}
                  ├─{firefox}
                  └─{firefox}

# 方法2: ps 命令
ps -ef --forest

# 方法3: htop (可视化工具)
htop
```

## 孤儿进程 (Orphan Process)

当**父进程先于子进程结束**时，子进程变成孤儿进程：

```
初始状态：
    父进程 (PID 100)
       |
    子进程 (PID 101, PPID 100)

父进程退出后：
    init/systemd (PID 1)
       |
    子进程 (PID 101, PPID 1) ← 被 init 收养
```

**处理机制**：
- 孤儿进程会被 **init 进程（PID 1）收养**
- init 成为孤儿进程的新父进程
- PPID 自动变为 1

### 孤儿进程示例

```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程
        printf("Child: PID=%d, PPID=%d\n", getpid(), getppid());
        sleep(5);  // 等待父进程退出
        printf("Child: PID=%d, PPID=%d (after parent exit)\n", 
               getpid(), getppid());  // PPID 变成 1
    }
    else {
        // 父进程
        printf("Parent: PID=%d, exiting...\n", getpid());
        exit(0);  // 父进程立即退出
    }
    
    return 0;
}
```

## 僵尸进程 (Zombie Process)

当**子进程结束，但父进程未回收其状态**时，子进程变成僵尸进程：

```
    父进程 (运行中)
       |
    子进程 (已结束) ← 僵尸状态 <defunct>
```

### 僵尸进程的特征：

- 进程已经终止，但**进程表项仍然存在**
- 保留退出状态，等待父进程读取
- 不占用内存和CPU，但占用进程表项（PID）
- 在 `ps` 命令中显示为 `<defunct>` 或 `Z` 状态

### 僵尸进程示例

```c
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程
        printf("Child: PID=%d, exiting...\n", getpid());
        exit(0);  // 子进程退出
    }
    else {
        // 父进程
        printf("Parent: PID=%d, sleeping...\n", getpid());
        sleep(30);  // 父进程不回收子进程状态
        // 在这30秒内，子进程是僵尸进程
    }
    
    return 0;
}

// 在另一个终端运行：ps aux | grep defunct
// 会看到僵尸进程
```

### 回收子进程：wait() 和 waitpid()

```c
#include <sys/wait.h>

pid_t wait(int *status);
pid_t waitpid(pid_t pid, int *status, int options);
```

**正确的父进程应该回收子进程**：

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程
        printf("Child: PID=%d\n", getpid());
        sleep(2);
        exit(42);  // 退出状态为 42
    }
    else {
        // 父进程
        int status;
        printf("Parent: waiting for child...\n");
        
        wait(&status);  // 等待并回收子进程
        
        if (WIFEXITED(status)) {
            printf("Child exited with status: %d\n", 
                   WEXITSTATUS(status));
        }
    }
    
    return 0;
}
```

## 进程组 (Process Group)

相关的进程可以组织成**进程组**：

```
进程组 PGID=100
├─ 进程 PID=100 (组长)
├─ 进程 PID=101
└─ 进程 PID=102

进程组 PGID=200
├─ 进程 PID=200 (组长)
└─ 进程 PID=201
```

**作用**：
- 方便**批量发送信号**
- Shell 中的**管道和作业控制**使用进程组

```bash
# 在 Shell 中
ls | grep txt | wc -l
# 这三个进程属于同一个进程组
```

```c
#include <unistd.h>

pid_t getpgid(pid_t pid);  // 获取进程组ID
int setpgid(pid_t pid, pid_t pgid);  // 设置进程组
```

## 会话 (Session)

多个进程组可以组成一个**会话**：

```
会话 SID=1000
├─ 前台进程组 PGID=100
│  ├─ bash
│  └─ vim
└─ 后台进程组 PGID=200
   └─ background_job
```

**特点**：
- 每个会话有一个**控制终端**
- 会话中有一个**前台进程组**（接收终端输入）
- 可以有多个**后台进程组**

```c
#include <unistd.h>

pid_t setsid(void);  // 创建新会话
```

## 完整的进程层次结构

```
系统级别：
    init/systemd (PID 1)
       |
    ┌──┴────────────┐
    |               |
会话层：
  Session 1      Session 2
    |               |
进程组层：
  PG 100          PG 200
    |               |
进程层：
  ┌─┴─┐          ┌─┴─┐
  P1  P2         P3  P4
       |              |
线程层：
     ┌─┴─┐         ┌─┴─┐
     T1  T2        T1  T2
```

## 进程树的实际应用

### 1. Shell 中的作业控制

```bash
# 前台运行
sleep 100

# 后台运行
sleep 100 &

# 暂停前台作业
Ctrl+Z

# 查看作业
jobs

# 恢复到前台
fg %1
```

### 2. 守护进程 (Daemon)

守护进程通常会：
1. fork() 创建子进程
2. 父进程退出（子进程变成孤儿）
3. 子进程调用 setsid() 创建新会话
4. 脱离控制终端，在后台运行

### 3. 进程终止时的清理

```c
// 杀死进程组中的所有进程
kill(-pgid, SIGTERM);

// 杀死整个进程树
killall process_name
```

## 总结

进程层次结构的关键点：

1. **树形结构**：每个进程有一个父进程（除了 init）
2. **PID 和 PPID**：标识进程及其父进程
3. **fork() 创建**：子进程继承父进程的大部分属性
4. **init 收养孤儿**：父进程退出后，子进程被 init 接管
5. **wait() 回收僵尸**：父进程应该回收子进程的退出状态
6. **进程组和会话**：更高层次的进程组织方式

这种层次结构是 Unix/Linux 进程管理的基础，提供了灵活的进程控制和资源管理机制。