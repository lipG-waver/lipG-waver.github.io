---
title: OpenMP 基础入门
date: 2025-10-08
author: 离谱纪-Waver
---

# OpenMP 基础入门

OpenMP (Open Multi-Processing) 是用于共享内存并行编程的 API。

## 什么是 OpenMP？

OpenMP 提供了一套编译器指令和库函数，可以轻松地将串行程序转换为并行程序。

## 基本示例
```c
#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }
    return 0;
}