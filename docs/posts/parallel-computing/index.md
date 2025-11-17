---
title: 并行计算专题
---

# 🚀 并行计算专题

记录并行计算领域的学习笔记和实践经验。

## 📚 文章列表

## 昇腾NPU编程
- [介绍：从英伟达和昇腾不同的地方讲起，CPU一侧](./ascend/intro.md)
- [介绍：继续上一篇，讲解NPU Kernel函数](./ascend/intro_npu.md)
- [AclrtSynchronizeStream实战示例](./ascend/aclrtSynchStream_example.md)
## 并行计算课程
- [第一课：如何加速矩阵的乘法，前置知识、缓存服用和分块](./lesson/ParallelC-lesson1)
- [第三课：多核编程-前置知识](./lesson/ParallelC-lesson3_pre.md)
- [第三课：多核编程](./lesson/ParallelC-lesson3.md)


## 线性注意力/Linear Attention
- [从传统注意力到线性注意力](./linear-attention/softmaxToLinear.md)
- [From Standard Attention to Linear Attention](./linear-attention/softmaxToLinear-en.md)
- [线性注意力的演化过程](./linear-attention/evolution.md)
- [Evolution of Linear Attention](./linear-attention/evolution-english.md)
- [Performer为什么在期望上能做到和Softmax相等](./linear-attention/performer.md)
- [Why Performer is equal to Softmax attention in the aspect of expectation](./linear-attention//performer-en.md)


## 问答
- [为什么商业公司选择GPT,而不是BERT?](./ask&answer/bert-vs-gpt-commerical-performance.md)
- [是否有必要在进行softmax减去最大值，不减的话是否影响精度?](./ask&answer/is_minus_max_necessary.md)
- [利用局部性和簇状最大来抽样Softmax](./ask&answer/sampling_softmax_by_locality.md)
- [是否能算出一行以后直接进行softmax?](./ask&answer/new.md)
