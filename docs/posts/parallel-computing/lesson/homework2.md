# Homework 2
### 一、Work-Span Analysis

给定长度为 $n$ 的数组 $A$。考虑如下并行归并排序：

1. 递归并行地对 $A[1..n/2]$、$A[n/2+1..n]$ 排序；
2. 用如下 **并行 Merge** 合并两个已排序数组 $X,Y$（总长 $n$）：
   - 取 $X$ 的中位元素 $x=X[\frac{|X|}{2}]$，在 $Y$ 中用二分搜索定位其秩 $k$；
   - 令 $x$ 放到输出的正确位置；
   - 递归并行合并左半部分与右半部分。

1.1 根据上述归并算法的描述，计算 Work $T_1(n)$ 和 Span $T_\infty(n)$。


2.2 改变 **并行 Merge** 的方式：

假设X与Y的长度均为$n$，下标从$1$开始。
1. 并行取$X[\log n],X[2\log n],X[3\log n],\dots,X[\frac{n}{\log n}\log n]$的元素，在 $Y$ 上二分搜索得到秩，记作 $XinY[1],XinY[2],XinY[3],\dots,XinY[\frac{n}{\log n}]$，令 $XinY[0]=1, XinY[\frac{n}{\log n}+1]=n$。将$X[i\log n \dots(i+1)\log n]$与 $Y[XinY[i]\dots XinY[i+1]]$的合并作为一个任务并行执行。
2. 对于$XinY[i+1]-XinY[i] > \log n$的情况，取$Y[XinY[i]+\log n], Y[XinY[i]+2\log n],...,Y[XinY[i+1]]$，并行地在 $X$ 上二分搜索得到秩。将这个任务变成 $\Theta(\frac{XinY[i+1]-XinY[i]}{\log n})$个子任务并行执行。
3. 每个子任务内，串行使用双指针的方式，依次将$X$与$Y$合并。

分析使用这种**并行 Merge**的归并排序的Work和Span。



### 二、Cache-oblivious Algorithm

一维卷积算法的输入为一个包含 $r + w$ 个元素的行数组 $A$ 和一个包含 $w$ 个元素的权重数组 $B$，输出结果为包含 $r$ 个元素的输出数组 $C$。对于所有 $i = 0, 1, \dots, r - 1$ 和 $j = 0, 1, \dots, w - 1$，需要计算函数 $\text{foo}(A[i + j], B[j])$的值，并将结果累加到 $C[i]$ 中。

下面，你将分析一维卷积的循环实现和递归实现的工作量和缓存复杂度。假设有一个全相联（fully associative）的缓存，且采用最近最少使用（LRU）替换策略。缓存的大小为 $M$，缓存行大小为 $B$。假设调用函数 $\text{foo}$ 会涉及 $\Theta(1)$ 的工作量和缓存未命中的次数。

```c
void loop_convolution(int64_t* A, int64_t* B, int64_t* C, int64_t r, int64_t w) {
	for(int64_t i = 0; i < r; i++) {
		for(int64_t j = 0; j < w; j++) {
			C[i] += foo(A[i + j], B[j]);
		}
	}
}
```

2.1`loop_convolution` 的工作量是多少？




2.2 当 $r + w < M / 100$ 时，`loop_convolution` 的缓存复杂度是多少？




2.3 当 $r > M$ 且 $w > M$ 时，`loop_convolution` 的缓存复杂度是多少？



```c
void rec_convolution(int64_t* A, int64_t* B, int64_t* C, int64_t r, int64_t A_index, int64_t w, int64_t B_index) {
	if (r == 1 && w == 1) {
		C[A_index] += foo(A[A_index + B_index], B[B_index]);
	} else {
		int64_t r_half = r / 2;
		int64_t w_half = w / 2;
		if (r > w) {
			rec_convolution(A, B, C, r_half, A_index, w, B_index);
			rec_convolution(A, B, C, r_half, A_index + r_half, w, B_index);
		} else {
			rec_convolution(A, B, C, r, A_index, w_half, B_index);
			rec_convolution(A, B, C, r, A_index, w_half, B_index + w_half);
		}
	}
}
```

2.4 `rec_convolution` 的工作量是多少？




2.5 `rec_convolution` 的缓存复杂度可以通过以下递推关系来描述：
$$
Q(r, w) \leq \begin{cases} 
2Q(r / 2, w) + \Theta(1) & \text{if} \; r \geq w \text{ and } r + w > \alpha M, \\
2Q(r, w / 2) + \Theta(1) & \text{if } r < w \text{ and } r + w > \alpha M, \\
\Theta((r + w) / B) & \text{if } r + w \leq \alpha M, 
\end{cases}
$$

其中 $\alpha < 1$ ，为某个常数。请解释原因。



### 三、Lock & Lockfree Synchronization
Adam 设计了一个并发队列。以下是他为并发队列编写的伪代码。

```c
enqueue(x):
	Q.lock()
	Q.push(x)
	Q.unlock()
```

```c
dequeue():
	Q.lock()
	if Q.size == 0:
		y = null
	else:
		y = Q.pop()
	Q.unlock()
	return y
```

3.1 Adam发现它的效率非常低。为什么？



3.2 在你的帮助下，Adam 重新设计了它。

- 线程1将 $x$ 插入队列 $Q$ 
- 线程2从队列 $Q$ 弹出 $y$ 

**伪代码如下：**

```c
enqueue(x):
	lock1.lock()
	lock2.lock()
	Q.push(x)
	lock2.unlock()
	lock1.unlock()
```

```c
dequeue():
	lock2.lock()
	lock1.lock()
	if Q.size == 0:
		y = null
	else:
		y = Q.pop()
	lock1.unlock()
	lock2.unlock()
	return y
```

该代码正确吗？为什么？简要描述发生死锁的 3 种条件。



3.3 为了解决这个问题，Adam 设计了一个无锁并发队列。请帮助他完成他的代码。
提示：`CAS(&x, y, z)` 比较 `x` 和 `y`，如果 `x == y`，则将 `x` 赋值为 `z` 并返回 `true`，否则返回 `false`。

```c
struct Q:
	data[SIZE] = {} // buffer to store data.
					// Assume SIZE is sufficiently large
	popIndex = 0 	// the index to pop
	popMax = 0 		// the index next to the max index that could be popped ,
					// which means data [ popMax ] couldn ’t be popped
	pushIndex = 0 	// the index to push

enqueue(x):
	do {
		pushIndex = Q.pushIndex
	} while (!CAS(&(Q.pushIndex), pushIndex, pushIndex + 1))

	Q.data[pushIndex] = x // write to buffer

	while (!CAS(&(Q.popMax), pushIndex, pushIndex + 1)) {
		yield()
	}

dequeue():
	do {
		popIndex = Q.popIndex
		popMax = Q.popMax
		// if the buffer is empty
		if ((1) _______________________________________________) {
			return null
		}
		y = Q.data[popIndex] // read from buffer
		// if read the right data
		if ((2) _______________________________________________) {
			return y
		}
	} while (true)

```

你需要填写 (1) 和 (2) 的条件来完成代码逻辑。



### 四、Parallel Algorithms
设计一个并行前缀$\otimes$算法，通过$x[1\dots n]$计算$y[1\dots n]$，满足：
$$y[1] = x[1]$$
$$y[2] = x[1] \otimes x[2]$$
$$y[3] = x[1] \otimes x[2] \otimes x[3]$$
$$\vdots$$
$$y[n] = x[1] \otimes x[2] \otimes x[3] \otimes \cdots \otimes x[n]$$

给出一种尽可能高效的计算前缀$\otimes$的方法，写出代码，并分析Work、Span和Parallelism，假设$\otimes$计算复杂度为$\Theta(1)$。
