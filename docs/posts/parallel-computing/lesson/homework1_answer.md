# Homework-递归表达式及并行度分析

#### 1:给定递归式：$T(n) = 4T(\frac{n}{3}) + n^3$，使用**主定理法**求通项解。（10）
答：根据主定理：

- $a = 4$, $b = 3$, $f(n) = n^3$
- 比较 $f(n) = n^3$ 与 $n^{\log_b{a}} = n^{\log_3{4}}$

这里 $f(n) > n^{\log_b{a}}$，$T(n) = \Theta(n^3)$

#### 2.给定递归式： $T(n) = 3T\left(\frac{n}{9}\right) + \sqrt{n}$ ，使用**主定理法**求通项解。（10）
答：根据主定理：

- $a = 3$, $b = 9$, $f(n) = n^\frac{1}{2}$
- 比较 $f(n) = n^\frac{1}{2}$ 与 $n^{\log_9{3}} = n^{\frac{1}{2}}$

这里 $f(n) \in \Theta(n^{\log_b{a}})$，$T(n) = \Theta(n^\frac{1}{2}\log n)$

#### 3.给定递归式：$T(n) = T(n-1) + \sqrt{n}$，猜测该递归式的解，并使用**数学归纳法**证明。（10）
证明：

令 $S(n) = n^\frac{3}{2}, S(n+1) = (n+1)^\frac{3}{2}$。 
我们来计算 $(n+1)^{3/2} - n^{3/2}$。


$
(n+1)^{3/2} = n^{3/2} \left(1 + \frac{1}{n}\right)^{3/2}
$

对 $\left(1 + \frac{1}{n}\right)^{3/2}$ 用二项展开（或泰勒展开）：


$\left(1 + \frac{1}{n}\right)^{3/2} = 1 + \frac{3}{2} \cdot \frac{1}{n} + \frac{\frac{3}{2} \cdot \frac{1}{2}}{2!} \cdot \frac{1}{n^2} + \dots
$


因此：

$(n+1)^{3/2} = n^{3/2} \left[ 1 + \frac{3}{2n} + \frac{3}{8n^2} - \frac{1}{16n^3} + O\left(\frac{1}{n^4}\right) \right] = n^{3/2} + \frac{3}{2} n^{1/2} + \frac{3}{8} n^{-1/2} - \frac{1}{16} n^{-3/2} + O\left(n^{-5/2}\right)$

因此，$S(n+1) - S(n)=\frac{3}{2}n^\frac{1}{2}+O(\frac{1}{\sqrt{n}})$

令 $P(n) = \frac{2}{3}P(n)， 则P(n) - P(n-1) = \sqrt{n} + O(\frac{1}{\sqrt{n}})$

由于$\sqrt{n} > \frac{1}{\sqrt{n}}$，可知P的递归表达式与T为同一量级，故$T(n) = P(n) = O(n^\frac{3}{2})$
#### 4.给定递归式： $T(n) = T\left(\frac{7}{8}n\right) + n$，猜测该递归式的解，并使用**数学归纳法**证明。（10）
通过递归展开，得到：$T(n) = T\left(\left(\frac{7}{8}\right)^k n\right) + n \left(1 + \frac{7}{8} + \left(\frac{7}{8}\right)^2 + \cdots + \left(\frac{7}{8}\right)^{k-1}\right)$

后面的等比数列的求和：$S_k = 8 \left( 1 - \left(\frac{7}{8}\right)^k \right)$

当 $k \to \infty$ 时，得到：$T(n) = C + 8n$。故猜测解为 $T(n) = O(n)$

 $T(1) = 1$

 $T(n) = C \cdot \frac{7}{8}n + n \leq Cn$，$if \ C \geq 8$

因此 $T(n) = \Theta(n)$
#### 5.给定递归式： $T(n) = 2T\left(\frac{n}{2}\right) + n ^2$，使用**递归展开法**求解。（10）
$$T(n) = n^2 + 2(\frac{n}{2})^2+2^2(\frac{n}{4})^2+...$$
$$ = n^2 (1 + \frac{1}{2} + \frac{1}{4} + ...)$$
$$ = \Theta(n^2)$$
#### 6.给定递归式： $T(n) = 4T(\frac{n}{2}) + \frac{n^2}{\log n}$，使用**递归展开法**求解。（10）
$$T(n) = \frac{n^2}{\log n} + 4\frac{(\frac{n}{2})^2}{\log \frac{n}{2}}+4^2\frac{(\frac{n}{4})^2}{\log \frac{n}{4}}$$
$$ = n^2 (\frac{1}{\log n} + \frac{1}{\log n - 1} + \frac{1}{\log n -2} + ...)$$
$$ = n^2 \sum_{i=1}^{\log n} \frac{1}{i}$$
由 $\sum_{i=1}^{k} \frac{1}{i} = \ln k+\gamma+\frac{1}{2k}-\frac{1}{12k^2}+O(\frac{1}{k^4})$

得 $T(n) = n^2\log\log n$

有1人做错。
#### 7.读程序，分析代码求出$T_1, T_\infty$，求并行度。(10)
```c
cilk_for ( int i = 0; i < N; ++i)
	for (int j = 0; j < N; ++j)
 		A[i][j]++;
```
$$ Work: \Theta(N^2) $$
$$ Span: \Theta(N +\log(N)) =  \Theta(N)$$
$$ Parallelism: \Theta(N) $$

#### 8.读程序，分析代码求出$T_1, T_\infty$，求并行度。(15)
```c
// Floyd-Warshall 算法
for (int k = 0; k < N ; ++ k )
	cilk_for (int i = 0; i < N ; ++ i )
		cilk_for (int j = 0; j < N ; ++ j )
			d[i][j] = min(d[i][j] ,d[i][k] + d[k][j]);
```
$$ Work: \Theta(N^3) $$
$$ Span: \Theta(N(\log N + \log N)) =  \Theta(N\log N)$$
$$ Parallelism: \Theta(\frac{N^2}{\log(N)}) $$

有1人做错。

#### 9.读程序，分析代码求出$T_1, T_\infty$，求并行度。(15)
```c
int sum(int *A, int n) {
    if (n == 1) return A[0];
    int mid = n / 2;
    int x, y;
    cilk_spawn x = sum(A, mid);
    y = sum(A + mid, n - mid);
    cilk_sync;
    return x + y;
}
```
$$ Work: \Theta(N) $$
$$ Span: \Theta(\log N)$$
$$ Parallelism: \Theta(\frac{N}{\log N}) $$