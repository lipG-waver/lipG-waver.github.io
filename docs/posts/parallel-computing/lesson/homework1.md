# Homework-递归表达式及并行度分析

#### 1:给定递归式：$T(n) = 4T(\frac{n}{3}) + n^3$，使用**主定理法**求通项解。（10）

#### 2.给定递归式： $T(n) = 3T\left(\frac{n}{9}\right) + \sqrt{n}$ ，使用**主定理法**求通项解。（10）

#### 3.给定递归式：$T(n) = T(n-1) + \sqrt{n}$，猜测该递归式的解，并使用**数学归纳法**证明。（10）

#### 4.给定递归式： $T(n) = T\left(\frac{7}{8}n\right) + n$，猜测该递归式的解，并使用**数学归纳法**证明。（10）

#### 5.给定递归式： $T(n) = 2T\left(\frac{n}{2}\right) + n ^2$，使用**递归展开法**求解。（10）

#### 6.给定递归式： $T(n) = 4T(\frac{n}{2}) + \frac{n^2}{\log n}$，使用**递归展开法**求解。（10）

#### 7.读程序，分析代码求出$T_1, T_\infty$，求并行度。(10)
```c
cilk_for ( int i = 0; i < N; ++i)
	for (int j = 0; j < N; ++j)
 		A[i][j]++;
```

#### 8.读程序，分析代码求出$T_1, T_\infty$，求并行度。(15)
```c
// Floyd-Warshall 算法
for (int k = 0; k < N ; ++ k )
	cilk_for (int i = 0; i < N ; ++ i )
		cilk_for (int j = 0; j < N ; ++ j )
			d[i][j] = min(d[i][j] ,d[i][k] + d[k][j]);
```

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
