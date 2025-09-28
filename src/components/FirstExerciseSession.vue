<template>
  <div class="page-container">
    <div class="content-wrapper">
      <div class="bg-white rounded-lg shadow-lg p-6">
        <h1 class="text-3xl font-bold text-gray-800 mb-6 text-center">截面与面板数据分析第一次习题课</h1>
        
        <div class="space-y-8">
          <!-- 问题1：迭代期望公式 -->
          <div class="bg-gray-50 rounded-lg p-6 border-l-4 border-green-500">
            <h3 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span class="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2">1</span>
              迭代期望公式
            </h3>
            
            <div class="space-y-4 text-left">
              <p class="text-gray-700 leading-relaxed">
                证明迭代期望公式（Law of Iterated Expectation）：
              </p>
              
              <div class="bg-white rounded p-4 shadow-sm text-center">
                $$
                \mathbb{E}[X] = \mathbb{E}[\,\mathbb{E}[X \mid Y]\,]
                $$
              </div>
            </div>
          </div>

          <!-- 1.1 离散随机变量证明 -->
          <div class="bg-gray-50 rounded-lg p-6 border-l-4 border-blue-500">
            <h3 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span class="bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2">1.1</span>
              离散随机变量证明
            </h3>
            
            <div class="bg-white rounded p-4 shadow-sm">
              $$
              \begin{aligned}
              \mathbb{E}[\,\mathbb{E}[X \mid Y]\,] &= \sum_y \mathbb{E}[X \mid Y = y] \cdot P(Y = y) \\
              &= \sum_y \left( \sum_x x \cdot P(X = x \mid Y = y) \right) \cdot P(Y = y) \\
              &= \sum_y \sum_x x \cdot P(X = x \mid Y = y) \cdot P(Y = y) \\
              &= \sum_y \sum_x x \cdot P(X = x, Y = y) \\
              &= \sum_x x \cdot \left( \sum_y P(X = x, Y = y) \right) \\
              &= \sum_x x \cdot P(X = x) \\
              &= \mathbb{E}[X]
              \end{aligned}
              $$
            </div>
          </div>

          <!-- 1.2 连续随机变量证明 -->
          <div class="bg-gray-50 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span class="bg-purple-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2">1.2</span>
              连续随机变量证明
            </h3>
            
            <div class="bg-white rounded p-4 shadow-sm">
              $$
              \begin{aligned}
              \mathbb{E}[\,\mathbb{E}[X \mid Y]\,] &= \int_{-\infty}^{\infty} \mathbb{E}[X \mid Y = y] \cdot f_Y(y) \, dy \\
              &= \int_{-\infty}^{\infty} \left( \int_{-\infty}^{\infty} x \cdot f_{X|Y}(x|y) \, dx \right) \cdot f_Y(y) \, dy \\
              &= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} x \cdot f_{X|Y}(x|y) f_Y(y) \, dx \, dy \\
              &= \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} x \cdot f_{X,Y}(x,y) \, dx \, dy \\
              &= \int_{-\infty}^{\infty} x \cdot \left( \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy \right) dx \\
              &= \int_{-\infty}^{\infty} x \cdot f_X(x) \, dx \\
              &= \mathbb{E}[X]
              \end{aligned}
              $$
            </div>

            <!-- 符号说明 -->
            <div class="mt-6 bg-white rounded p-4 shadow-sm text-left">
              <h4 class="font-semibold text-gray-700 mb-3">符号说明：</h4>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <div class="flex items-center">
                  <span class="bg-gray-200 rounded px-2 py-1 mr-2 font-mono">$f_Y(y)$</span>
                  <span class="text-gray-600">Y 的边缘概率密度函数</span>
                </div>
                <div class="flex items-center">
                  <span class="bg-gray-200 rounded px-2 py-1 mr-2 font-mono">$f_{X|Y}(x|y)$</span>
                  <span class="text-gray-600">给定 Y=y 时 X 的条件概率密度函数</span>
                </div>
                <div class="flex items-center">
                  <span class="bg-gray-200 rounded px-2 py-1 mr-2 font-mono">$f_{X,Y}(x,y)$</span>
                  <span class="text-gray-600">X 和 Y 的联合概率密度函数</span>
                </div>
                <div class="flex items-center">
                  <span class="bg-gray-200 rounded px-2 py-1 mr-2 font-mono">$f_X(x)$</span>
                  <span class="text-gray-600">X 的边缘概率密度函数</span>
                </div>
              </div>
            </div>
          </div>

          <!-- 1.3 积分顺序交换的数学依据 -->
          <div class="bg-blue-50 rounded-lg p-6 border-l-4 border-indigo-500">
            <h3 class="text-lg font-semibold text-blue-800 mb-4 flex items-center">
              <span class="bg-indigo-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2">1.3</span>
              积分顺序交换的数学依据
            </h3>
            
            <div class="space-y-4 text-left">
              <p class="text-blue-700 leading-relaxed">
                在连续随机变量的证明中，从第4行到第5行的积分顺序交换是合法的，这基于以下数学定理：
              </p>

              <div class="bg-white rounded p-4 shadow-sm">
                <h4 class="font-semibold text-purple-600 mb-2">富比尼定理（Fubini's Theorem）</h4>
                <p class="text-gray-700 mb-3">
                  如果函数 $|x \cdot f_{X,Y}(x,y)|$ 在 $\mathbb{R}^2$ 上可积，那么可以交换积分的顺序。
                </p>
                
                <h4 class="font-semibold text-purple-600 mb-2">托内利定理（Tonelli's Theorem）</h4>
                <p class="text-gray-700">
                  对于非负函数，即使不可积，也可以交换积分顺序。由于 $f_{X,Y}(x,y) \geq 0$，我们也可以应用托内利定理。
                </p>
              </div>

              <div class="bg-white rounded p-4 shadow-sm">
                <h4 class="font-semibold text-gray-700 mb-2">适用条件：</h4>
                <ul class="list-disc list-inside space-y-1 text-gray-600">
                  <li>$f_{X,Y}(x,y)$ 是联合概率密度函数，满足 $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx \, dy = 1$</li>
                  <li>若 $\mathbb{E}[|X|] \lt \infty$（即 X 的绝对期望存在），则 $|x \cdot f_{X,Y}(x,y)|$ 在 $\mathbb{R}^2$ 上可积</li>
                </ul>
              </div>

              <p class="text-blue-700 font-medium">
                因此，在 $\mathbb{E}[|X|] \lt \infty$ 的条件下，积分顺序的交换在概率论中是严格合法的。
              </p>
            </div>
          </div>
        </div>

        <!-- 可视化图表 -->
        <div class="mt-8 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-6 border border-gray-200">
          <h3 class="text-lg font-semibold text-gray-800 mb-4 text-center">迭代期望公式的直观解释</h3>
          <canvas ref="canvas" width="600" height="400" class="mx-auto border border-gray-300 rounded shadow-sm"></canvas>
        </div>

        <!-- 经典反例：基于有理数/无理数的随机变量 -->
        <div class="mt-8 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-6 border border-amber-200">
          <h3 class="text-lg font-semibold text-amber-800 mb-4 text-center">经典反例：有理数/无理数随机变量</h3>
          
          <div class="space-y-4 text-left">
            <p class="text-amber-700 leading-relaxed">
              虽然我们已经证明了迭代期望公式在一般条件下成立，但需要注意的是，如果不满足相应的数学条件，该公式可能不成立。以下是一个经典反例：
            </p>

            <!-- 第1步：设置场景 -->
            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-amber-700 mb-2">第1步：设置场景</h4>
              <p class="text-gray-700 mb-3">
                假设我们有一个非常奇怪的随机实验。在单位区间 $[0,1]$ 上按照均匀分布随机选取一个点 $\omega$。定义两个随机变量：
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3">
                <p class="font-mono text-sm">
                  $Y(\omega) = \omega$ （就是选取的点本身）
                </p>
                <p class="font-mono text-sm mt-2">
                  $$X(\omega) = 
                  \begin{cases}
                    1, & \text{如果 } \omega \text{ 是有理数} \\
                    0, & \text{如果 } \omega \text{ 是无理数}
                  \end{cases}$$
                </p>
              </div>
              
              <p class="text-gray-700">
                已知：$P(Y=y) = 0$ 对于任意 $y \in [0,1]$，因为是在连续区间上取单点。
              </p>
            </div>

            <!-- 第2步：引导学生进行"直觉"推导 -->
            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-amber-700 mb-2">第2步："直觉"推导（错误的推导）</h4>
              
              <p class="text-gray-700 mb-3">
                现在，我们来计算 $\mathbb{E}[\,\mathbb{E}[X \mid Y]\,]$。
              </p>
              
              <h5 class="font-semibold text-gray-700 mb-2">计算内层条件期望 $\mathbb{E}[X \mid Y = y]$：</h5>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mb-3">
                <li>"在给定 $Y = y$ 的条件下，$\omega$ 被固定为 $y$。"</li>
                <li>那么，$X$ 的值也就完全确定了：</li>
                <li>如果 $y$ 是有理数，则 $X = 1$，所以 $\mathbb{E}[X \mid Y = y] = 1$。</li>
                <li>如果 $y$ 是无理数，则 $X = 0$，所以 $\mathbb{E}[X \mid Y = y] = 0$。</li>
              </ul>
              
              <p class="text-gray-700 mb-3">
                因此，我们可以定义一个函数：
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3 text-center">
                $$
                h(y) = \mathbb{E}[X \mid Y = y] = 
                \begin{cases}
                  1, & \text{如果 } y \text{ 是有理数} \\
                  0, & \text{如果 } y \text{ 是无理数}
                \end{cases}
                $$
              </div>
              
              <h5 class="font-semibold text-gray-700 mb-2">计算外层期望 $\mathbb{E}[h(Y)]$：</h5>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mb-3">
                <li>$\mathbb{E}[\,\mathbb{E}[X \mid Y]\,] = \mathbb{E}[h(Y)]$。</li>
                <li>由于 $Y$ 是 $[0,1]$ 上的均匀分布，这个期望就是函数 $h(y)$ 在 $[0,1]$ 上的积分。</li>
                <li>$h(y)$ 是一个狄利克雷函数：在有理数点上为1，在无理数点上为0。</li>
                <li>在黎曼积分的意义下，这个函数不可积，但如果我们强行用概率的"加权平均"来想，可能会有：因为有理数测度为0，无理数测度为1，所以这个"平均值"应该是 $1 \times 0 + 0 \times 1 = 0$。</li>
                <li>因此，他们可能会得出结论：$\mathbb{E}[\,\mathbb{E}[X \mid Y]\,] = 0$。</li>
              </ul>
              
              <h5 class="font-semibold text-gray-700 mb-2">计算真实的 $\mathbb{E}[X]$：</h5>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mb-3">
                <li>$X$ 以概率0取值为1（因为有理数集是零测集），以概率1取值为0。</li>
                <li>所以，$\mathbb{E}[X] = 1 \times 0 + 0 \times 1 = 0$。</li>
              </ul>
              
              <p class="text-gray-700 font-medium">
                表面上的"一致"：看起来，$\mathbb{E}[\,\mathbb{E}[X \mid Y]\,] = 0 = \mathbb{E}[X]$。迭代期望定律似乎成立了？别急，我们还没利用这个例子的全部威力。
              </p>
            </div>

            <!-- 第3步：揭示矛盾 -->
            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-amber-700 mb-2">第3步：揭示矛盾（教学的关键转折点）</h4>
              
              <p class="text-gray-700 mb-3">
                现在，我们调换角色。考虑 $\mathbb{E}[\,\mathbb{E}[Y \mid X]\,]$。
              </p>
              
              <h5 class="font-semibold text-gray-700 mb-2">计算内层条件期望 $\mathbb{E}[Y \mid X]$：</h5>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mb-3">
                <li>当 $X = 1$：这意味着 $\omega$ 是有理数。有理数在 $[0,1]$ 上是稠密的但可数。学生可能会困惑："给定 $\omega$ 是有理数，$Y$ 的期望是多少？" 他们无法给出一个确定的数值，因为均匀分布在一个可数集上没有定义（不是连续分布）。</li>
                <li>当 $X = 0$：这意味着 $\omega$ 是无理数。同样，均匀分布在无理数集上也无法用常规方法定义期望。</li>
              </ul>
              
              <p class="text-gray-700 font-medium">
                陷入困境：学生发现他们根本无法用初等的方法定义 $\mathbb{E}[Y \mid X = x]$！这个表达式变得没有意义。
              </p>
              
              <p class="text-gray-700 font-medium">
                矛盾出现：迭代期望定律 $\mathbb{E}[\,\mathbb{E}[Y \mid X]\,] = \mathbb{E}[Y]$ 的右边 $\mathbb{E}[Y] = 0.5$ 是明确可知的。但左边却无法计算。这强烈地暗示了我们关于条件期望的直觉概念在此崩溃了。
              </p>
            </div>

            <!-- 第4步：引出教训和严格理论 -->
            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-amber-700 mb-2">第4步：引出教训和严格理论</h4>
              
              <p class="text-gray-700 mb-3">
                问题的根源：
              </p>
              
              <h5 class="font-semibold text-gray-700 mb-2">问题的核心：</h5>
              <p class="text-gray-700 mb-3">
                当条件事件（如 $\{Y = y\}$ 或 $\{X = 1\}$）的概率为零时，我们熟悉的条件概率公式 $P(A \mid B) = P(A \cap B) / P(B)$ 完全失效。我们之前的推导是建立在沙滩上的。
              </p>
              
              <h5 class="font-semibold text-gray-700 mb-2">测度论的解决方案：</h5>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mb-3">
                <li>在测度论中，条件期望 $\mathbb{E}[X \mid Y]$ 不是关于点 $Y = y$，而是关于 $Y$ 生成的整个σ-代数 $\sigma(Y)$。</li>
                <li>它被定义为一个新的随机变量 $Z$，满足两个条件：
                  <ol class="list-decimal list-inside ml-5 space-y-1">
                    <li>$Z$ 是 $\sigma(Y)$-可测的（意味着它的值只依赖于 $Y$）。</li>
                    <li>对于任意事件 $A \in \sigma(Y)$，有 $\int_A Z \, dP = \int_A X \, dP$。</li>
                  </ol>
                </li>
                <li>在这个框架下，可以严格证明迭代期望定律 $\mathbb{E}[\,\mathbb{E}[X \mid \mathcal{G}]\,] = \mathbb{E}[X]$ 是成立的，其中 $\mathcal{G}$ 是一个σ-代数。</li>
              </ul>
              
              <h5 class="font-semibold text-gray-700 mb-2">分析我们的例子：</h5>
              <ul class="list-disc list-inside space-y-1 text-gray-700">
                <li>在第一个计算 $\mathbb{E}[\,\mathbb{E}[X \mid Y]\,]$ 中，我们歪打正着得到了正确答案，但这只是巧合。因为我们构造的函数 $h(y)$ 恰好满足测度论中条件期望的定义（可以验证它满足上述两个条件）。</li>
                <li>在第二个计算 $\mathbb{E}[\,\mathbb{E}[Y \mid X]\,]$ 中，我们的直觉完全失败，因为它无法处理零概率事件。而测度论可以给出严格的定义。</li>
              </ul>
            </div>
          </div>


        </div>
                <!-- 概率不等式的应用示例 -->
        <div class="mt-8 bg-gradient-to-r from-green-50 to-teal-50 rounded-lg p-6 border border-green-200">
            <h3 class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
              <span class="bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-2">2</span>
              概率论几个公式的估计
            </h3>          
          <div class="space-y-4 text-left">
            <p class="text-green-700 leading-relaxed">
              在实际问题中，我们经常需要估计随机变量取值的概率，但并不知道其确切的分布。这时，概率不等式提供了有力的工具。以下是一个考试分数估计的例子：
            </p>

            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-green-700 mb-2">问题设定</h4>
              <p class="text-gray-700 mb-3">
                假设一次考试的分数 $X$ 满足：
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3 text-center">
                $$
                \mu = \mathbb{E}[X] = 75, \quad \sigma = \sqrt{\mathrm{Var}(X)} = 5
                $$
              </div>
              
              <p class="text-gray-700">
                我们需要估计以下概率：
              </p>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mt-2">
                <li>有多少人分数高于90分？即 $P(X \geq 90)$</li>
                <li>有多少人分数少于60分？即 $P(X \leq 60)$</li>
              </ul>
            </div>

            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-green-700 mb-2">1. 使用Markov不等式</h4>
              <p class="text-gray-700 mb-3">
                Markov不等式适用于非负随机变量，形式为：对于 $X \geq 0$ 和 $a > 0$，
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3 text-center">
                $$
                P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}
                $$
              </div>
              
              <p class="text-gray-700 mb-3">
                为了估计 $P(X \geq 90)$，我们考虑非负随机变量 $Y = |X-75|$（因为分数通常不低于0分）：
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3">
                <p class="text-gray-700 mb-2">
                  $\mathbb{E}[Y] = \mathbb{E}[|X - 75|] = 75 - 60 = 15$
                </p>
                <p class="text-gray-700">
                  $P(X \geq 90) = P(Y \geq 30) \leq \frac{\mathbb{E}[Y]}{30} = \frac{15}{30} = 0.5 = 50\%$
                </p>
              </div>
              
              <p class="text-gray-700">
                这个估计比较宽松，利用的只是一阶矩信息。
              </p>
            </div>

            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-green-700 mb-2">2. 使用切比雪夫不等式</h4>
              <p class="text-gray-700 mb-3">
                切比雪夫不等式利用了随机变量的均值和方差信息，形式为：对于任意 $k > 0$，
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3 text-center">
                $$
                P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}
                $$
              </div>
              
              <p class="text-gray-700 mb-3">
                对于 $P(X \geq 90)$，我们有：
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3">
                <p class="text-gray-700 mb-2">
                  $P(X \geq 90) = P(X - 75 \geq 15) \leq P(|X - 75| \geq 15)$
                </p>
                <p class="text-gray-700">
                  由于 $\sigma = 5$，所以 $k = 15/5 = 3$，因此：
                </p>
                <p class="text-gray-700 mt-2">
                  $P(|X - 75| \geq 15) \leq \frac{1}{3^2} = \frac{1}{9} \approx 11.1\%$
                </p>
              </div>
              
              <p class="text-gray-700 mb-3">
                对于 $P(X \leq 60)$，我们有：
              </p>
              
              <div class="bg-gray-50 rounded p-3">
                <p class="text-gray-700 mb-2">
                  $P(X \leq 60) = P(75 - X \geq 15) \leq P(|X - 75| \geq 15) \leq \frac{1}{9} \approx 11.1\%$
                </p>
              </div>
              
              <p class="text-gray-700 mt-3">
                切比雪夫不等式比Markov不等式更紧，因为它利用了二阶矩信息。
              </p>
            </div>

            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-green-700 mb-2">3. 假设正态分布</h4>
              <p class="text-gray-700 mb-3">
                如果我们进一步假设分数服从正态分布 $X \sim N(75, 5^2)$，则可以得到更精确的估计：
              </p>
              
              <div class="bg-gray-50 rounded p-3 mb-3">
                <p class="text-gray-700 mb-2">
                  $P(X \geq 90) = P\left(\frac{X - 75}{5} \geq \frac{90 - 75}{5}\right) = P(Z \geq 3)$
                </p>
                <p class="text-gray-700">
                  查标准正态分布表得：$P(Z \geq 3) \approx 0.0013 = 0.13\%$
                </p>
              </div>
              
              <div class="bg-gray-50 rounded p-3 mb-3">
                <p class="text-gray-700 mb-2">
                  $P(X \leq 60) = P\left(\frac{X - 75}{5} \leq \frac{60 - 75}{5}\right) = P(Z \leq -3)$
                </p>
                <p class="text-gray-700">
                  查标准正态分布表得：$P(Z \leq -3) \approx 0.0013 = 0.13\%$
                </p>
              </div>
              
              <p class="text-gray-700">
                可以看出，不同方法给出的估计差别很大：
              </p>
              <ul class="list-disc list-inside space-y-1 text-gray-700 mt-2">
                <li>Markov不等式：$\leq 50\%$</li>
                <li>切比雪夫不等式：$\leq 11.1\%$</li>
                <li>正态分布假设：$\approx 0.13\%$</li>
              </ul>
            </div>

            <div class="bg-white rounded p-4 shadow-sm">
              <h4 class="font-semibold text-green-700 mb-2">结论</h4>
              <ul class="list-disc list-inside space-y-2 text-gray-700">
                <li>概率不等式在不知道确切分布时提供了有用的概率上界</li>
                <li>Markov不等式最宽松，只需要一阶矩信息</li>
                <li>切比雪夫不等式更紧，利用了方差信息</li>
                <li>当知道确切分布时（如正态分布），可以得到最精确的结果</li>
                <li>在实际应用中，选择哪种方法取决于我们对问题的了解程度和所需的精度</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
/* global MathJax */
import { onMounted, ref } from "vue";

const canvas = ref(null);

// 更可靠地加载和配置 MathJax
const loadMathJax = () => {
  // 如果 MathJax 已经存在并已准备就绪
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
    return;
  }

  // 配置 MathJax
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
      ignoreHtmlClass: 'tex2jax_ignore'
    },
    startup: {
      pageReady: () => {
        console.log('MathJax is loaded and ready');
        // 页面准备就绪时渲染数学公式
        return MathJax.startup.defaultPageReady().then(() => {
          console.log('MathJax typesetting complete');
        });
      }
    }
  };

  // 创建脚本元素并加载 MathJax
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  script.async = true;
  script.id = 'mathjax-script';
  
  // 当脚本加载完成时触发渲染
  script.onload = () => {
    console.log('MathJax script loaded');
    if (window.MathJax) {
      // 等待 MathJax 完全初始化后再渲染
      window.MathJax.startup.promise.then(() => {
        console.log('MathJax startup complete, rendering math');
        return window.MathJax.typesetPromise();
      }).catch((err) => {
        console.error('MathJax rendering error:', err);
      });
    }
  };
  
  script.onerror = (err) => {
    console.error('Failed to load MathJax:', err);
  };

  document.head.appendChild(script);
};

onMounted(() => {
  // 加载 MathJax
  loadMathJax();

  // Canvas 绘图
  const ctx = canvas.value.getContext("2d");

  // 背景
  ctx.fillStyle = "#f9f9f9";
  ctx.fillRect(0, 0, canvas.value.width, canvas.value.height);

  // 标题
  ctx.fillStyle = "black";
  ctx.font = "18px Arial";
  ctx.fillText("迭代期望公式的直观解释", 180, 40);

  // 绘制随机变量 Y 的条件层次
  ctx.fillStyle = "#4ade80";
  ctx.fillRect(80, 100, 120, 50);
  ctx.fillStyle = "black";
  ctx.fillText("Y=y1", 110, 130);

  ctx.fillStyle = "#60a5fa";
  ctx.fillRect(250, 100, 120, 50);
  ctx.fillStyle = "black";
  ctx.fillText("Y=y2", 280, 130);

  ctx.fillStyle = "#facc15";
  ctx.fillRect(420, 100, 120, 50);
  ctx.fillStyle = "black";
  ctx.fillText("Y=y3", 450, 130);

  // 箭头和 E[X|Y]
  ctx.strokeStyle = "black";
  ctx.beginPath();
  ctx.moveTo(140, 150);
  ctx.lineTo(140, 200);
  ctx.moveTo(310, 150);
  ctx.lineTo(310, 200);
  ctx.moveTo(480, 150);
  ctx.lineTo(480, 200);
  ctx.stroke();

  ctx.fillStyle = "black";
  ctx.fillText("E[X|Y=y1]", 90, 230);
  ctx.fillText("E[X|Y=y2]", 260, 230);
  ctx.fillText("E[X|Y=y3]", 430, 230);

  // 汇总到 E[E[X|Y]]
  ctx.beginPath();
  ctx.moveTo(140, 250);
  ctx.lineTo(300, 300);
  ctx.moveTo(310, 250);
  ctx.lineTo(300, 300);
  ctx.moveTo(480, 250);
  ctx.lineTo(300, 300);
  ctx.stroke();

  ctx.fillStyle = "red";
  ctx.fillText("E[E[X|Y]]", 260, 330);
});
</script>

<style scoped>
.page-container {
  display: flex;
  justify-content: center;
  width: 100%;
  padding: 20px;
  background-color: #f9f9f9;
}

.content-wrapper {
  width: 100%;
  max-width: 900px;
}

canvas {
  display: block;
  margin-top: 20px;
}

@media (max-width: 768px) {
  .page-container {
    padding: 10px;
  }
  
  .content-wrapper {
    max-width: 100%;
  }
}
</style>
