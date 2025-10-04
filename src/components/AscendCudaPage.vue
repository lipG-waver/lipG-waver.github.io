<template>
  <div class="article-page">
    <article class="article">
      <header class="header">
        <h1>从CUDA到AscendC：编程视角的转变</h1>
        <p class="subtitle">理解昇腾AI处理器的编程范式</p>
      </header>

      <div class="content">
        <!-- CUDA Section -->
        <section>
          <h2>CUDA：线程视角编程</h2>
          
          <div class="code-example">
            <p class="code-title">CUDA Kernel 示例</p>
            <pre><code>__global__ void vectorAdd(const float *A, const float *B, 
                          float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i &lt; numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}</code></pre>
          </div>

          <div class="info-block">
            <h3>CUDA层次结构解析</h3>
            <p><strong>i</strong> = blockDim.x × blockIdx.x + threadIdx.x</p>
            <ul>
              <li>这是一个索引方式，用于计算线程在全局中的唯一索引。什么是线程？什么是块？不妨比喻为小弟和大哥。出现一个计算任务以后，先想到的是把任务布置给小弟（线程）。然而我们名单上管理的都是大哥（块），每个大哥手下都有小弟，所以我们计算哪个大哥的哪个小弟做哪个任务，就是上面这一行式子。</li>
              <li><strong>Block（块，也是大哥）</strong>：管理一组线程的单元</li>
              <li><strong>Thread（线程，也是小弟）</strong>：实际执行任务的最小单元</li>
              <li><strong>blockDim</strong>：每个Block中的线程数（每个大哥带多少小弟）</li>
              <li><strong>blockIdx</strong>：Block的索引（第几个大哥）</li>
              <li><strong>threadIdx</strong>：线程在Block中的索引（小弟在大哥名册上的序号）</li>
            </ul>
            
            <p class="example"><strong>示例：</strong>假设有10个Block，每个Block有10个线程（blockDim.x=10）<br/>
            当 <code>i=66</code> 时 → blockIdx.x=6（第7个Block），threadIdx.x=6（第7个线程）</p>
          </div>

          <p class="note"><strong>注意：</strong>程序员按线程视角编写，但实际执行按Warp（线程束，32个线程）调度。这是后话，目前从线程视角理解即可。</p>
        </section>

        <!-- AscendC Section -->
        <section>
          <h2>AscendC：数据块视角编程</h2>

          <div class="info-block">
            <h3>设计哲学</h3>
            <p>写这些内容的时候，突然想到了华为的一个管理理念：华为很少开除最底层员工，一般优化的都是领导。因为华为认为要干不好，都是领导的锅。没有一个好领导，再好的员工都发挥不出威力。
            </p>
            <p><strong>昇腾AscendC是按照数据块视角编写，从领导的编程视角来看的。</strong></p>
            <p>这是什么含义呢？</p>
            <p>Ascend芯片内部对于几个数据的同时操作（同一时钟周期）提供了统一的简单算子。比如，加法、减法、乘法、除法这些。内部如何实现的，不得而知。</p>
            <p>只要你能将这些你希望处理数据按照昇腾需要的形状填入，那昇腾就能为你做高效计算。相比之下，英伟达的Cuda编程就不需要你有什么通用的形状，因为Cuda处理的往往是通用计算，而非形状在大多时候都是可预测的神经网络计算。</p>
            <p>让我们通过实例来进行理解。</p>
          </div>

          <div class="code-example">
            <p class="code-title">AscendC 初始化示例</p>
            <pre><code>__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
  xGm.SetGlobalBuffer((__gm__ half *)x + 
        BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    
    pipe.InitBuffer(inQueueX, BUFFER_NUM, 
        TILE_LENGTH * sizeof(half));
}</code></pre>
          </div>

          <ul>
            <li>一开始的时候，数据在内存条(DDR)上。经过了SetGlobalBuffer这一句，数据被搬运到了NPU内部的全局缓存上。</li>
            <li>还是用大哥和小弟的思路进行理解。每位大哥会被分配到要处理一定的数据,也就是BLOCK_LENGTH的数据。每个人从哪里开始处理数据呢？就是初始位置加上每个大哥处理的位置*这个大哥的序号。比如初始位置是100，每个大哥处理100个，这个大哥的序号是8，那第0个大哥从100处理到199，8号的处理序号就是900-999.当然了，现在这一步是声明位置，还没涉及到直接的搬运处理。</li>
            </ul>
    <ul>
              <div class="code-example">
            <pre><code>

  xGm.SetGlobalBuffer((__gm__ half *)x + 
        BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
    
    pipe.InitBuffer(inQueueX, BUFFER_NUM, 
        TILE_LENGTH * sizeof(half));
</code></pre>
          </div>

            <li><strong>GetBlockIdx()</strong>：告诉你这是第几个Core（第几位大哥）</li>
            <li><strong>BLOCK_LENGTH</strong>：每个Core处理多少数据</li>
            <li>在pipe.InitBuffer这一行为什么又出现了TILE_LENGTH呢？每个大哥也不可能一下子处理所有数据，具体原因在下面会写到。如果先接受这个观念，那你就会接受要先分块。每一块再加载到队列中，这时候就有直接搬运了。</li>
          </ul>

          <p class="formula">
            <strong>数据分配：</strong><code>Core数量 × 每个Core处理的数据量 = 总数据量</code><br/>
            <span class="example">示例：USE_CORE_NUM = 8, BLOCK_LENGTH = 2048 → 总数据量 = 16,384</span>
          </p>
        </section>

        <!-- Pipeline Section -->
        <section>
          <h2>为什么要分批处理？流水线优化</h2>

          <div class="warning">
            <p><strong>问题：能否一次性处理2048个数据？</strong></p>
            <p>答案：硬件上可以，但效率极低！</p>
          </div>

          <h3>原因1：Local Memory装不下</h3>
          <ul>
            <li>昇腾Local Memory只有256-512KB</li>
            <li>2048个half数据 ≈ 4KB</li>
            <li>双缓冲：4KB × 3(x,y,z) × 2 = 24KB（还能接受）</li>
            <li>但是很多复杂算子（如卷积）可能需要几百KB → 装不下</li>
            <li>一次处理一定量的数据效率比较高。</li>
          </ul>

          <h3>原因2：流水线效率低</h3>
          <div class="info-block">
            <p><strong>一次性处理的时间线：</strong></p>
            <ul>
              <li>前160μs：搬运工干活，AI Core <strong>空转</strong></li>
              <li>中间80μs：AI Core干活，搬运工<strong>空转</strong></li>
              <li>后160μs：搬运工干活，AI Core <strong>空转</strong></li>
            </ul>
            <p><strong>总耗时 = 400μs</strong></p>
          </div>

          <div class="success">
            <h3>✓ 优化方案：分批+流水线</h3>
            
            <p><strong>分成8批，每批256个（TILE_LENGTH=256），配合双缓冲：</strong></p>
            <ul>
              <li>搬入引擎、AI Core、搬出引擎可以"三管齐下"</li>
              <li>搬入第2批时，第1批在计算</li>
              <li>计算第2批时，第1批在搬出、第3批在搬入</li>
            </ul>

            <div class="code-example">
              <p class="code-title">流水线时间线：</p>
              <pre>批次1: [搬入] → [计算] → [搬出]
批次2:       [搬入] → [计算] → [搬出]
批次3:             [搬入] → [计算] → [搬出]
...</pre>
            </div>

            <p class="result"><strong>总耗时 ≈ 100μs（省了75%的时间！🚀）</strong></p>

            <p class="note"><strong>关键点：</strong>搬入/计算/搬出是三个独立的硬件单元，互不影响！所以分成小批次能让三个单元都"满负荷运转"</p>
          </div>

          <div class="info-block">
            <h4>为什么是双缓冲而非三缓冲？</h4>
            <p>由于只有一个AI Core，计算步骤注定只能串行。所以只需把操作分为两类：<strong>加载（搬入/搬出）</strong> 与 <strong>计算</strong></p>
            <p><strong>→ TILE_NUM=8、BUFFER_NUM=2 是最优配置</strong></p>
            <p>要理解BUFFER_NUM = 2, 是因为只有一个计算单元（我们是给每个计算单元进行编程，所以每个计算单元中只有一个），所以BUFFER_NUM 不可能是3，只能是2。</p>
          </div>
        </section>

        <!-- Buffer Management -->
        <section>
          <h2>缓冲区管理机制</h2>

          <div class="code-example">
            <p class="code-title">InitBuffer</p>
            <pre><code>pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));</code></pre>
            <p>生成BUFFER_NUM个队列（2个），每次长度为TILE_LENGTH，供数据反复填入移出</p>
          </div>

          <p><strong>处理总量：</strong>Process需完成 <code>TILE_NUM × BUFFER_NUM</code> 次操作<br/>
          2个buffer × 8个tile = 16块数据</p>

          <h3>数据处理流程</h3>
          
          <ol class="flow-steps">
            <li>
              <strong>分配本地张量</strong>
              <pre><code>LocalTensor&lt;half&gt; xLocal = inQueueX.AllocTensor&lt;half&gt;();</code></pre>
            </li>
            <li>
              <strong>数据拷贝</strong>
              <pre><code>DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);</code></pre>
            </li>
            <li>
              <strong>入队</strong>
              <pre><code>inQueueX.EnQue(xLocal);</code></pre>
              <p class="small">CopyIn函数结束，xLocal销毁</p>
            </li>
            <li>
              <strong>计算中出队</strong>
              <pre><code>xLocal = inQueueX.DeQue();</code></pre>
              <p class="small">在Compute中使用，获取新的xLocal</p>
            </li>
            <li>
              <strong>释放张量</strong>
              <pre><code>inQueueX.FreeTensor(xLocal);</code></pre>
              <p class="small">Add操作完成后及时释放</p>
            </li>
          </ol>

          <div class="info-block">
            <h4>为什么先入队再在Compute中出队？</h4>
            <p>因为CopyIn和Compute中的xLocal不是同一个实例！CopyIn中入队后xLocal销毁，Compute中DeQue获得新的xLocal。队列内容在整个函数周期都存在，计算后需要释放资源，并将输出结果Z入队。</p>
          </div>
        </section>

        <!-- Summary -->
        <section class="summary">
          <h2>核心要点总结</h2>
          <ul>
            <li><strong>CUDA范式：</strong>线程视角，程序员管理每个线程的执行逻辑</li>
            <li><strong>AscendC范式：</strong>数据块视角，从整体数据流的角度组织计算</li>
            <li><strong>流水线优化：</strong>分批处理+双缓冲，让硬件单元满负荷运转</li>
            <li><strong>队列管理：</strong>精细的入队/出队/释放机制确保资源高效利用</li>
          </ul>
        </section>
      </div>

      <footer class="footer">
        <p>理解编程范式的转变，掌握AI芯片编程的核心思想</p>
      </footer>
    </article>
  </div>
</template>

<script>
export default {
  name: 'AscendCudaPage'
}
</script>

<style scoped>
/* 基础布局 */
.article-page {
  min-height: 100vh;
  background: #f5f7fa;
  padding: 2rem 1rem;
}

.article {
  max-width: 900px;
  margin: 0 auto;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.header {
  background: linear-gradient(135deg, #1976d2, #283593);
  color: white;
  padding: 2rem;
  border-radius: 8px 8px 0 0;
}

.header h1 {
  margin: 0 0 0.5rem 0;
  font-size: 1.8rem;
}

.subtitle {
  margin: 0;
  opacity: 0.9;
  font-size: 0.9rem;
}

.content {
  padding: 2rem;
  line-height: 1.8;
}

/* 章节标题 */
section {
  margin-bottom: 2.5rem;
}

h2 {
  color: #1976d2;
  border-bottom: 2px solid #1976d2;
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
}

h3 {
  color: #333;
  margin: 1.5rem 0 0.75rem 0;
  font-size: 1.1rem;
}

h4 {
  color: #555;
  margin: 1rem 0 0.5rem 0;
  font-size: 1rem;
}

/* 代码块 */
.code-example {
  background: #f5f5f5;
  border-left: 3px solid #1976d2;
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 4px;
}

.code-title {
  color: #666;
  font-size: 0.85rem;
  margin: 0 0 0.5rem 0;
  font-family: monospace;
}

pre {
  background: #263238;
  color: #66ff66;
  padding: 1rem;
  border-radius: 4px;
  overflow-x: auto;
  margin: 0.5rem 0;
}

code {
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 0.9em;
}

p code, li code {
  background: #f0f0f0;
  padding: 2px 6px;
  border-radius: 3px;
  color: #c7254e;
  font-size: 0.9em;
}

/* 信息框 */
.info-block {
  background: #e3f2fd;
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 4px;
  border-left: 3px solid #1976d2;
}

.warning {
  background: #ffebee;
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 4px;
  border-left: 3px solid #c62828;
}

.success {
  background: #e8f5e9;
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 4px;
  border-left: 3px solid #2e7d32;
}

.note {
  background: #fff3e0;
  padding: 0.75rem;
  margin: 1rem 0;
  border-radius: 4px;
  border-left: 3px solid #f57c00;
  font-size: 0.95rem;
}

/* 列表 */
ul, ol {
  margin: 0.5rem 0;
  padding-left: 2rem;
}

li {
  margin: 0.5rem 0;
}

.flow-steps {
  counter-reset: step-counter;
  list-style: none;
  padding-left: 0;
}

.flow-steps li {
  counter-increment: step-counter;
  position: relative;
  padding-left: 2.5rem;
  margin: 1.5rem 0;
}

.flow-steps li::before {
  content: counter(step-counter);
  position: absolute;
  left: 0;
  top: 0;
  background: #1976d2;
  color: white;
  width: 1.8rem;
  height: 1.8rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 0.9rem;
}

/* 辅助样式 */
.example {
  font-size: 0.9rem;
  color: #555;
  margin-top: 0.5rem;
}

.formula {
  background: #f5f5f5;
  padding: 0.75rem;
  margin: 1rem 0;
  border-radius: 4px;
}

.result {
  background: #2e7d32;
  color: white;
  padding: 0.75rem;
  border-radius: 4px;
  text-align: center;
  margin: 1rem 0;
}

.small {
  font-size: 0.85rem;
  color: #666;
  font-style: italic;
  margin-top: 0.25rem;
}

.summary {
  background: #f5f5f5;
  padding: 1.5rem;
  border-radius: 4px;
}

.footer {
  background: #f5f5f5;
  padding: 1rem 2rem;
  text-align: center;
  border-radius: 0 0 8px 8px;
  border-top: 1px solid #e0e0e0;
  color: #666;
  font-size: 0.9rem;
}

.footer p {
  margin: 0;
}

/* 响应式 */
@media (max-width: 768px) {
  .article-page {
    padding: 1rem 0.5rem;
  }
  
  .header {
    padding: 1.5rem;
  }
  
  .header h1 {
    font-size: 1.4rem;
  }
  
  .content {
    padding: 1.5rem;
  }
  
  pre {
    font-size: 0.8rem;
  }
}
</style>