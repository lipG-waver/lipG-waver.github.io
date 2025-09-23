<template>
  <div class="dl-blog-page">
    <h1 class="page-title">神经网络</h1>
    <div class="content">
<p>在我修读的这门神经网络课程中，Anant Sahai教授的教学方式让我深刻体会到，真正理解复杂概念需要绝对的专注。他课堂上禁止使用电子设备的规定，起初是一种挑战，但后来却成为了一种馈赠。它迫使我在当下完全投入于逻辑的推演和板书的脉络，而将整理、消化和反思的工作留到课后。这个过程让我对神经网络的本质有了更深的领悟。</p>

<p>在我看来，现代神经网络可以被看作是一种“伪装成参数方法的非参数模型”。从形式上看，一个网络的结构是由固定数量的权重和偏置定义的，这符合参数化模型的特征。然而，当我们步入实践领域，特别是在深度学习中，我们构建的模型往往拥有数百万甚至数十亿个参数，其规模远远超过我们手中的训练样本数量。这种过参数化的设置，使得它的行为更接近于非参数模型。</p>

<p>在这种过参数化的 regime 下，神经网络展现出了令人惊异的特性。依据万能逼近定理，它们拥有近乎无限的拟合能力，可以在训练集上达到近乎完美的精度。但更神奇的是，它们并不会像经典理论预言的那样发生灾难性的过拟合，反而在未见过的测试数据上表现出良好的泛化性能。这种看似矛盾的现象，正是当前研究的热点，试图从“隐式正则化”或“神经切线核”等新视角来理解。</p>

<p>如果要用一个直观的比喻向初学者解释，我会说神经网络是一种“超级非线性”模型。只要它的结构足够复杂（层数深、参数多），并有充足的数据进行训练，它就能学会数据中极其复杂的模式和关系，从而在图像识别、自然语言处理等任务上远超传统的线性模型。</p>

<p>课程的内容是系统性的，它从最基础的生物学灵感开始。我们首先了解了 McCulloch-Pitts 神经元模型如何将生物神经元抽象为一个简单的计算单元，然后是感知机模型，它展示了如何通过简单的学习规则让机器进行模式识别。但感知机的局限性——无法解决异或问题——清晰地指出了单层网络的不足，从而引出了对“深度”的追求。</p>

<p>而整个课程的灵魂，无疑是反向传播算法。它完美地解决了多层网络如何学习的问题。我们深入研究了损失函数如何定义错误，梯度下降如何指引优化方向，以及链式法则是如何被巧妙地用于将误差从输出层一层层地反向传播回去，从而计算出每个参数应有的更新量。理解反向传播，就像是掌握了驱动这个复杂系统运转的引擎。</p>

<p>随后，课程重点介绍了两种最重要的神经网络架构。卷积神经网络专为图像处理而生，它的核心思想——局部连接和参数共享——极大地减少了参数数量，并让网络具备了平移不变性。我们从LeNet讲到ResNet，看到了深度如何一步步突破极限。而循环神经网络则专为序列数据设计，通过其循环结构赋予网络“记忆”。LSTM和GRU中精巧的门控机制，有效地解决了长期依赖问题，成为了处理语言和时序数据的基石。</p>

<p>最后，视野会扩展到更现代的架构，如基于注意力机制的Transformer，它如何彻底改变了自然语言处理；以及生成式模型如GAN和扩散模型，它们如何获得了令人惊叹的创造能力。</p>

<p>总而言之，这门课教会我的远不止是一系列算法和模型。它更是一种思维方式，一种理解如何通过分层抽象和梯度优化，从数据中构建复杂智能系统的思维框架。Sahai教授强调的专注与反思，让我得以真正沉浸其中，去体会这些概念背后的深刻与优美。</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DlBlogPage'
}
</script>

<style scoped>
.dl-blog-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  text-align: left;
}

.content {
  background-color: #fff;
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
}

.page-title {
  color: #2c3e50;
  text-align: center;
  margin-bottom: 30px;
  font-size: 2rem;
  position: relative;
  padding-bottom: 10px;
}

.page-title::after {
  content: '';
  display: block;
  width: 60px;
  height: 3px;
  background: linear-gradient(90deg, #3498db, #2c3e50);
  margin: 10px auto 0;
  border-radius: 3px;
}

h2 {
  color: #2c3e50;
  margin-top: 25px;
  font-size: 1.5rem;
  position: relative;
  padding-bottom: 8px;
}

h2::after {
  content: '';
  display: block;
  width: 40px;
  height: 2px;
  background: linear-gradient(90deg, #3498db, #2c3e50);
  margin: 8px 0 0 0;
  border-radius: 2px;
}

.date {
  color: #2980b9;
  font-style: italic;
  margin-bottom: 15px;
  font-weight: 500;
}

.article-item {
  margin-bottom: 40px;
}

.article-item:last-child {
  margin-bottom: 0;
}

p {
  font-size: 1.05rem;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 15px;
  text-align: justify;
}

ul, ol {
  margin-bottom: 20px;
  padding-left: 20px;
}

li {
  margin-bottom: 10px;
  line-height: 1.6;
  color: #34495e;
}

strong {
  color: #2c3e50;
  font-weight: 600;
}

/* 数学公式样式 */
p:has(img) {
  text-align: center;
}

@media (max-width: 768px) {
  .dl-blog-page {
    padding: 10px;
  }
  
  .content {
    padding: 20px;
  }
  
  .page-title {
    font-size: 1.7rem;
  }
  
  h2 {
    font-size: 1.3rem;
  }
  
  p {
    font-size: 1rem;
    line-height: 1.7;
  }
}
</style>