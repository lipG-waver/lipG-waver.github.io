<template>
  <div class="dl-blog-page">
    <h1>神经网络</h1>
    <div class="content">
      <p>
        在我就读的加州大学伯克利分校神经网络课程中，由 Anant Sahai 教授授课。他采取了独特的教学方式：课堂上不允许使用任何电子设备，以鼓励学生专注于思考和板书内容。这种传统式的授课风格起初让我有些意外，但也促使我更认真地在课下整理和消化课程内容。
      </p>

      <p>
        就我个人的理解而言，现代神经网络可以被视为"伪装成参数方法的非参数模型"。尽管从形式上看，神经网络是由固定数量的参数构成（参数化模型），但在实际应用中，这些模型往往包含数百万甚至数十亿个参数，远远超过训练样本的规模，构成了所谓的<em>过参数化</em>（over-parameterization）设置。
      </p>

      <p>
        在这种设置下，神经网络表现出非参数模型的诸多特性。例如，它们具备极强的拟合能力，能够逼近任意复杂的函数（依据万能逼近定理），并在训练集上达到零误差。同时令人惊讶的是，它们在测试集上的泛化能力依然良好。这种现象在经典统计学习理论中较难解释，但近年来已有许多研究从"隐式正则化"、"神经切线核"（NTK）等角度尝试建模这种行为。
      </p>

      <p>
        对于没有接触过神经网络的读者，可以将其理解为一种"超级非线性"的模型：只要神经网络的层数足够深、参数足够多，并且训练数据足够充分，它几乎可以逼近任何函数。这使其在图像识别、自然语言处理等高维复杂任务中具备远超线性模型的表达能力。
      </p>
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
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

h1 {
  color: #333;
  text-align: center;
  margin-bottom: 30px;
}

h2 {
  color: #444;
  margin-top: 25px;
}

.date {
  color: #666;
  font-style: italic;
  margin-bottom: 15px;
}

.article-item {
  margin-bottom: 40px;
}

.article-item:last-child {
  margin-bottom: 0;
}

p {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 15px;
}

ul, ol {
  margin-bottom: 20px;
  padding-left: 20px;
}

li {
  margin-bottom: 10px;
  line-height: 1.5;
}

strong {
  color: #333;
}

/* 数学公式样式 */
p:has(img) {
  text-align: center;
}
</style>