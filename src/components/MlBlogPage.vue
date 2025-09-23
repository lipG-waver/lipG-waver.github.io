<template>
  <div class="ml-blog-page">
    <h1>机器学习</h1>
    <div class="content">
      <p>
        我在伯克利学习的机器学习课程由 Jonathan Shewchuk 教授授课，他以清晰的讲解和风趣的风格而著称，是一位极具启发性的老师。
      </p>

      <p>
        这门课程主要聚焦于经典机器学习方法的系统训练，同时也介绍了部分现代方法。课程从支持向量机（SVM）入手，随后讲解了高斯判别分析（GDA）、线性回归与逻辑回归、岭回归与 Lasso 等正则化方法，以及树模型（包括决策树、随机森林、AdaBoost）等。后半部分内容涉及神经网络基础以及无监督学习方法，如主成分分析（PCA）和聚类算法。
      </p>

      <p>
        由于神经网络在现代机器学习中占据核心地位，我还专门选修了另一门聚焦神经网络的课程。相关内容可见"神经网络"一节。
      </p>

      <p>
        在这门课中，我深刻体会到机器学习与经济学研究方法的差异。机器学习强调模型的预测能力与泛化性能，因此常采用交叉验证（Cross Validation）等方式对模型进行评估，并引入正则化手段如 Weight Decay（例如 Lasso 和 Ridge）来控制模型复杂度。而传统经济学更关注估计结果的无偏性，强调结构模型的因果解释，因此往往不采用交叉验证来选择模型。这种对预测精度关注的差异，也部分解释了为何经济学在实际预测中的表现常与结果存在偏差。
      </p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MlBlogPage'
}
</script>

<style scoped>
.ml-blog-page {
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
</style>