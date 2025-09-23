<template>
  <div class="ml-blog-page">
    <h1 class="page-title">机器学习</h1>
    <div class="content">
    <p> 我在伯克利学习的机器学习课程由 Jonathan Shewchuk 教授主讲，他以其清晰的逻辑、深入浅出的讲解以及幽默风趣的授课风格深受学生喜爱。Shewchuk 教授不仅传授知识，更善于引导学生建立对机器学习本质的理解，是一位极具启发力的教师。 </p>
    <p> 这门课程系统性地介绍了经典机器学习方法，并适度融入了部分现代内容，形成了从理论到实践的知识闭环。课程从支持向量机（SVM）入手，详细讲解了其数学原理与几何直观，帮助学生理解最大间隔分类的思想。随后逐步展开高斯判别分析（GDA）、线性回归与逻辑回归等内容，并引入岭回归（Ridge Regression）和 Lasso 等正则化技术，引导学生思考如何平衡模型复杂度和过拟合问题。在树模型部分，课程不仅讲解了决策树的基本构建方法，还进一步介绍了集成学习技术，如随机森林和 AdaBoost，分析它们如何通过组合多个弱学习器来提升模型稳健性与准确率。 </p>
    <p> 课程后半部分转向神经网络基础与无监督学习，涵盖了主成分分析（PCA）和常见聚类算法（如 K-Means）。通过这部分内容，学生能够理解如何从高维数据中提取有效特征，以及如何发现数据中隐藏的结构模式。 </p>
    <p> 由于认识到神经网络在当今机器学习领域的核心地位，我还额外选修了一门专门讲解神经网络的课程，相关内容将在“神经网络”部分进一步展开。 </p>
    <p> 在这门课程中，我深刻体会到机器学习与传统经济学研究方法之间的显著差异。机器学习将预测准确性和模型泛化能力作为核心目标，因此广泛采用交叉验证等技术评估模型表现，并运用权重衰减（如 Lasso 和 Ridge 正则化）来控制模型复杂度，防止过拟合。相比之下，经济学研究更注重参数估计的无偏性和因果推断的可靠性，通常依赖结构模型和理论假设，而较少使用交叉验证来选择模型。这种根本目标的不同，也部分解释了为什么经济学模型在实际预测中有时表现不佳——因为其设计初衷并非追求极致的预测精度，而是揭示变量间的内在机制和政策效果。 </p>
    <p> 在深度学习盛行的今天，学习支持向量机等经典方法仍具有重要意义。首先，SVM 等模型具有坚实的数学基础和优美的理论体系，能够帮助学习者深入理解机器学习中的关键概念，如间隔最大化、核技巧和凸优化等。其次，对于中小规模数据集或特定应用场景，SVM 等经典方法往往能提供更可解释、更稳定的结果，且训练成本更低。此外，理解这些经典方法的发展脉络，有助于我们把握机器学习思想的演进，从而更深刻地认识当前技术的优势与局限。正是通过对经典与现代方法的对比学习，我们才能在不同问题中选择最合适的工具，而非盲目追随技术潮流。 </p>
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

p {
  font-size: 1.05rem;
  line-height: 1.8;
  margin-bottom: 15px;
  color: #34495e;
  text-align: justify;
  text-indent: 2em;
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

@media (max-width: 768px) {
  .ml-blog-page {
    padding: 10px;
  }
  
  .content {
    padding: 20px;
  }
  
  .page-title {
    font-size: 1.7rem;
  }
  
  p {
    font-size: 1rem;
    line-height: 1.7;
  }
  
  h2 {
    font-size: 1.3rem;
  }
}
</style>