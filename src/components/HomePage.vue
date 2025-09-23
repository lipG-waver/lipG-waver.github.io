<template>
  <div class="container">
    <ProfilePhoto />
    
    <h2>关于我</h2>
    <p>我来自复旦大学，复旦"自由而无用"的精神是我成长的重要底色。自高中起我便受到复旦氛围的熏陶，虽然人生中曾有过一段"走岔路"的经历（回望时颇具戏剧性），最终仍回到复旦大学完成本科学业。复旦的理念与气质，在我身上留下了深刻的印记。</p> 
    <p>我的世界观主要在复旦时期形成。其中，两门课程对我影响尤深：一是孙立坚老师的《货币银行学》，课程结合实际政策进行深入分析，极具现实意义；另一门是王永钦老师的《经济学前沿专题H》，聚焦人类社会最具挑战的不完全合约问题（或称承诺问题），并探讨了人类历史上对这一问题的多种应对方式——从"威尼斯商人"时代对违约者施以剁肉之刑，到现代通过坐牢惩戒，再到如今以抵押品为主要机制，社会逐步实现了对契约双方更平衡、理性的保护。</p> 
    <p>来到加州大学伯克利分校后，我重新审视了在复旦的学习，尤其意识到数学课程的重要性。例如，我曾低估数学分析在计算机中的作用，直到我学习扩散模型（Diffusion Model）才深刻理解其价值。线性代数是我较晚接触的内容，而概率论与数理统计课程则由洪永淼教授的学生付中昊老师授课，为我打下了扎实的理论基础。与数学密切相关的计量经济学也带来了很大收获，包括统计学、计量经济学H、截面与面板数据分析H等课程。起初我认为朱宏飞老师的授课风格不如洪老师严谨，但后来体会到朱老师善于简化复杂问题，这种思路也逐渐成为我解决问题的方法论；相比之下，洪老师的课程则更注重数学推导。</p> 
    <p>在伯克利，我的重心逐渐转向技术学习。我先后选修了多门计算机科学核心课程，如 CS61B（数据结构）、CS61C（计算机结构）等。起初我并未计划学习计算机，但由于无法选修伯克利的研究生经济学课程，为保持学分数量，我不得不转向计算机课程（也因为我认为这边的本科经济学课质量远不如复旦经院，辜负了母亲辛苦赚来的学费）。就这样，我在一年内完成了计算机专业的核心课程，尽管尚未正式取得计算机学位，我正考虑是否申请计算机硕士，以发出更明确的学术信号。</p>
    <p>我的计算机学习主要沿着两条主线展开：一是算法方向，二是计算机系统方向。算法路线包括数据结构（作为算法入门必修课）、高效算法与不可解问题（如通过学习认识到中国象棋尚未被完全计算解决）、数据分析（Data 100）、机器学习（教授如何在无明确理论前提下进行回归与分类）、神经网络（让我深刻反思人类认知方式的一门课）。系统路线则涵盖了计算机结构和操作系统等课程。计算机结构课程首次让我真正掌握了 C 语言，尤其是指针的使用，并初步了解汇编语言和逻辑电路的构成；操作系统课程虽完成得不尽如人意，但我依旧从中领悟到同步、异步与调度等基本思想。</p>
    <p>我热爱学习的过程，尤其是那些追求本质与真理的知识。真正的学习带来的是持续的快乐。尽管世界上存在许多"伪知识"——它们也许需要时间去辨别——但借助逻辑与理性，我们可以更早地洞察真伪，找到通向真知的路径。</p>
    
    <h2>学习经历</h2>
    <div class="education">
      <div class="education-item">
        <span>2021 - 2024</span>: 复旦大学经济学院<br />
        主修经济学，深入学习数学、计量经济学及金融相关课程。
      </div>
      <div class="education-item">
        <span>2024 - 2025</span>: UC Berkeley 访学<br />
        专注于计算机科学，包括数据结构、算法、操作系统及机器学习。
      </div>
    </div>
  </div>
</template>

<script>
import ProfilePhoto from './ProfilePhoto.vue'

export default {
  name: 'HomePage',
  components: {
    ProfilePhoto
  }
}
</script>

<style scoped>
.home-page {
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
}

.content {
  background-color: #fff;
  padding: 40px;
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  text-align: center;
  margin: 60px auto;
}

.profile-section {
  margin-bottom: 40px;
}

.education-section {
  text-align: left;
  margin-top: 30px;
}

.education-item {
  margin-bottom: 20px;
}

.education-item span {
  font-weight: bold;
  color: #0a84ff;
}

h1 {
  font-size: 2rem;
  margin-bottom: 20px;
  color: #333;
}

h2 {
  font-size: 1.5rem;
  margin-top: 40px;
  margin-bottom: 10px;
  color: #333;
  border-bottom: 1px solid #eee;
  padding-bottom: 6px;
}

p {
  font-size: 1.05rem;
  line-height: 1.75;
  color: #555;
  text-align: left;
}

.profile-photo {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  margin-bottom: 20px;
}
</style>