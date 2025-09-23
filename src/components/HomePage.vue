<template>
  <div class="container">
    <ProfilePhoto />
    
    <h2 class="section-title">关于我</h2>
    <div class="about-content">
      <p>我是复旦大学的学生。有一天也会离开复旦大学，但我还会称自己是复旦的学生，究其根本，复旦"自由而无用"的精神是我成长的底色。复旦的理念与气质，在我身上留下了深刻的印记。我曾经多次撰文，提到自由而无用的含义。所谓有用，乃为无用。所谓无用，实则有用。当今时代，盲目追求有用的，容易求得简单的有用。因此百无一用者，乃是改良无用之精华者，故为有用。</p> 
      <p>我的世界观主要在复旦时期形成。其中，两门课程对我影响尤深：一是孙立坚老师的《货币银行学》，课程结合实际政策进行深入分析，极具现实意义；另一门是王永钦老师的《经济学前沿专题H》，聚焦人类社会永恒的不完全合约问题，从"威尼斯商人"时代对违约者施以剁肉之刑，到现代通过坐牢惩戒，再到如今以抵押品为主要机制，社会逐步实现了对契约双方更平衡、理性的保护。</p> 
      <p>在复旦我学习了很多数学课。数学分析，我刷了很多史济怀一书的题目。线性代数是我较晚接触的内容，而概率论与数理统计课程则由世界闻名的洪永淼教授的学生付中昊老师授课，为我打下了扎实的理论基础。与数学密切相关的计量经济学也带来了很大收获，包括统计学、计量经济学H、截面与面板数据分析H等课程。起初我认为朱宏飞老师的授课风格不如洪老师严谨，但后来体会到朱老师善于简化复杂问题，这种思路也逐渐成为我解决问题的方法论；相比之下，洪老师的课程则更注重数学推导。我的数学课还不仅于此，我还修读了谢锡麟老师的实变函数与泛函分析基础一门课。</p> 
      <p>2024-2025年，我自费到了伯克利，开始了难忘的学习生活。我初次接触技术，慢慢变得痴迷于技术，以及术背后的道。我修读了数值分析、数据结构、计算机结构、算法、操作系统、机器学习、神经网络、区块链金融等课程。技术中，有趣的还不少。Java, 一开始带我体会面向对象编程的思想，虽然我现在已经不写Java了。C，最有魅力的语言。可能是我没掌握的很好的语言我觉得很有魅力。操作系统需要C来Debug, 这方面我的一位朋友做的很好，我很佩服他。还好玩的是RISC-V汇编语言，越底层了越了解这个人造世界原来是这样的。</p>
      <p>不论是计算机还是经济学，本质上都是对人类世界的好奇。我对大自然没有一丝好奇的地方。记得我高中以前误以为自己喜欢生物。我自己也认为自己擅长考试，结果到生物考试就出个班级倒数。我个人想想看，确实没什么兴趣。人类社会还是更有意思的存在。</p>
      <p>我热爱学习的过程，尤其是那些追求本质与真理knowledge。真正的学习带来的是持续的快乐。尽管世界上存在许多"伪知识"——它们也许需要时间去辨别——但借助逻辑与理性，我们可以更早地洞察真伪，找到通往真知的路径。</p>
    </div>
    
    <h2 class="section-title">学习经历</h2>
    <div class="education">
      <div class="education-item">
        <span>2021 - 2024</span>: 复旦大学经济学院<br />
        主修经济学，深入学习数学、计量经济学及金融相关课程。
      </div>
      <div class="education-item">
        <span>2024 - 2025</span>: UC Berkeley 访学<br />
        计算机科学，包括数据结构、算法、操作系统及机器学习。
      </div>
      <div class="education-item">
        <span>2025 - 2026</span>: 复旦大学经济学院<br />
        修完剩余的课程，准备离开复旦。顺便当个助教，做点研究。
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
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
}

.about-content {
  background-color: #fff;
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  margin-bottom: 40px;
}

.section-title {
  font-size: 1.8rem;
  margin: 40px 0 20px 0;
  color: #2c3e50;
  text-align: center;
  position: relative;
  padding-bottom: 10px;
}

.section-title::after {
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
  color: #34495e;
  text-align: justify;
  margin-bottom: 20px;
  text-indent: 2em;
}

.education {
  background-color: #fff;
  padding: 30px;
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
  text-align: left;
  margin-top: 20px;
}

.education-item {
  margin-bottom: 25px;
  padding: 15px;
  border-left: 4px solid #3498db;
  background-color: #f8f9fa;
  border-radius: 0 8px 8px 0;
}

.education-item:last-child {
  margin-bottom: 0;
}

.education-item span {
  font-weight: bold;
  color: #2980b9;
  display: inline-block;
  margin-bottom: 5px;
}

@media (max-width: 768px) {
  .container {
    padding: 10px;
  }
  
  .about-content,
  .education {
    padding: 20px;
  }
  
  .section-title {
    font-size: 1.5rem;
  }
  
  p {
    font-size: 1rem;
    line-height: 1.7;
  }
}
</style>