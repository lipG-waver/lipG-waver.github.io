<template>
  <div class="container">
    <!-- 语言切换按钮 -->
    <div class="lang-switcher">
      <button 
        :class="{ active: locale === 'zh' }" 
        @click="locale = 'zh'"
      >中文</button>
      <button 
        :class="{ active: locale === 'en' }" 
        @click="locale = 'en'"
      >EN</button>
    </div>

    <ProfilePhoto />
    
    <h2 class="section-title">{{ t('aboutMe') }}</h2>
    <div class="about-content">
      <p v-for="(para, index) in t('aboutParagraphs')" :key="index">{{ para }}</p>
    </div>
    
    <h2 class="section-title">{{ t('education') }}</h2>
    <div class="education">
      <div class="education-item" v-for="(item, index) in t('educationItems')" :key="index">
        <span>{{ item.period }}</span>: {{ item.school }}<br />
        {{ item.description }}
      </div>
    </div>

    <h2 class="section-title">{{ t('research') }}</h2>
    <div class="research-content">
      <p v-for="(para, index) in t('researchParagraphs')" :key="index">{{ para }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ProfilePhoto from './ProfilePhoto.vue'

const locale = ref('zh')

const messages = {
  zh: {
    aboutMe: '关于我',
    education: '学习经历',
    research: '研究与项目',
    aboutParagraphs: [
      '我是复旦大学的学生。有一天也会离开复旦大学，但我还会称自己是复旦的学生，究其根本，复旦"自由而无用"的精神是我成长的底色。复旦的理念与气质，在我身上留下了深刻的印记。所谓有用，乃为无用。所谓无用，实则有用。当今时代，盲目追求有用的，容易求得简单的有用。因此百无一用者，乃是改良无用之精华者，故为有用。',
      '我从初中开始就不是一个"成功"的孩子。正因如此，我养成了观察的习惯——观察那些成绩好的同学是怎么学的，观察他们的父母是怎么培养的。这种旁观者的视角从未离开我。高中时我当了排球队队长，接手的是一支全输的队伍，一年后我们拿下了区冠军。我一直好奇：为什么有的方法能让个体或集体发生质变？',
      '后来我做过家教，走进别人的家庭内部。我发现，每个"生病"的孩子背后，往往都有一个"生病"的家庭。那些沉默寡言的男孩，背后常常有一个无论做什么都会被指手画脚的母亲。问题从来不只是个体的问题，而是系统的问题。',
      '我曾被父亲逼迫去中科大，感受到压迫后，我选择了退学，那以后我的内心变得无比强大。我重新考回复旦，选择了经济学——因为我想理解人的行为，理解人类社会如何运作。我有一个"小媳妇理论"：所有人在这个世界上都处于某种结构性的不自由中，你总要帮上级完成什么，哪怕你是大资本家，也得帮地方政府做考虑。小媳妇，就要两头瞒。',
      '我的世界观主要在复旦时期形成。两门课程对我影响尤深：孙立坚老师的《货币银行学》，结合实际政策进行分析；王永钦老师的《经济学前沿专题H》，聚焦人类社会永恒的不完全合约问题——从"威尼斯商人"时代对违约者剁肉之刑，到现代坐牢惩戒，再到如今以抵押品为主要机制，社会逐步实现了对契约双方更理性的保护。',
      '我运营着一个微信公众号，订阅人数超过一万。我在上面写教育政策分析、个人成长思考、以及对复旦的观察。正是因为这些文章，很多学弟学妹愿意相信我、找我聊天，这也是Meal Chatting Hour能够持续进行的基础。写作对我来说是整理思维的方式，也是与更多人建立连接的桥梁。',
      '2024-2025年，我自费到伯克利交换，开始了难忘的学习生活。我初次接触技术，慢慢变得痴迷于技术，以及术背后的道。数据结构、算法、操作系统、机器学习、神经网络……越底层越了解这个人造世界原来是这样运作的。不论是计算机还是经济学，本质上都是对人类世界的好奇。我热爱学习的过程，尤其是那些追求本质与真理的知识。'
    ],
    educationItems: [
      {
        period: '2021 - 2024',
        school: '复旦大学经济学院',
        description: '主修经济学，深入学习数学、计量经济学及金融相关课程。'
      },
      {
        period: '2024 - 2025',
        school: 'UC Berkeley 访学',
        description: '计算机科学，包括数据结构、算法、操作系统及机器学习。'
      },
      {
        period: '2025 - 2026',
        school: '复旦大学经济学院',
        description: '修完剩余的课程，准备离开复旦。顺便当个助教，做点研究。'
      }
    ],
    researchParagraphs: [
      '我的核心研究兴趣是"注意力主权"（Attention Sovereignty）——将人的注意力视为稀缺的经济资源，需要从系统层面加以保护，而非仅仅依赖个人意志力。五分钟的刷手机变成一小时，这不是意志力的失败，而是人类认知系统与每毫秒迭代的对抗性算法之间的不对称战争。很显然，保护人的注意力不是一个应用层的问题，它需要重新想象整个计算栈——从高效的边缘推理，到操作系统架构，再到激励兼容的机制设计。我提出了Attention OS的概念架构，包含四个核心模块：Guard（防护）、Enhance（增强）、Navigate（导航）、Nurture（滋养）。',
      '当然，世界上有很多保护人类注意力、减轻认知负担的方式。其中重要的一点是提早帮助年轻学生确立目标，减少内耗。我开发了MindForge，一个基于苏格拉底式对话的AI认知教练系统，通过结构化的三阶段对话，帮助用户先澄清自己的价值观，再探索选择，最后做出决策。目前有50多位用户给过反馈，最主要的反馈集中在缺乏垂直领域的知识。',
      '为了更好地理解年轻学生的需求，我开始进行"Meal Chatting Hour"，与大一大二学生进行一对一的结构化访谈，试图理解：为什么人会被困在错误的框架里？我记录了每次的谈话内容并进行结构化总结。我发现，人们的焦虑很少来自真正的能力不足，而是来自有缺陷的心智模型——比如误解GPA的计算方式，高估早期选择的不可逆性，或者把恐惧误认为信息。我现在本质上是在手动执行"注意力操作系统"应该做的事：帮助用户建立正确的心智模型，对齐短期行动与长期价值。',
      '我曾花两年时间倡导复旦的绩点政策改革，最终成功推动绩点限制从30%的A和A-，变为A-理论上可以无限、决策权由老师控制。另外，我在公众号上提出了Pass/No Pass选项的实施建议，后续在教务处的改革方案中得到落实。然而，后续数据让我沉默：80%的学生用它来逃避挑战，而非拥抱探索。规则变了，人没变。针对这一现象，我收集了学生访谈内容，撰写了情况报告并提交给教务处。',
      '目前我正在进行FlashAttention在华为昇腾NPU上的优化研究。这是为未来的边缘部署做底层准备——如果注意力保护系统要无处不在，它必须足够轻量。'
    ]
  },
  en: {
    aboutMe: 'About Me',
    education: 'Education',
    research: 'Research & Projects',
    aboutParagraphs: [
      'I am a student at Fudan University. One day I will leave Fudan, but I will still call myself a Fudan student. At its core, Fudan\'s spirit of "free and useless" has been the foundation of my growth. What seems useful may be useless; what seems useless may actually be useful. In today\'s era, blindly pursuing utility often leads to superficial gains. Therefore, those who appear utterly useless may be the ones refining the essence of the "useless," making them truly useful.',
      'I became an observer because I wasn\'t a successful kid. In middle school, while classmates seemed to glide effortlessly toward good grades, I found myself watching from the margins, asking: what are they doing that I\'m not? What are their parents doing differently? This observer\'s perspective never left me. In high school, I became captain of the volleyball team—a team that had lost every match. Within a year, we won the district championship. I\'ve always been curious: why do some methods unlock potential in individuals and groups, while others fail?',
      'Later, I started tutoring and entered other families\' homes. I discovered that behind every "sick" child, there is often a "sick" family. Those quiet, withdrawn boys often have mothers who criticize everything they do. Problems are never just individual—they are systemic.',
      'I was once pressured by my father to attend USTC. After feeling oppressed, I chose to withdraw. After that, my inner strength became unshakeable. I retook the gaokao and returned to Fudan, choosing economics—because I wanted to understand human behavior and how society operates. I have a "daughter-in-law theory": everyone in this world exists in some form of structural unfreedom. You always have to help your superiors accomplish something. Even if you\'re a big capitalist, you still have to consider the local government\'s interests. A daughter-in-law must keep secrets from both sides.',
      'My worldview was largely shaped during my time at Fudan. Two courses had a profound impact: Professor Sun Lijian\'s "Money and Banking," which combined deep analysis with real-world policy, and Professor Wang Yongqin\'s "Frontiers in Economics H," which focused on the eternal problem of incomplete contracts—from the era of "The Merchant of Venice" when defaulters had flesh cut from their bodies, to modern imprisonment, to today\'s collateral-based mechanisms.',
      'I run a WeChat public account with over 10,000 subscribers. I write about educational policy analysis, personal growth reflections, and observations about Fudan. It is precisely because of these articles that many younger students trust me and are willing to talk with me—this is the foundation that makes Meal Chatting Hours possible. Writing, for me, is a way to organize my thoughts and a bridge to connect with more people.',
      'In 2024-2025, I went to Berkeley at my own expense and began an unforgettable learning journey. I encountered technology for the first time and gradually became fascinated by both the techniques and the underlying principles. Data Structures, Algorithms, Operating Systems, Machine Learning, Neural Networks... The deeper you go, the more you understand how this man-made world actually works. Whether it\'s computer science or economics, both fundamentally stem from curiosity about the human world. I love the process of learning, especially knowledge that pursues essence and truth.'
    ],
    educationItems: [
      {
        period: '2021 - 2024',
        school: 'School of Economics, Fudan University',
        description: 'Majored in Economics, with in-depth study of mathematics, econometrics, and finance-related courses.'
      },
      {
        period: '2024 - 2025',
        school: 'UC Berkeley (Exchange)',
        description: 'Computer Science, including Data Structures, Algorithms, Operating Systems, and Machine Learning.'
      },
      {
        period: '2025 - 2026',
        school: 'School of Economics, Fudan University',
        description: 'Completing remaining courses and preparing to leave Fudan. Also serving as a TA and doing research.'
      }
    ],
    researchParagraphs: [
      'My core research interest is "Attention Sovereignty"—treating human attention as a scarce economic resource that requires protection at the system level, rather than relying solely on individual willpower. A five-minute scroll that expands into an hour is not a personal failure of willpower—it is an asymmetric war between human cognitive systems that evolved over millennia and adversarial algorithms that iterate every millisecond. Clearly, protecting human attention is not an app-level problem. It requires reimagining the entire computational stack—from efficient edge inference to operating system architecture to incentive-compatible mechanism design. I proposed the conceptual architecture of Attention OS, comprising four core modules: Guard, Enhance, Navigate, and Nurture.',
      'Of course, there are many ways to protect human attention and reduce cognitive burden. One crucial approach is helping young students establish goals early to reduce internal friction. I developed MindForge, a Socratic dialogue-based AI cognitive coaching system. Through structured three-stage conversations, it helps users first clarify their values, then explore options, and finally make decisions. Over 50 users have provided feedback, with the main concern being the lack of domain-specific knowledge.',
      'To better understand the needs of young students, I started conducting "Meal Chatting Hours"—one-on-one structured interviews with freshmen and sophomores, trying to understand: why do people get stuck in wrong frameworks? I recorded each conversation and created structured summaries. I discovered that people\'s anxiety rarely stems from genuine lack of ability, but from flawed mental models—misunderstanding how GPA calculations work, overestimating the permanence of early choices, or confusing fear with information. I am essentially manually executing what an "Attention OS" should do: helping users build correct mental models and align short-term actions with long-term values.',
      'I spent two years advocating for GPA policy reform at Fudan, eventually succeeding in changing the grade cap from 30% A and A- to unlimited A- with decisions controlled by professors. Additionally, I proposed the implementation of a Pass/No Pass option on my public account, which was later adopted in the administration\'s reform plan. However, subsequent data silenced me: 80% of students used it to avoid challenges, not embrace exploration. Rules changed, but people didn\'t. In response, I collected student interviews, wrote a situation report, and submitted it to the academic affairs office.',
      'I am currently conducting FlashAttention optimization research on Huawei\'s Ascend NPU. This is foundational preparation for future edge deployment—if attention protection systems are to be ubiquitous, they must be lightweight enough.'
    ]
  }
}

const t = (key) => {
  return messages[locale.value][key]
}
</script>

<style scoped>
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
  position: relative;
}

.lang-switcher {
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  gap: 8px;
}

.lang-switcher button {
  padding: 6px 14px;
  border: 1px solid #3498db;
  background: white;
  color: #3498db;
  border-radius: 20px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.3s ease;
}

.lang-switcher button:hover {
  background: #ecf5fc;
}

.lang-switcher button.active {
  background: #3498db;
  color: white;
}

.about-content,
.research-content {
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
  margin-bottom: 40px;
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
    padding-top: 60px;
  }
  
  .lang-switcher {
    top: 10px;
    right: 10px;
  }
  
  .about-content,
  .research-content,
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