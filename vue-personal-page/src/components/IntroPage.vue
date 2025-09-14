<template>
  <div class="intro-page">
    <h1>建立网站的过程</h1>
    <div class="content">
      <p>起初，我通过自己的 GitHub 账户创建了静态网页仓库，并用最基本的 <code>index.html</code> 页面作为网站主页。这是我第一次从零开始搭建个人站点，起初所有内容都是手动嵌入的，导航栏和脚注每页都需要单独复制粘贴，维护起来非常繁琐。</p>

      <p>随后，我对项目结构进行了优化：将导航栏 <code>navbar.html</code> 和脚注 <code>footer.html</code> 抽离成单独文件，通过 <code>navbar.js</code> 脚本在页面加载时动态插入，实现了模块化。现在只需修改一个文件，整个站点就会统一更新，极大地提升了维护效率。</p>

      <p>网站样式通过 <code>style.css</code> 管理，保证了内容展示的一致性和扩展性。每新增一个页面，我只需关注内容本身，而不必重复搭建结构。</p>

      <p>为了进一步提升网站的技术水平和可维护性，我使用Vue框架对网站进行了重构。通过Vue的组件化架构，我将网站的不同部分拆分成独立的组件，使得代码结构更加清晰，易于维护和扩展。Vue的响应式数据绑定和虚拟DOM特性也大大提升了网站的性能和用户体验。</p>

      <p>未来，我希望将这个流程整理成一个简单的"建站接口"或脚手架，结合我的公众号"离谱纪"，为关注者提供一键生成网站的服务，让更多人能轻松拥有自己的技术主页。</p>

      <p>虽然我最初希望这个网站承载更多技术输出，但现在我更重视"先做出东西"。因为我还有很多想法在路上，网站本身就是我"先实现再分享"的实验场。</p>

      <p>项目源代码及更新计划可见我的 GitHub 仓库。</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'IntroPage'
}
</script>

<style scoped>
.intro-page {
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

p {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 15px;
}

ul {
  margin-bottom: 20px;
  padding-left: 20px;
}

li {
  margin-bottom: 10px;
  line-height: 1.5;
}
</style>