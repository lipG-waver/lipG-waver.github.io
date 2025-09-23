<template>
  <div class="daily-insights-page">
    <h1 class="page-title">每日收获分享</h1>
    <div class="content">
      <div class="insight-item">
        <h2 class="section-title">移动端菜单栏问题修复</h2>
        <p class="date">2025年8月7日</p>
        <p>今天解决了网站在移动端设备上菜单栏无法正常显示和交互的问题。在排查过程中发现了几个关键问题：</p>
        <ul>
          <li>代码冗余：菜单切换功能的JavaScript代码在多个地方重复定义</li>
          <li>事件监听重复：addEventListener在每次页面加载时都会重复绑定，导致事件处理异常</li>
          <li>未定义变量引用：尝试访问一个未声明的变量，导致脚本执行中断</li>
          <li>CSS样式缺失：缺少针对移动端的特定样式规则，导致菜单在小屏幕上显示异常</li>
        </ul>
        <p>通过重构代码、优化事件监听机制、修复变量引用问题以及添加响应式CSS规则，成功解决了这些问题。这次修复让我更深入地理解了移动端Web开发的复杂性和重要性。</p>
        <p>未来我计划进一步优化网站的响应式设计，提升在各种设备上的用户体验，同时也会关注网站性能优化和无障碍访问的改进。</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DailyInsightsPage'
}
</script>

<style scoped>
.daily-insights-page {
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

.section-title {
  color: #2c3e50;
  margin: 25px 0 15px 0;
  font-size: 1.5rem;
  position: relative;
  padding-bottom: 8px;
}

.section-title::after {
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

.insight-item {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.insight-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

p {
  font-size: 1.05rem;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 15px;
  text-align: justify;
}

ul {
  margin-bottom: 20px;
  padding-left: 20px;
}

li {
  margin-bottom: 10px;
  line-height: 1.6;
  color: #34495e;
}

@media (max-width: 768px) {
  .daily-insights-page {
    padding: 10px;
  }
  
  .content {
    padding: 20px;
  }
  
  .page-title {
    font-size: 1.7rem;
  }
  
  .section-title {
    font-size: 1.3rem;
  }
  
  p {
    font-size: 1rem;
    line-height: 1.7;
  }
}
</style>