<template>
  <div class="leetcode-blog-page">
    <h1>算法题解</h1>
    <div class="content">
      <p>这里将分享我的LeetCode刷题经验和算法学习心得。</p>
      <p>算法是计算机科学的基础，通过刷题可以锻炼逻辑思维和编程能力。</p>
      <!-- 这里可以添加具体的算法题解内容 -->
    </div>
  </div>
</template>

<script>
export default {
  name: 'LeetcodeBlogPage'
}
</script>

<style scoped>
.leetcode-blog-page {
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

.difficulty {
  color: #666;
  font-weight: bold;
  margin-bottom: 15px;
}

.problem-item {
  margin-bottom: 40px;
}

.problem-item:last-child {
  margin-bottom: 0;
}

p {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 15px;
}

pre {
  background-color: #f8f8f8;
  padding: 15px;
  border-radius: 5px;
  overflow-x: auto;
  margin: 20px 0;
}

code {
  font-family: 'Courier New', monospace;
  font-size: 14px;
}
</style>