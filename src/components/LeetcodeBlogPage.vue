<template>
  <div class="leetcode-blog-page">
    <h1 class="page-title">算法题解</h1>
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

.difficulty {
  color: #2980b9;
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
  font-size: 1.05rem;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 15px;
  text-align: justify;
}

pre {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 20px 0;
  border: 1px solid #eee;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

code {
  font-family: 'Courier New', monospace;
  font-size: 0.95rem;
  color: #e74c3c;
}

@media (max-width: 768px) {
  .leetcode-blog-page {
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