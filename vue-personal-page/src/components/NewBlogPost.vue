<template>
  <div class="new-blog-page">
    <h1>新文章</h1>
    <div class="content">
      <div class="article-item">
        <h2 class="article-title">文章标题</h2>
        <p class="date">发布日期：2025年9月14日</p>
        <p>
          这里是文章的内容。你可以在这里添加你的新文章内容。
        </p>
        <p>
          你可以添加多个段落，列表，或者其他HTML元素来丰富你的文章内容。
        </p>
        <ul>
          <li>列表项1</li>
          <li>列表项2</li>
          <li>列表项3</li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'NewBlogPost'
}
</script>

<style scoped>
.new-blog-page {
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