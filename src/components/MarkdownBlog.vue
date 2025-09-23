<template>
  <div class="markdown-blog">
    <h1>Markdown博客</h1>
    <div class="content">
      <div class="article-list">
        <div class="article-item" v-for="post in posts" :key="post.id">
          <h2 class="article-title">
            <router-link :to="`/markdown-blog/${post.id}`">{{ post.title }}</router-link>
          </h2>
          <p class="date">发布日期：{{ post.date }}</p>
          <p class="excerpt">{{ post.excerpt }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MarkdownBlog',
  data() {
    return {
      posts: [
        {
          id: 'sample',
          title: '第一篇Markdown文章',
          date: '2025-09-14',
          excerpt: '这是第一篇Markdown文章的摘要...'
        },
        {
          id: 'example',
          title: '示例Markdown文章',
          date: '2025-09-15',
          excerpt: '这是一个示例Markdown文章，展示了各种Markdown语法...'
        },
        {
          id: 3,
          title: '如何使用Markdown写博客',
          date: '2025-09-16',
          excerpt: '介绍如何使用Markdown格式编写博客文章...'
        }
      ]
    }
  }
}
</script>

<style scoped>
.markdown-blog {
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
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.article-item:last-child {
  margin-bottom: 0;
  border-bottom: none;
}

.article-title a {
  text-decoration: none;
  color: #333;
}

.article-title a:hover {
  color: #0a84ff;
}

.excerpt {
  color: #666;
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