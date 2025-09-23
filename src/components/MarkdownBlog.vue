<template>
  <div class="markdown-blog">
    <h1 class="page-title">Markdown博客</h1>
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

.date {
  color: #2980b9;
  font-style: italic;
  margin-bottom: 15px;
  font-weight: 500;
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
  color: #2c3e50;
  font-weight: 600;
  transition: color 0.3s ease;
}

.article-title a:hover {
  color: #3498db;
}

.excerpt {
  color: #7f8c8d;
}

p {
  font-size: 1.05rem;
  line-height: 1.8;
  color: #34495e;
  margin-bottom: 15px;
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
  .markdown-blog {
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