<template>
  <div class="markdown-post">
    <div class="content">
      <h1 class="post-title">{{ post.title }}</h1>
      <p class="date">发布日期：{{ post.date }}</p>
      <vue-markdown :source="post.content" class="markdown-content"></vue-markdown>
    </div>
  </div>
</template>

<script>
import VueMarkdown from 'vue-markdown-render'

export default {
  name: 'MarkdownPost',
  components: {
    'vue-markdown': VueMarkdown
  },
  data() {
    return {
      post: {
        id: this.$route.params.id,
        title: '',
        date: '',
        content: ''
      }
    }
  },
  methods: {
    async loadMarkdownFile() {
      try {
        // 从public目录加载markdown文件
        const response = await fetch(`/markdown/${this.post.id}.md`)
        if (response.ok) {
          const markdownContent = await response.text()
          this.post.content = markdownContent
          this.post.title = '示例Markdown文章' // 实际应用中可以从文件名或元数据中提取标题
          this.post.date = '2025-09-14' // 实际应用中可以从文件元数据中提取日期
        } else {
          console.error('无法加载Markdown文件')
        }
      } catch (error) {
        console.error('加载Markdown文件时出错:', error)
      }
    }
  },
  mounted() {
    // 从markdown文件加载数据
    this.loadMarkdownFile()
  }
}
</script>

<style scoped>
.markdown-post {
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

.post-title {
  color: #2c3e50;
  text-align: center;
  margin-bottom: 30px;
  font-size: 2rem;
  position: relative;
  padding-bottom: 10px;
}

.post-title::after {
  content: '';
  display: block;
  width: 60px;
  height: 3px;
  background: linear-gradient(90deg, #3498db, #2c3e50);
  margin: 10px auto 0;
  border-radius: 3px;
}

.date {
  color: #2980b9;
  font-style: italic;
  text-align: center;
  margin-bottom: 30px;
  font-weight: 500;
}

.markdown-content {
  font-size: 1.05rem;
  line-height: 1.8;
  color: #34495e;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6 {
  color: #2c3e50;
  margin-top: 24px;
  margin-bottom: 16px;
  position: relative;
}

.markdown-content h1 {
  font-size: 1.8rem;
  padding-bottom: 10px;
}

.markdown-content h1::after {
  content: '';
  display: block;
  width: 60px;
  height: 3px;
  background: linear-gradient(90deg, #3498db, #2c3e50);
  margin: 10px 0 0 0;
  border-radius: 3px;
}

.markdown-content h2 {
  font-size: 1.5rem;
  padding-bottom: 8px;
}

.markdown-content h2::after {
  content: '';
  display: block;
  width: 40px;
  height: 2px;
  background: linear-gradient(90deg, #3498db, #2c3e50);
  margin: 8px 0 0 0;
  border-radius: 2px;
}

.markdown-content p {
  margin-bottom: 16px;
  text-align: justify;
}

.markdown-content a {
  color: #3498db;
  text-decoration: none;
  border-bottom: 1px dashed #3498db;
}

.markdown-content a:hover {
  text-decoration: none;
  border-bottom: 1px solid #3498db;
}

.markdown-content strong {
  font-weight: 600;
  color: #2c3e50;
}

.markdown-content em {
  font-style: italic;
}

.markdown-content ul,
.markdown-content ol {
  margin-bottom: 16px;
  padding-left: 24px;
}

.markdown-content li {
  margin-bottom: 8px;
  line-height: 1.6;
}

.markdown-content blockquote {
  margin: 0 0 16px;
  padding: 15px 20px;
  border-left: 4px solid #3498db;
  color: #34495e;
  background-color: #f8f9fa;
  border-radius: 0 8px 8px 0;
}

.markdown-content pre {
  background-color: #f8f9fa;
  border-radius: 8px;
  padding: 16px;
  overflow: auto;
  margin-bottom: 16px;
  border: 1px solid #eee;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.markdown-content code {
  background-color: #f8f9fa;
  border-radius: 4px;
  padding: 2px 6px;
  font-family: monospace;
  color: #e74c3c;
  font-size: 0.95rem;
}

@media (max-width: 768px) {
  .markdown-post {
    padding: 10px;
  }
  
  .content {
    padding: 20px;
  }
  
  .post-title {
    font-size: 1.7rem;
  }
  
  .markdown-content {
    font-size: 1rem;
    line-height: 1.7;
  }
  
  .markdown-content h1 {
    font-size: 1.5rem;
  }
  
  .markdown-content h2 {
    font-size: 1.3rem;
  }
}
</style>