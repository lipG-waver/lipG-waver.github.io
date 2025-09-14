<template>
  <div class="markdown-post">
    <div class="content">
      <h1>{{ post.title }}</h1>
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
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

h1 {
  color: #333;
  text-align: center;
  margin-bottom: 30px;
}

.date {
  color: #666;
  font-style: italic;
  text-align: center;
  margin-bottom: 30px;
}

.markdown-content {
  font-size: 16px;
  line-height: 1.6;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3,
.markdown-content h4,
.markdown-content h5,
.markdown-content h6 {
  color: #333;
  margin-top: 24px;
  margin-bottom: 16px;
}

.markdown-content p {
  margin-bottom: 16px;
}

.markdown-content a {
  color: #0a84ff;
  text-decoration: none;
}

.markdown-content a:hover {
  text-decoration: underline;
}

.markdown-content strong {
  font-weight: 600;
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
}

.markdown-content blockquote {
  margin: 0 0 16px;
  padding: 0 16px;
  border-left: 4px solid #ddd;
  color: #666;
}

.markdown-content pre {
  background-color: #f6f8fa;
  border-radius: 6px;
  padding: 16px;
  overflow: auto;
  margin-bottom: 16px;
}

.markdown-content code {
  background-color: #f6f8fa;
  border-radius: 6px;
  padding: 2px 4px;
  font-family: monospace;
}
</style>