import { defineConfig } from 'vitepress'

export default defineConfig({
  // 基本信息
  title: '我的博客',
  description: '个人技术博客 - 记录学习与成长',
  lang: 'zh-CN',
  
//   如果部署在子路径，取消注释并修改
  base: '/blog/',
  
  // 清理 URL
  cleanUrls: true,
  
  // Head 配置
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3498db' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }],
  ],
  
  // 主题配置
  themeConfig: {
    // 网站 Logo
    logo: '/logo.png', // 需要在 docs/public/ 目录下放置 logo.png
    
    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '博客', link: '/posts/' },
      { text: '归档', link: '/archive' },
      { text: '标签', link: '/tags' },
      { text: '关于', link: '/about' },
      { 
        text: '更多',
        items: [
          { text: '友情链接', link: '/friends' },
          { text: 'RSS', link: '/feed.xml' }
        ]
      }
    ],
    
    // 侧边栏
    sidebar: {
      '/posts/': [
        {
          text: '开始阅读',
          items: [
            { text: '所有文章', link: '/posts/' }
          ]
        },
        {
          text: '分类',
          items: [
            { text: 'Vue.js', link: '/posts/categories/vue' },
            { text: 'JavaScript', link: '/posts/categories/javascript' },
            { text: '前端工程化', link: '/posts/categories/engineering' },
            { text: '学习笔记', link: '/posts/categories/notes' }
          ]
        }
      ]
    },
    
    // 社交链接
    socialLinks: [
      { icon: 'github', link: 'https://github.com/lipg-waver' }
    ],
    
    // 页脚
    footer: {
      message: '基于 <a href="https://vitepress.dev" target="_blank">VitePress</a> 构建',
      copyright: 'Copyright © 2025-present'
    },
    
    // 编辑链接
    editLink: {
      pattern: 'https://github.com/yourusername/yourrepo/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },
    
    // 最后更新时间
    lastUpdated: {
      text: '最后更新',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },
    
    // 本地搜索
    search: {
      provider: 'local',
      options: {
        locales: {
          root: {
            translations: {
              button: {
                buttonText: '搜索文档',
                buttonAriaLabel: '搜索文档'
              },
              modal: {
                noResultsText: '无法找到相关结果',
                resetButtonTitle: '清除查询条件',
                footer: {
                  selectText: '选择',
                  navigateText: '切换',
                  closeText: '关闭'
                }
              }
            }
          }
        }
      }
    },
    
    // 文档页脚
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    
    // 大纲配置
    outline: {
      level: [2, 3],
      label: '本页目录'
    },
    
    // 返回顶部
    returnToTopLabel: '返回顶部',
    
    // 深色模式
    darkModeSwitchLabel: '外观',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    
    // 侧边栏菜单标签
    sidebarMenuLabel: '菜单',
  },
  
  // Markdown 配置
  markdown: {
    // 显示行号
    lineNumbers: true,
    
    // 图片懒加载
    image: {
      lazyLoading: true
    },
    
    // 自定义容器
    container: {
      tipLabel: '💡 提示',
      warningLabel: '⚠️ 警告',
      dangerLabel: '❌ 危险',
      infoLabel: 'ℹ️ 信息',
      detailsLabel: '详细信息'
    },
    
    // 数学公式支持（可选）
    math: true
  },
  
  // Vite 配置
  vite: {
    ssr: {
      noExternal: ['vue-markdown-render']
    }
  },
  
  // 站点地图
  sitemap: {
    hostname: 'https://yoursite.com'
  }
})