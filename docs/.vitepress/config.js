export default {
  base: '/blog/',
  title: "LipG's Blog",
  description: '技术探索与思考 | 记录学习旅程',
  ignoreDeadLinks: true,
  
  head: [
    ['link', { rel: 'icon', href: '/blog/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#3498db' }]
  ],
  
  themeConfig: {
    logo: '/logo.svg', // 可选：添加 logo 到 docs/public/logo.svg
    
    nav: [
      { text: '🏠 首页', link: '/' },
      { text: '📝 文章', link: '/posts/' },
      { text: '🚀 并行计算', link: '/posts/parallel-computing/' },
      { text: '👤 关于', link: '/about' },
      { 
        text: '🔗 返回主站', 
        link: 'https://lipg-waver.github.io/',
        target: '_blank'
      }
    ],
    
    sidebar: {
      '/posts/': [
        {
          text: '📝 所有文章',
          items: [
            { text: '文章首页', link: '/posts/' }
          ]
        },
        {
          text: '🚀 并行计算',
          collapsed: false,
          items: [
            { text: '专题介绍', link: '/posts/parallel-computing/' },
            { text: 'OpenMP 基础', link: '/posts/parallel-computing/openmp-basics' }
          ]
        }
      ]
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/lipG-waver' }
    ],
    
    footer: {
      message: '基于 VitePress 构建',
      copyright: 'Copyright © 2025 LipG | 保持学习，持续进步'
    },
    
    // 搜索
    search: {
      provider: 'local'
    },
    
    // 编辑链接
    editLink: {
      pattern: 'https://github.com/lipG-waver/lipG-waver.github.io/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },
    
    // 最后更新时间
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'short'
      }
    }
  }
}
