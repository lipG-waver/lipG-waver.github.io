export default {
  base: '/blog/',  // 如果仓库名不是 username.github.io，改成 /仓库名/blog/
  title: '我的博客',
  description: '个人技术博客',
  ignoreDeadLinks: true,
  
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '文章', link: '/posts/' }
    ],
    
    sidebar: [
      {
        text: '文章列表',
        items: [
          { text: '介绍', link: '/intro' }
        ]
      }
    ]
  }
}