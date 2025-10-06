import DefaultTheme from 'vitepress/theme'
import './custom.css'

// 导入主站的共享样式
import '../../../src/assets/styles/variables.css'
import '../../../src/assets/styles/common.css'

// 如果需要自定义布局组件
// import CustomLayout from './components/CustomLayout.vue'

export default {
  extends: DefaultTheme,
  
  // 如果需要自定义布局
  // Layout: CustomLayout,
  
  enhanceApp({ app, router, siteData }) {
    // 注册全局组件
    // app.component('CustomComponent', CustomComponent)
    
    // 路由守卫
    router.onBeforeRouteChange = (to) => {
      console.log('Navigating to:', to)
    }
  }
}