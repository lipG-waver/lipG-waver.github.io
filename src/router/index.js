import { createRouter, createWebHistory } from 'vue-router'
import Home from '../components/HomePage.vue'
import Intro from '../components/IntroPage.vue'
import DailyInsight from '../components/DailyInsightsPage.vue'
import LeetcodeBlog from '../components/LeetcodeBlogPage.vue'
import MlBlogPage from '../components/MlBlogPage.vue'
import DlBlogPage from '../components/DlBlogPage.vue'
import NewBlogPost from '../components/NewBlogPost.vue'
import MarkdownBlog from '../components/MarkdownBlog.vue'
import MarkdownPost from '../components/MarkdownPost.vue'
import TeachingAssistantPage from '../components/TeachingAssistantPage.vue'
// import FirstExerciseSession from '../components/FirstExerciseSession.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/intro',
    name: 'Intro',
    component: Intro
  },
  {
    path: '/daily-insight',
    name: 'DailyInsight',
    component: DailyInsight
  },
  {
    path: '/leetcode-blog',
    name: 'LeetcodeBlog',
    component: LeetcodeBlog
  },
  {
    path: '/ml-blog',
    name: 'MlBlogPage',
    component: MlBlogPage
  },
  {
    path: '/dl-blog',
    name: 'DlBlogPage',
    component: DlBlogPage
  },
  {
    path: '/new-blog-post',
    name: 'NewBlogPost',
    component: NewBlogPost
  },
  {
    path: '/markdown-blog',
    name: 'MarkdownBlog',
    component: MarkdownBlog
  },
  {
    path: '/markdown-blog/:id',
    name: 'MarkdownPost',
    component: MarkdownPost,
    props: true
  },
  {
    path: '/teaching-assistant',
    name: 'TeachingAssistantPage',
    component: TeachingAssistantPage
  },
  // {
  //   path: '/first-exercise-session',
  //   name: 'FirstExerciseSession',
  //   component: FirstExerciseSession
  // }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router