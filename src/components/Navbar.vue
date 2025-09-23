<template>
  <nav class="navbar">
    <div class="nav-container">
      <div class="nav-logo">
  <router-link to="/">LipG</router-link>
</div>
      <ul class="nav-menu">
        <li class="nav-item">
          <router-link to="/" class="nav-link">首页</router-link>
        </li>
        <li class="nav-item">
          <router-link to="/intro" class="nav-link">自我介绍</router-link>
        </li>
        <li class="nav-item">
          <router-link to="/daily-insight" class="nav-link">日常思考</router-link>
        </li>
        <li class="nav-item dropdown">
          <a href="#" class="nav-link">技术博客 <i class="fas fa-chevron-down"></i></a>
          <ul class="dropdown-menu">
            <li><router-link to="/leetcode-blog" class="dropdown-link">LeetCode题解</router-link></li>
            <li><router-link to="/ml-blog" class="dropdown-link">机器学习</router-link></li>
            <li><router-link to="/dl-blog" class="dropdown-link">深度学习</router-link></li>
            <li><router-link to="/new-blog-post" class="dropdown-link">新文章</router-link></li>
            <li><router-link to="/markdown-blog" class="dropdown-link">Markdown博客</router-link></li>
          </ul>
        </li>
      </ul>
      <div class="nav-toggle" @click="toggleMenu">
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
      </div>
    </div>
  </nav>
</template>

<script>
export default {
  name: 'AppNavbar'
}
</script>

<style scoped>
.navbar {
  background-color: #000;
  padding: 20px 0;
  text-align: center;
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.nav-logo a {
  color: white;
  font-size: 2rem;
  font-weight: bold;
  text-decoration: none;
  letter-spacing: 0.5px;
  margin: 0;
}

.nav-menu {
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
  margin-top: 10px;
}

.nav-item {
  margin: 0 15px;
}

.nav-link {
  color: #bbb;
  text-decoration: none;
  font-weight: 500;
  font-size: 1rem;
  transition: color 0.3s ease;
}

.nav-link:hover {
  color: #0a84ff;
}

.dropdown {
  position: relative;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  background-color: #fff;
  min-width: 160px;
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
  border-radius: 4px;
  list-style: none;
  padding: 10px 0;
  opacity: 0;
  visibility: hidden;
  transform: translateY(10px);
  transition: all 0.3s ease;
}

.dropdown:hover .dropdown-menu {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}

.dropdown-link {
  display: block;
  padding: 10px 20px;
  color: #333;
  text-decoration: none;
  transition: background-color 0.3s ease;
}

.dropdown-link:hover {
  background-color: #f5f5f5;
}

.nav-toggle {
  display: none;
  flex-direction: column;
  cursor: pointer;
}

.bar {
  width: 25px;
  height: 3px;
  background-color: white;
  margin: 3px 0;
  transition: 0.3s;
}

@media screen and (max-width: 768px) {
  .nav-container {
    flex-direction: column;
    padding: 10px;
  }

  .nav-menu {
    flex-direction: column;
    width: 100%;
    margin-top: 20px;
    display: none;
  }

  .nav-menu.active {
    display: flex;
  }

  .nav-item {
    margin: 10px 0;
    text-align: center;
  }

  .nav-toggle {
    display: flex;
  }

  .dropdown-menu {
    position: static;
    box-shadow: none;
    opacity: 1;
    visibility: visible;
    transform: none;
    padding: 0;
    background-color: transparent;
  }
}
</style>