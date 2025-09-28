<template>
  <nav class="navbar">
    <div class="nav-container">
      <div class="nav-logo">
  <router-link to="/">LipG</router-link>
</div>
      <ul class="nav-menu">
        <li class="nav-item">
          <router-link to="/" class="nav-link">自我介绍</router-link>
        </li>
        <li class="nav-item dropdown">
          <a href="#" class="nav-link">作为助教 <i class="fas fa-chevron-down"></i></a>
          <ul class="dropdown-menu">
            <li><router-link to="/teaching-assistant" class="dropdown-link">助教寄语</router-link></li>
            <li><a href="/ProblemSession1.pdf" target="_blank" class="dropdown-link">第一次习题课</a></li>
          </ul>
        </li>

        <!-- <li class="nav-item">
          <router-link to="/intro" class="nav-link">建立网站</router-link>
        </li> -->
        <!-- <li class="nav-item">
          <router-link to="/daily-insight" class="nav-link">日常思考</router-link>
        </li> -->
        <li class="nav-item dropdown">
          <a href="#" class="nav-link">技术博客 <i class="fas fa-chevron-down"></i></a>
          <ul class="dropdown-menu">
            <!-- <li><router-link to="/leetcode-blog" class="dropdown-link">LeetCode题解</router-link></li> -->
            <li><router-link to="/ml-blog" class="dropdown-link">机器学习</router-link></li>
            <li><router-link to="/dl-blog" class="dropdown-link">深度学习</router-link></li>
            <!-- <li><router-link to="/new-blog-post" class="dropdown-link">新文章</router-link></li>
            <li><router-link to="/markdown-blog" class="dropdown-link">Markdown博客</router-link></li> -->
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
  background: linear-gradient(135deg, #2c3e50, #1a2a3a);
  padding: 15px 0;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 1000;
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
  font-size: 1.8rem;
  font-weight: 700;
  text-decoration: none;
  letter-spacing: 1px;
  margin: 0;
  background: linear-gradient(90deg, #3498db, #2c3e50);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.nav-menu {
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
}

.nav-item {
  margin: 0 10px;
  position: relative;
}

.nav-link {
  color: #ecf0f1;
  text-decoration: none;
  font-weight: 500;
  font-size: 1.05rem;
  padding: 8px 12px;
  border-radius: 4px;
  transition: all 0.3s ease;
  display: block;
}

.nav-link:hover {
  color: #3498db;
  background-color: rgba(255, 255, 255, 0.1);
  transform: translateY(-2px);
}

.nav-link.router-link-exact-active {
  color: #3498db;
  background-color: rgba(52, 152, 219, 0.1);
}

.dropdown {
  position: relative;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  background: linear-gradient(135deg, #34495e, #2c3e50);
  min-width: 180px;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  list-style: none;
  padding: 10px 0;
  opacity: 0;
  visibility: hidden;
  transform: translateY(10px);
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  margin-top: 5px;
}

.dropdown:hover .dropdown-menu {
  opacity: 1;
  visibility: visible;
  transform: translateY(0);
}

.dropdown-link {
  display: block;
  padding: 12px 20px;
  color: #ecf0f1;
  text-decoration: none;
  transition: all 0.3s ease;
  font-size: 1rem;
}

.dropdown-link:hover {
  background-color: rgba(52, 152, 219, 0.2);
  color: #3498db;
  padding-left: 25px;
}

.nav-toggle {
  display: none;
  flex-direction: column;
  cursor: pointer;
}

.bar {
  width: 25px;
  height: 3px;
  background-color: #ecf0f1;
  margin: 3px 0;
  transition: 0.4s;
  border-radius: 2px;
}

/* 汉堡菜单动画 */
.nav-toggle.active .bar:nth-child(1) {
  transform: rotate(-45deg) translate(-5px, 6px);
}

.nav-toggle.active .bar:nth-child(2) {
  opacity: 0;
}

.nav-toggle.active .bar:nth-child(3) {
  transform: rotate(45deg) translate(-5px, -6px);
}

@media screen and (max-width: 768px) {
  .navbar {
    padding: 10px 0;
  }
  
  .nav-container {
    flex-direction: column;
    padding: 10px;
  }

  .nav-menu {
    flex-direction: column;
    width: 100%;
    margin-top: 20px;
    display: none;
    background: rgba(44, 62, 80, 0.95);
    border-radius: 10px;
    padding: 15px 0;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
  }

  .nav-menu.active {
    display: flex;
  }

  .nav-item {
    margin: 5px 0;
    text-align: center;
  }
  
  .nav-link {
    padding: 12px 20px;
    margin: 5px 20px;
    border-radius: 6px;
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
    margin-top: 0;
    border-radius: 0;
    min-width: auto;
  }
  
  .dropdown-link {
    padding: 10px 30px;
  }
  
  .dropdown-link:hover {
    padding-left: 35px;
  }
}
</style>