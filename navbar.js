// 动态加载导航栏
function loadNavbar() {
  fetch('navbar.html')
    .then(response => response.text())
    .then(html => {
      // 找到导航栏容器并插入内容
      const navContainer = document.getElementById('nav-container');
      if (navContainer) {
        navContainer.innerHTML = html;
      }
    })
    .catch(error => {
      console.error('加载导航栏失败:', error);
    });
}

function loadFooter(){
  fetch('footer.html')
    .then(response => response.text())
    .then(html => {
      const navContainer = document.getElementById('foot-container');
      if (navContainer){
        navContainer.innerHTML = html;
      }
    })
    .catch(error => {
      console.error('加载脚注失败:',error);
    });
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', loadNavbar); 
document.addEventListener('DOMContentLoaded',loadFooter());
