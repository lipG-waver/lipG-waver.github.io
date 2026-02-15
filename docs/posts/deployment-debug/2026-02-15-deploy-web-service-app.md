# GRE单词学习系统——从本地项目到云端部署全记录

> 作者：Yunlong Zhou | 日期：2026年2月15日
>
> 本文记录了将一个基于React + Express + SQLite的全栈应用，从本地开发环境部署到阿里云ECS服务器的完整过程，包括遇到的问题和解决方案。适合有一定编程基础但没有部署经验的同学参考。

---

## 一、项目背景

这是一个AI驱动的GRE单词学习追踪系统，核心技术栈：

- **前端**：React 18 + Tailwind CSS + Axios
- **后端**：Express 5 + SQLite3 + JWT认证
- **AI**：通义千问VL视觉模型（题目OCR识别与解析）
- **存储**：阿里云OSS（图片上传）

项目原本只在本地 `localhost:3003` 运行，所有API地址都是硬编码的。部署到服务器需要解决：环境变量配置、前后端同源部署、进程守护、反向代理等问题。

---

## 二、服务器选型

**最终选择**：阿里云ECS，2核4G内存，50G系统盘，Ubuntu 22.04，上海地域。

选择依据：

- **2核4G**：Node.js进程约150MB，Nginx约20MB，系统约500MB，`npm run build`构建React时峰值吃内存可达1GB+，4G留够余量
- **上海地域**：离复旦最近，通义千问API也在阿里云，内网调用更快
- **Ubuntu 22.04**：Node.js生态支持最好的Linux发行版

费用：约60-80元/月，团队6人分摊每人约10-13元/月。

---

## 三、代码改造——从localhost到生产环境

### 3.1 前端：统一API地址管理

**问题**：14个前端组件文件里散落着硬编码的 `http://localhost:3003`，部署后全部失效。

**解决**：创建统一配置文件 `frontend/src/config/api.js`：

```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || '';
export default API_BASE_URL;
```

所有组件改为：

```javascript
import API_BASE_URL from '../config/api';
// 原来：axios.get('http://localhost:3003/api/words')
// 现在：axios.get(`${API_BASE_URL}/api/words`)
```

**原理**：`REACT_APP_API_URL` 为空字符串时，axios会发送相对路径请求（如 `/api/words`），在生产环境中由Nginx代理到后端。开发环境可以通过 `setupProxy.js` 或设置环境变量来代理。

### 3.2 后端：环境变量驱动

**问题**：数据库路径、CORS域名、端口号都是写死的。

**解决**：用 `dotenv` 从 `.env` 文件读取配置：

```javascript
// 原来
app.use(cors({ origin: 'http://localhost:3002' }));
const db = new sqlite3.Database('gre_words.db');

// 现在
const allowedOrigins = (process.env.CORS_ORIGIN || 'http://localhost:3002').split(',');
const db = new sqlite3.Database(process.env.DB_PATH || 'gre_words.db');
```

`.env` 文件模板（`.env.example`）：

```
NODE_ENV=production
PORT=3003
CORS_ORIGIN=http://47.118.31.151
JWT_SECRET=your-random-secret-key
ADMIN_EMAIL=admin@gre.app
ADMIN_PASSWORD=admin123
DB_PATH=./data/gre_words.db
DASHSCOPE_API_KEY=your_api_key
```

> **安全提醒**：`.env` 文件包含密钥，绝对不能提交到Git。已在 `.gitignore` 中排除。

### 3.3 用户系统与角色路由

**需求**：管理员能上传题目、管理词库；普通用户只能背单词、做题。

**实现架构**：

```
App.js
  ├── 未登录 → LoginPage（注册/登录）
  ├── admin角色 → AdminLayout（题目导入、单词管理、全部功能）
  └── user角色 → UserLayout（背单词、做题、生词本、底部Tab栏）
```

后端认证流程：

1. 注册/登录 → 服务器用 `crypto.pbkdf2` 哈希密码，返回JWT token
2. 前端把token存入 `localStorage`，每次请求带 `Authorization: Bearer xxx` 头
3. 敏感API（如通义千问调用、批量删除）加 `authMiddleware` + `adminMiddleware` 保护
4. 首次启动自动创建默认管理员账户

### 3.4 生产环境静态文件服务

**关键设计**：生产环境下，Express同时提供API和React构建产物，只需一个端口。

```javascript
if (process.env.NODE_ENV === 'production') {
  // 提供React build出的静态文件（JS/CSS/图片等）
  app.use(express.static(path.join(__dirname, '../frontend/build')));
}

// ...所有API路由...

if (process.env.NODE_ENV === 'production') {
  // 所有未匹配API的请求都返回index.html（SPA路由支持）
  app.get('{*path}', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/build', 'index.html'));
  });
}
```

> **踩坑记录**：Express 5 的路由库 `path-to-regexp` 升级后不再支持 `app.get('*', ...)`，必须写成 `app.get('{*path}', ...)`，否则会报 `PathError: Missing parameter name at index 1`。这是Express 4→5的Breaking Change。

---

## 四、服务器部署——逐步操作详解

### 4.1 SSH登录服务器

```bash
ssh root@47.118.31.151
```

`ssh` 是Secure Shell协议，通过加密通道远程登录服务器。`root` 是Linux超级管理员用户，`@` 后面是服务器的公网IP地址。

### 4.2 安装基础软件

```bash
apt update && apt install -y nginx
```

- `apt update`：更新Ubuntu的软件包索引，让系统知道有哪些软件可以安装
- `apt install -y nginx`：安装Nginx web服务器。`-y` 表示自动确认，不需要手动输入yes
- **Nginx的作用**：作为反向代理，监听80端口（HTTP默认端口），把请求转发给运行在3003端口的Node.js应用。用户访问 `http://47.118.31.151` → Nginx(80) → Express(3003)

### 4.3 上传代码到服务器

在**本地电脑**（不是服务器）执行：

```bash
scp WordRecitation-prod.tar.gz root@47.118.31.151:/root/
```

`scp` (Secure Copy) 通过SSH加密通道把本地文件复制到远程服务器。格式是 `scp 本地文件 用户@IP:远程路径`。

### 4.4 解压代码

```bash
cd /root
tar xzf WordRecitation-prod.tar.gz
cd WordRecitation-prod
```

- `tar xzf`：解压 `.tar.gz` 压缩包。`x`=解压，`z`=gzip格式，`f`=指定文件名
- `cd`：切换目录（Change Directory）

### 4.5 配置环境变量

```bash
cp .env.example backend/.env
vim backend/.env
```

- `cp`：复制文件，从模板创建实际配置文件
- `vim`：Linux文本编辑器。按 `i` 进入编辑模式，修改完按 `Esc` 然后输入 `:wq` 保存退出

### 4.6 安装Node.js

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

- `curl -fsSL`：从网上下载NodeSource的安装脚本。`-f`=失败时静默，`-s`=静默模式，`-S`=出错时显示错误，`-L`=跟随重定向
- `| sudo -E bash -`：管道符 `|` 把下载的脚本传给bash执行，`sudo -E` 以管理员权限运行并保留环境变量
- 这两条命令的作用是添加Node.js 20.x的官方软件源，然后安装

### 4.7 安装依赖并构建

```bash
# 后端依赖
cd backend && npm install --production && cd ..

# 前端依赖 + 构建
cd frontend && npm install && REACT_APP_API_URL="" npm run build && cd ..
```

- `npm install --production`：只安装 `dependencies`，不安装 `devDependencies`（如nodemon），节省空间
- `npm run build`：React的构建命令，把JSX/ES6源码编译成浏览器能直接运行的HTML/CSS/JS文件，输出到 `frontend/build/` 目录
- `REACT_APP_API_URL=""`：构建时将API地址设为空字符串，这样前端会使用相对路径请求API

> **耗时提醒**：`npm run build` 在2核4G服务器上大约需要2-3分钟，会看到 `Creating an optimized production build...`，耐心等待直到出现 `Compiled successfully`。

### 4.8 初始化数据库

```bash
cd backend
node initDatabase.js        # 创建words、questions等表
node initPomodoroDatabase.js # 创建番茄钟会话表
node initVocabularyDatabase.js # 创建生词本表
cd ..
```

这三个脚本执行 `CREATE TABLE IF NOT EXISTS` SQL语句，在 `backend/data/` 目录下生成SQLite数据库文件。`IF NOT EXISTS` 保证重复运行不会覆盖已有数据。

### 4.9 用PM2启动应用

```bash
# 安装PM2
sudo npm install -g pm2

# 启动应用
cd backend
NODE_ENV=production pm2 start server.js --name gre-word-app

# 设置开机自启
pm2 save
pm2 startup
```

**PM2是什么**：Node.js的进程管理器。直接用 `node server.js` 启动的话，SSH断开连接后进程就会终止。PM2解决了三个问题：

1. **守护进程**：应用在后台持续运行，不受SSH会话影响
2. **崩溃自重启**：如果应用报错退出，PM2会自动重新启动它
3. **开机自启**：`pm2 startup` 生成systemd服务，服务器重启后自动拉起应用

常用PM2命令：

| 命令 | 作用 |
|------|------|
| `pm2 status` | 查看所有应用状态 |
| `pm2 logs gre-word-app` | 查看实时日志 |
| `pm2 restart gre-word-app` | 重启应用 |
| `pm2 stop gre-word-app` | 停止应用 |
| `pm2 monit` | 实时监控CPU/内存 |

### 4.10 配置Nginx反向代理

```bash
cp nginx.conf /etc/nginx/sites-available/gre-app
ln -sf /etc/nginx/sites-available/gre-app /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
```

逐行解释：

- `cp`：把我们的Nginx配置复制到Nginx的配置目录
- `ln -sf`：创建软链接（快捷方式）。Nginx只会加载 `sites-enabled/` 目录下的配置，通过软链接指向 `sites-available/` 中的实际文件
- `rm -f sites-enabled/default`：删除Nginx默认的欢迎页配置，否则会和我们的配置冲突
- `nginx -t`：测试配置语法是否正确（不会影响正在运行的Nginx）
- `systemctl reload nginx`：重新加载配置，不中断现有连接

Nginx配置文件的核心内容：

```nginx
server {
    listen 80;                        # 监听80端口
    server_name 47.118.31.151;        # 匹配的域名/IP

    location / {                      # 所有请求
        proxy_pass http://127.0.0.1:3003;  # 转发到Express
        proxy_set_header Host $host;       # 传递原始域名
        proxy_set_header X-Real-IP $remote_addr; # 传递真实客户端IP
    }
}
```

**为什么不直接让Express监听80端口？** 因为Nginx擅长处理静态文件、SSL终止、Gzip压缩、负载均衡，Express擅长处理业务逻辑。分工明确，各司其职。

### 4.11 开放阿里云安全组

这一步在阿里云Web控制台操作，不是终端命令：

**路径**：ECS控制台 → 实例 → 安全组 → 入方向规则 → 手动添加

| 协议 | 端口范围 | 授权对象 | 说明 |
|------|----------|----------|------|
| TCP | 80 | 0.0.0.0/0 | HTTP访问 |
| TCP | 443 | 0.0.0.0/0 | HTTPS访问（未来用） |

> **安全组 vs 防火墙**：阿里云安全组是云平台层面的网络ACL，在流量到达服务器之前就过滤。即使服务器上没有开防火墙（ufw/iptables），安全组不开端口照样连不上。

---

## 五、踩坑记录

### 坑1：Express 5 路由通配符语法变更

**现象**：PM2显示应用online但不断重启，curl localhost:3003 连接被拒。

**日志**：`PathError [TypeError]: Missing parameter name at index 1: *`

**原因**：Express 5 底层的 `path-to-regexp` 库从v1升级到v8，不再支持 `app.get('*', handler)`。

**修复**：改为 `app.get('{*path}', handler)`。

**教训**：Express 5还比较新（2025年发布），很多Express 4的写法不再兼容。如果看到 `path-to-regexp` 相关错误，先查 [Express 5迁移指南](https://expressjs.com/en/guide/migrating-5.html)。

### 坑2：Nginx默认页覆盖自定义配置

**现象**：配置了Nginx代理，但访问IP显示 "Welcome to nginx!"。

**原因**：`/etc/nginx/sites-enabled/default` 仍然存在，且优先匹配。

**修复**：`rm -f /etc/nginx/sites-enabled/default && systemctl reload nginx`

### 坑3：前端未构建

**现象**：Nginx代理正常，API健康检查通过，但访问首页返回404。

**原因**：`frontend/build/` 目录不存在，Express的 `express.static` 找不到文件。

**修复**：`cd frontend && npm install && REACT_APP_API_URL="" npm run build`

---

## 六、最终架构图

```
用户浏览器
    │
    ▼ HTTP请求 (端口80)
┌─────────┐
│  Nginx  │  反向代理，监听80端口
└────┬────┘
     │ proxy_pass → 127.0.0.1:3003
     ▼
┌──────────────────────────┐
│    Express (PM2守护)      │  端口3003
│  ┌─────────────────────┐ │
│  │ 静态文件服务          │ │  → frontend/build/ (React产物)
│  │ /api/* 路由          │ │  → 业务逻辑
│  │ JWT认证中间件         │ │  → 权限控制
│  │ SPA catch-all        │ │  → 所有非API请求返回index.html
│  └─────────────────────┘ │
│          │               │
│          ▼               │
│  ┌──────────────┐        │
│  │   SQLite     │        │  本地数据库文件
│  │ gre_words.db │        │
│  │ pomodoro.db  │        │
│  │ vocabulary.db│        │
│  └──────────────┘        │
│          │               │
│          ▼               │
│  ┌──────────────┐        │
│  │ 外部API调用   │        │
│  │ 通义千问VL    │        │  题目OCR识别
│  │ 阿里云OSS    │        │  图片存储
│  └──────────────┘        │
└──────────────────────────┘
```

---

## 七、日常运维速查

```bash
# 查看应用状态
pm2 status

# 查看实时日志
pm2 logs gre-word-app

# 重启应用（改了后端代码后）
pm2 restart gre-word-app

# 重新构建前端（改了前端代码后）
cd /root/WordRecitation-prod/frontend
npm run build
pm2 restart gre-word-app

# 更新代码（从本地上传新版本后）
cd /root/WordRecitation-prod/backend
npm install --production
cd ../frontend
npm install && npm run build
pm2 restart gre-word-app

# 查看Nginx日志
tail -f /var/log/nginx/error.log
```

---

## 八、后续计划

- [ ] 配置域名 + HTTPS（Let's Encrypt免费证书）
- [ ] SQLite迁移到PostgreSQL（支持并发写入）
- [ ] 升级间隔复习算法（SM-2/FSRS）
- [ ] 接入更多AI模型（GPT-4o、Claude API）
- [ ] 响应式移动端优化

---

*本文是复旦大学2026年"卓越杯"创新创业大赛（AI+专项赛道）参赛项目的技术文档之一。*