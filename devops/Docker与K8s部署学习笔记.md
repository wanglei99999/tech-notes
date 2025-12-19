# Docker 与 Kubernetes 学习笔记

## 一、Docker 基础概念

### 1.1 什么是 Docker？

Docker 是一个容器化平台，可以把应用程序和它的运行环境打包在一起，确保在任何地方都能一致运行。

**类比理解：**
- 传统部署：搬家时把家具一件件搬，到新家还要重新组装
- Docker 部署：把整个房间打包成集装箱，到哪都是一样的

### 1.2 核心概念

| 概念 | 说明 | 类比 |
|------|------|------|
| 镜像 (Image) | 只读的应用模板，包含代码+环境+依赖 | 软件安装包 |
| 容器 (Container) | 镜像的运行实例，可以启动、停止、删除 | 运行中的程序 |
| 仓库 (Registry) | 存放镜像的服务器 | 应用商店 |
| Dockerfile | 构建镜像的脚本文件 | 安装说明书 |

### 1.3 镜像名称结构

```
registry.example.com:5000/myproject/my-app:v1.0.0
│                    │    │         │      │
│                    │    │         │      └── tag（版本号）
│                    │    │         └── 镜像名
│                    │    └── 命名空间/项目名
│                    └── 端口
└── 仓库地址（Registry）
```

**重点**：Docker 的推送机制是根据镜像名称来的，名称中包含目标仓库地址，push 时自动找到要去的地方。

### 1.4 Docker vs 虚拟机

| | Docker 容器 | 虚拟机 |
|--|------------|--------|
| 启动速度 | 秒级 | 分钟级 |
| 资源占用 | MB 级 | GB 级 |
| 隔离级别 | 进程级 | 系统级 |
| 性能 | 接近原生 | 有损耗 |

## 二、Docker 常用命令

### 2.1 镜像操作

```bash
# 查看本地镜像
docker images

# 搜索镜像
docker search nginx

# 拉取镜像
docker pull nginx:latest
docker pull registry.example.com:5000/myproject/my-app:v1.0

# 删除镜像
docker rmi 镜像ID或名称

# 构建镜像
docker build -t 镜像名:标签 .

# 给镜像打标签（重命名）
docker tag 旧名称 新名称

# 推送镜像到仓库
docker push 镜像名:标签

# 导出镜像为文件
docker save 镜像名:标签 > 文件名.tar
docker save -o 文件名.tar 镜像名:标签

# 从文件导入镜像
docker load < 文件名.tar
docker load -i 文件名.tar
```

### 2.2 容器操作

```bash
# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 运行容器
docker run -d --name 容器名 -p 主机端口:容器端口 镜像名
# -d: 后台运行
# --name: 指定容器名
# -p: 端口映射

# 示例：运行 nginx
docker run -d --name my-nginx -p 8080:80 nginx

# 停止容器
docker stop 容器名或ID

# 启动已停止的容器
docker start 容器名或ID

# 重启容器
docker restart 容器名或ID

# 删除容器
docker rm 容器名或ID
docker rm -f 容器名或ID  # 强制删除运行中的容器

# 进入容器内部
docker exec -it 容器名 /bin/bash
docker exec -it 容器名 sh  # 如果没有 bash

# 查看容器日志
docker logs 容器名
docker logs -f 容器名      # 实时跟踪
docker logs --tail 100 容器名  # 最后100行

# 查看容器详细信息
docker inspect 容器名
```

### 2.3 其他常用命令

```bash
# 查看 Docker 系统信息
docker info

# 查看 Docker 版本
docker version

# 清理无用资源
docker system prune        # 清理停止的容器、无用的网络、悬空镜像
docker system prune -a     # 更彻底的清理

# 查看资源占用
docker stats
```

## 三、Dockerfile 编写

### 3.1 基本结构

```dockerfile
# 基础镜像
FROM openjdk:8-jdk-alpine

# 维护者信息
LABEL maintainer="your-email@example.com"

# 设置工作目录
WORKDIR /app

# 复制文件
COPY target/app.jar app.jar

# 设置环境变量
ENV JAVA_OPTS="-Xms512m -Xmx512m"

# 暴露端口
EXPOSE 8080

# 启动命令
ENTRYPOINT ["java", "-jar", "app.jar"]
```

### 3.2 常用指令

| 指令 | 说明 | 示例 |
|------|------|------|
| `FROM` | 指定基础镜像 | `FROM openjdk:8` |
| `WORKDIR` | 设置工作目录 | `WORKDIR /app` |
| `COPY` | 复制文件到镜像 | `COPY src/ /app/src/` |
| `ADD` | 复制文件（支持URL和解压） | `ADD app.tar.gz /app/` |
| `RUN` | 构建时执行命令 | `RUN apt-get update` |
| `ENV` | 设置环境变量 | `ENV APP_ENV=prod` |
| `ARG` | 构建参数（build时传入） | `ARG VERSION=1.0` |
| `EXPOSE` | 声明端口 | `EXPOSE 8080` |
| `CMD` | 容器启动命令（可被覆盖） | `CMD ["npm", "start"]` |
| `ENTRYPOINT` | 容器启动命令（不易被覆盖） | `ENTRYPOINT ["java", "-jar"]` |

### 3.3 Spring Boot 项目 Dockerfile 示例

```dockerfile
FROM openjdk:8-jdk-alpine

# 构建参数
ARG FINAL_NAME
ARG APP_NAME
ARG APP_VERSION
ARG BUILD_NUMBER
ARG GIT_COMMIT

# 设置时区
RUN apk add --no-cache tzdata \
    && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

WORKDIR /app

# 复制 jar 包
COPY target/${FINAL_NAME}.jar app.jar

# 环境变量
ENV JAVA_OPTS="-Xms512m -Xmx1024m -Djava.security.egd=file:/dev/./urandom"
ENV APP_NAME=${APP_NAME}
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
```

### 3.4 前端项目 Dockerfile 示例

```dockerfile
# 构建阶段
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 运行阶段
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 四、Docker 进阶知识

### 4.1 Docker 网络模式

Docker 提供了多种网络模式，用于容器之间以及容器与外部的通信。

| 网络模式 | 说明 | 使用场景 |
|----------|------|----------|
| `bridge` | 默认模式，容器有独立网络，通过端口映射访问 | 大多数场景 |
| `host` | 容器直接使用宿主机网络，无隔离 | 需要高性能网络 |
| `none` | 无网络，完全隔离 | 安全敏感场景 |
| `container` | 共享另一个容器的网络 | Sidecar 模式 |

```bash
# 使用不同网络模式运行容器
docker run -d --network bridge nginx      # 默认
docker run -d --network host nginx        # 使用宿主机网络
docker run -d --network none nginx        # 无网络

# 创建自定义网络
docker network create my-network

# 容器加入自定义网络（同一网络内的容器可以用容器名互相访问）
docker run -d --name app1 --network my-network nginx
docker run -d --name app2 --network my-network nginx
# app2 可以直接 ping app1

# 查看网络
docker network ls
docker network inspect my-network
```

**自定义网络的好处：**
- 容器之间可以用名称互相访问（自动 DNS）
- 更好的隔离性
- 可以随时连接/断开

### 4.2 数据卷（Volume）

容器默认是无状态的，删除后数据就没了。数据卷用于持久化存储。

**三种挂载方式：**

```bash
# 1. 命名卷（推荐）- Docker 管理存储位置
docker run -d -v my-data:/app/data nginx
# my-data 是卷名，/app/data 是容器内路径

# 2. 绑定挂载 - 指定宿主机路径
docker run -d -v /host/path:/container/path nginx
docker run -d -v $(pwd)/config:/app/config nginx

# 3. 匿名卷 - 自动生成卷名
docker run -d -v /app/data nginx
```

**数据卷常用命令：**

```bash
# 创建卷
docker volume create my-volume

# 查看所有卷
docker volume ls

# 查看卷详情
docker volume inspect my-volume

# 删除卷
docker volume rm my-volume

# 清理无用卷
docker volume prune
```

**实际应用示例：**

```bash
# MySQL 数据持久化
docker run -d \
  --name mysql \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -v mysql-data:/var/lib/mysql \
  -p 3306:3306 \
  mysql:8.0

# 挂载配置文件
docker run -d \
  --name nginx \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/html:/usr/share/nginx/html \
  -p 80:80 \
  nginx
# :ro 表示只读
```

### 4.3 多阶段构建（Multi-stage Build）

多阶段构建可以大幅减小镜像体积，把构建环境和运行环境分开。

**问题：传统构建镜像太大**
```dockerfile
# 不好的做法：镜像包含了构建工具，体积很大
FROM node:16
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
# 镜像包含 node_modules、源码、构建工具... 可能 1GB+
```

**解决：多阶段构建**
```dockerfile
# 阶段1：构建
FROM node:16 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 阶段2：运行（只包含构建产物）
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
# 最终镜像只有几十 MB
```

**Java 项目多阶段构建：**
```dockerfile
# 阶段1：Maven 构建
FROM maven:3.8-openjdk-8 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn package -DskipTests

# 阶段2：运行
FROM openjdk:8-jre-alpine
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
```

**效果对比：**
| | 单阶段构建 | 多阶段构建 |
|--|-----------|-----------|
| Node 项目 | ~1GB | ~30MB |
| Java 项目 | ~500MB | ~150MB |

### 4.4 镜像优化最佳实践

**1. 选择合适的基础镜像**
```dockerfile
# 不推荐：完整版，体积大
FROM node:16           # ~900MB
FROM openjdk:8         # ~500MB

# 推荐：alpine 版，体积小
FROM node:16-alpine    # ~110MB
FROM openjdk:8-jre-alpine  # ~85MB
```

**2. 合并 RUN 指令，减少层数**
```dockerfile
# 不好：每个 RUN 都是一层
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN apt-get clean

# 好：合并成一层，并清理缓存
RUN apt-get update && \
    apt-get install -y curl vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**3. 利用构建缓存，把不常变的放前面**
```dockerfile
# 好的顺序：依赖文件先复制，源码后复制
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build

# 这样改源码时，npm install 可以用缓存
```

**4. 使用 .dockerignore 排除不需要的文件**
```dockerignore
# .dockerignore 文件
node_modules
.git
*.log
*.md
.idea
.vscode
target
dist
```

### 4.5 容器健康检查

```dockerfile
# 在 Dockerfile 中定义健康检查
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

```bash
# 运行时指定健康检查
docker run -d \
  --health-cmd="curl -f http://localhost:8080/health || exit 1" \
  --health-interval=30s \
  --health-timeout=3s \
  --health-retries=3 \
  my-app

# 查看健康状态
docker ps  # STATUS 列会显示 healthy/unhealthy
docker inspect --format='{{.State.Health.Status}}' 容器名
```

### 4.6 容器资源限制

```bash
# 限制内存
docker run -d --memory=512m nginx

# 限制 CPU
docker run -d --cpus=1.5 nginx          # 最多使用 1.5 个 CPU
docker run -d --cpu-shares=512 nginx    # CPU 权重（相对值）

# 组合使用
docker run -d \
  --memory=1g \
  --memory-swap=2g \
  --cpus=2 \
  my-app
```

### 4.7 Docker 日志管理

```bash
# 查看日志
docker logs 容器名
docker logs -f 容器名              # 实时跟踪
docker logs --tail 100 容器名      # 最后 100 行
docker logs --since 1h 容器名      # 最近 1 小时

# 配置日志驱动
docker run -d \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  my-app
# 单个日志文件最大 10MB，最多保留 3 个文件
```

## 五、Docker Compose

### 4.1 什么是 Docker Compose？

用于定义和运行多容器应用的工具，通过一个 YAML 文件配置所有服务。

### 4.2 docker-compose.yml 示例

```yaml
version: '3.8'

services:
  # 后端服务
  backend:
    build: ./backend
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=prod
      - DB_HOST=database
    depends_on:
      - database
    networks:
      - app-network

  # 前端服务
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - app-network

  # 数据库
  database:
    image: postgres:13
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_USER=admin
      - POSTGRES_PASSWORD=secret
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - app-network

networks:
  app-network:
    driver: bridge

volumes:
  db-data:
```

### 4.3 Docker Compose 常用命令

```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 重新构建并启动
docker-compose up -d --build

# 只启动某个服务
docker-compose up -d backend
```

## 五、Kubernetes (K8s) 基础

### 5.1 什么是 Kubernetes？

Kubernetes 是容器编排平台，用于自动化部署、扩展和管理容器化应用。

**类比理解：**
- Docker = 集装箱（打包应用）
- Kubernetes = 港口调度中心（管理成百上千个集装箱）

### 5.2 K8s 核心概念

| 概念 | 说明 |
|------|------|
| **Cluster** | 集群，由多个节点组成 |
| **Node** | 节点，运行容器的服务器（物理机或虚拟机） |
| **Pod** | 最小部署单元，包含一个或多个容器 |
| **Deployment** | 部署，管理 Pod 的副本数量和更新策略 |
| **Service** | 服务，为 Pod 提供稳定的访问入口 |
| **Namespace** | 命名空间，用于隔离资源 |
| **ConfigMap** | 配置管理，存储非敏感配置 |
| **Secret** | 密钥管理，存储敏感信息 |
| **Ingress** | 入口，管理外部访问的路由规则 |

### 5.3 K8s 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                        │
│  │   Master Node   │  控制平面                               │
│  │  ┌───────────┐  │                                        │
│  │  │ API Server│  │  ← 所有操作的入口                       │
│  │  ├───────────┤  │                                        │
│  │  │ Scheduler │  │  ← 决定 Pod 运行在哪个节点              │
│  │  ├───────────┤  │                                        │
│  │  │Controller │  │  ← 维护集群状态                         │
│  │  ├───────────┤  │                                        │
│  │  │   etcd    │  │  ← 存储集群数据                         │
│  │  └───────────┘  │                                        │
│  └─────────────────┘                                        │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │   Worker Node   │  │   Worker Node   │  工作节点          │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                   │
│  │  │  kubelet  │  │  │  │  kubelet  │  │  ← 管理本节点容器  │
│  │  ├───────────┤  │  │  ├───────────┤  │                   │
│  │  │ kube-proxy│  │  │  │ kube-proxy│  │  ← 网络代理       │
│  │  ├───────────┤  │  │  ├───────────┤  │                   │
│  │  │  Pod Pod  │  │  │  │  Pod Pod  │  │  ← 运行的容器     │
│  │  └───────────┘  │  │  └───────────┘  │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 K8s 的核心能力

1. **自动调度**：自动把容器分配到合适的节点
2. **自愈能力**：容器挂了自动重启，节点挂了自动迁移
3. **水平扩展**：一键扩容/缩容
4. **滚动更新**：不停服更新应用
5. **服务发现**：自动注册和发现服务
6. **负载均衡**：自动分发流量

## 六、K8s 资源配置文件

### 6.1 Deployment 示例

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: default
spec:
  replicas: 3                    # 副本数量
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: registry.example.com:5000/myproject/my-app:v1.0
        ports:
        - containerPort: 8080
        resources:
          requests:              # 最小资源
            memory: "256Mi"
            cpu: "250m"
          limits:                # 最大资源
            memory: "512Mi"
            cpu: "500m"
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "prod"
```

### 6.2 Service 示例

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
  - port: 80              # Service 端口
    targetPort: 8080      # Pod 端口
  type: ClusterIP         # 类型：ClusterIP/NodePort/LoadBalancer
```

### 6.3 ConfigMap 示例

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: my-app-config
data:
  application.yml: |
    server:
      port: 8080
    spring:
      profiles:
        active: prod
```

## 七、kubectl 常用命令

### 7.1 查看资源

```bash
# 查看所有 Pod
kubectl get pods
kubectl get pods -n 命名空间
kubectl get pods -o wide          # 显示更多信息

# 查看所有 Deployment
kubectl get deployments

# 查看所有 Service
kubectl get services

# 查看所有资源
kubectl get all

# 查看资源详情
kubectl describe pod Pod名称
kubectl describe deployment 部署名称
```

### 7.2 操作资源

```bash
# 应用配置文件
kubectl apply -f deployment.yaml

# 删除资源
kubectl delete -f deployment.yaml
kubectl delete pod Pod名称

# 扩缩容
kubectl scale deployment 部署名称 --replicas=5

# 更新镜像
kubectl set image deployment/部署名称 容器名=新镜像:标签

# 回滚
kubectl rollout undo deployment/部署名称
kubectl rollout history deployment/部署名称
```

### 7.3 调试排查

```bash
# 查看 Pod 日志
kubectl logs Pod名称
kubectl logs -f Pod名称           # 实时跟踪
kubectl logs Pod名称 -c 容器名    # 多容器时指定

# 进入 Pod 内部
kubectl exec -it Pod名称 -- /bin/bash

# 端口转发（本地调试）
kubectl port-forward Pod名称 8080:8080
```

## 八、企业内网部署流程（通用场景）

### 8.1 整体流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                        部署流程                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  开发环境                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
│  │ 代码开发 │ → │Maven 打包│ → │Docker   │                       │
│  │         │    │         │    │Build    │                       │
│  └─────────┘    └─────────┘    └────┬────┘                       │
│                                     │                            │
│                              docker save                         │
│                                     ↓                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    文件传输链路                           │    │
│  │  跳板机 → 中转机 → 堡垒机 → 运维机                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                     │                            │
│                              docker load                         │
│                              docker push                         │
│                                     ↓                            │
│  内网环境                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
│  │ Harbor  │ → │   K8s   │ → │ 应用运行 │                       │
│  │ 仓库    │    │ 拉取镜像 │    │         │                       │
│  └─────────┘    └─────────┘    └─────────┘                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 8.2 开发环境操作

```bash
# 1. Maven 打包
mvn clean package -DskipTests

# 2. 构建 Docker 镜像
docker build -t registry.example.com:5000/myproject/my-gateway:20250703 \
  --build-arg FINAL_NAME=my-gateway \
  --build-arg APP_NAME=my-gateway \
  --build-arg APP_VERSION=1.0 \
  --build-arg BUILD_NUMBER=1 \
  --build-arg GIT_COMMIT=abc123 .

# 3. 导出镜像
docker save registry.example.com:5000/myproject/my-gateway:20250703 > my-gateway-20250703.tar
```

### 8.3 文件传输链路（内外网隔离场景）

```
1. 开发机 → 跳板机
   工具：WinSCP / SCP

2. 跳板机 → 中转机
   通过网络映射或 SFTP

3. 中转机 → 运维机
   通过堡垒机访问
   工具：SFTP
```

### 8.4 运维机操作

```bash
# 切换到 root
su root

# 导入镜像
docker load -i my-gateway-20250703.tar

# 推送到 Harbor
docker push registry.example.com:5000/myproject/my-gateway:20250703
```

### 8.5 K8s 平台更新

1. 通过堡垒机访问 K8s 管理平台
2. 找到对应的 Deployment
3. 修改镜像版本号为新版本
4. 保存，K8s 自动拉取新镜像并滚动更新

## 九、常见问题排查

### 9.1 镜像拉取失败

```bash
# 检查镜像是否存在
docker pull 镜像地址

# 检查仓库登录状态
docker login 仓库地址

# 查看 Pod 事件
kubectl describe pod Pod名称
```

### 9.2 容器启动失败

```bash
# 查看容器日志
kubectl logs Pod名称

# 查看之前崩溃的容器日志
kubectl logs Pod名称 --previous

# 进入容器排查
kubectl exec -it Pod名称 -- sh
```

### 9.3 服务无法访问

```bash
# 检查 Service
kubectl get svc

# 检查 Endpoints
kubectl get endpoints

# 检查 Pod 是否正常
kubectl get pods -o wide
```

## 十、Docker 实战案例

### 10.1 本地开发环境搭建

**一键启动开发所需的中间件：**

```yaml
# docker-compose-dev.yml
version: '3.8'

services:
  # MySQL 数据库
  mysql:
    image: mysql:8.0
    container_name: dev-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root123
      MYSQL_DATABASE: myapp
      TZ: Asia/Shanghai
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # 初始化脚本
    command: --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci

  # Redis 缓存
  redis:
    image: redis:7-alpine
    container_name: dev-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  # PostgreSQL（如果需要）
  postgres:
    image: postgres:13
    container_name: dev-postgres
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin123
      POSTGRES_DB: myapp
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    container_name: dev-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./html:/usr/share/nginx/html

  # RabbitMQ 消息队列
  rabbitmq:
    image: rabbitmq:3-management
    container_name: dev-rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin123
    ports:
      - "5672:5672"    # AMQP 端口
      - "15672:15672"  # 管理界面
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq

  # Elasticsearch
  elasticsearch:
    image: elasticsearch:7.17.0
    container_name: dev-es
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    volumes:
      - es-data:/usr/share/elasticsearch/data

volumes:
  mysql-data:
  redis-data:
  postgres-data:
  rabbitmq-data:
  es-data:
```

```bash
# 启动所有服务
docker-compose -f docker-compose-dev.yml up -d

# 只启动需要的服务
docker-compose -f docker-compose-dev.yml up -d mysql redis

# 停止并清理
docker-compose -f docker-compose-dev.yml down
```

### 10.2 常用中间件快速启动命令

**MySQL：**
```bash
docker run -d \
  --name mysql \
  -e MYSQL_ROOT_PASSWORD=root123 \
  -e MYSQL_DATABASE=myapp \
  -p 3306:3306 \
  -v mysql-data:/var/lib/mysql \
  mysql:8.0 \
  --character-set-server=utf8mb4 \
  --collation-server=utf8mb4_unicode_ci
```

**Redis：**
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  -v redis-data:/data \
  redis:7-alpine \
  redis-server --appendonly yes
```

**PostgreSQL：**
```bash
docker run -d \
  --name postgres \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=admin123 \
  -e POSTGRES_DB=myapp \
  -p 5432:5432 \
  -v postgres-data:/var/lib/postgresql/data \
  postgres:13
```

**MongoDB：**
```bash
docker run -d \
  --name mongo \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=admin123 \
  -p 27017:27017 \
  -v mongo-data:/data/db \
  mongo:6
```

**Nginx：**
```bash
docker run -d \
  --name nginx \
  -p 80:80 \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v $(pwd)/html:/usr/share/nginx/html \
  nginx:alpine
```

### 10.3 调试技巧

**1. 查看容器内部文件**
```bash
# 进入容器
docker exec -it 容器名 sh

# 不进入容器，直接执行命令
docker exec 容器名 cat /etc/nginx/nginx.conf
docker exec 容器名 ls -la /app
```

**2. 从容器复制文件**
```bash
# 从容器复制到宿主机
docker cp 容器名:/path/in/container /path/on/host

# 从宿主机复制到容器
docker cp /path/on/host 容器名:/path/in/container
```

**3. 查看容器资源使用**
```bash
# 实时监控
docker stats

# 查看单个容器
docker stats 容器名
```

**4. 查看容器进程**
```bash
docker top 容器名
```

**5. 查看容器变更**
```bash
# 查看容器文件系统变更
docker diff 容器名
```

**6. 导出容器为镜像**
```bash
# 把运行中的容器保存为新镜像（包含修改）
docker commit 容器名 新镜像名:标签
```

### 10.4 镜像版本管理策略

**版本号命名规范：**
```
镜像名:YYYYMMDD      # 日期版本：my-app:20250703
镜像名:v1.0.0        # 语义化版本：my-app:v1.0.0
镜像名:git-abc123    # Git 提交号：my-app:git-abc123
镜像名:latest        # 最新版本（不推荐生产使用）
```

**推荐做法：**
```bash
# 构建时同时打多个标签
docker build -t my-app:20250703 -t my-app:v1.2.0 -t my-app:latest .

# 或者构建后追加标签
docker tag my-app:20250703 my-app:v1.2.0
docker tag my-app:20250703 my-app:latest
```

## 十一、Docker 安全最佳实践

### 11.1 镜像安全

```dockerfile
# 1. 使用官方镜像或可信来源
FROM nginx:alpine  # 官方镜像

# 2. 指定具体版本，不用 latest
FROM node:16.20.0-alpine  # 具体版本

# 3. 使用非 root 用户运行
FROM node:16-alpine
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser
WORKDIR /app
COPY --chown=appuser:appgroup . .

# 4. 最小化安装，只装必要的包
RUN apk add --no-cache curl  # --no-cache 不保留缓存
```

### 11.2 运行时安全

```bash
# 1. 以只读方式运行
docker run --read-only my-app

# 2. 限制资源
docker run --memory=512m --cpus=1 my-app

# 3. 不使用特权模式
docker run my-app  # 不要加 --privileged

# 4. 限制网络
docker run --network=none my-app  # 无网络访问

# 5. 删除不需要的能力
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE my-app
```

### 11.3 敏感信息处理

```bash
# 不好：密码写在 Dockerfile 或命令行
docker run -e PASSWORD=secret123 my-app  # 会被记录在历史

# 好：使用 Docker secrets 或环境变量文件
echo "PASSWORD=secret123" > .env
docker run --env-file .env my-app

# 更好：使用 Docker secrets（Swarm 模式）或外部密钥管理
```

## 十二、关键概念补充

### 12.1 Docker 与 K8s 的关系

```
┌─────────────────────────────────────────────────────────────┐
│                        技术演进                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  单机时代          集群时代           云原生时代             │
│                                                             │
│  ┌─────────┐      ┌─────────┐       ┌─────────┐            │
│  │ Docker  │  →   │ Docker  │   →   │   K8s   │            │
│  │ 单容器  │      │ Compose │       │ 集群编排 │            │
│  └─────────┘      └─────────┘       └─────────┘            │
│                                                             │
│  1个容器          多个容器           成百上千容器            │
│  1台机器          1台机器            多台机器               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 为什么内网部署流程复杂？

企业内网和外网通常完全隔离：
- 不能直接传文件
- 堡垒机是唯一入口
- 所有操作都有审计记录
- 需要通过多台中转机器

### 12.3 各组件作用

| 组件 | 作用 |
|------|------|
| 堡垒机 | 安全跳板，二次认证，操作审计 |
| 运维机 | 内网 Linux 服务器，执行 docker 命令 |
| Harbor | 企业级 Docker 镜像仓库 |
| K8s 平台 | 容器编排管理的 Web 界面 |
| 跳板机/中转机 | 内外网之间的文件传输桥梁 |

## 十三、学习资源推荐

### 13.1 官方文档
- Docker 官方文档：https://docs.docker.com/
- Kubernetes 官方文档：https://kubernetes.io/docs/
- Docker Hub：https://hub.docker.com/

### 13.2 实践建议

1. **先学 Docker，再学 K8s**
   - Docker 是基础，K8s 建立在 Docker 之上

2. **动手实践**
   - 本地搭建开发环境
   - 尝试容器化自己的项目
   - 用 Docker Compose 编排多个服务

3. **循序渐进**
   ```
   入门：docker run/build/push
      ↓
   进阶：Dockerfile 优化、网络、数据卷
      ↓
   实战：Docker Compose 多服务编排
      ↓
   高级：Kubernetes 集群部署
   ```

4. **常见学习路径**
   - 开发人员：Docker 基础 → Dockerfile → Docker Compose
   - 运维人员：Docker 基础 → K8s 部署 → 监控告警 → CI/CD

---
*最后更新：2024年12月19日*
