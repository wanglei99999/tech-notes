# Git 与 GitHub 使用指南

## 一、基础概念

### 1.1 Git vs GitHub
- **Git**：分布式版本控制系统，在本地管理代码版本
- **GitHub**：基于 Git 的远程代码托管平台，用于团队协作和代码备份

### 1.2 核心术语
| 术语 | 说明 |
|------|------|
| Repository (仓库) | 存放项目代码的地方，包含所有版本历史 |
| Branch (分支) | 代码的平行版本，可独立开发不同功能 |
| Commit (提交) | 保存代码变更的快照，带有描述信息 |
| Remote (远程) | 远程仓库，通常指 GitHub 上的仓库 |
| Origin | 远程仓库的默认别名 |
| HEAD | 当前所在的提交/分支位置 |
| Clone (克隆) | 从远程下载完整仓库到本地 |
| Pull (拉取) | 从远程获取更新并合并到本地 |
| Push (推送) | 将本地提交上传到远程 |
| Merge (合并) | 将一个分支的代码合并到另一个分支 |

### 1.3 工作区域
```
工作区 (Working Directory)     暂存区 (Staging Area)     本地仓库 (Local Repo)     远程仓库 (Remote)
        │                              │                         │                        │
        │──── git add ────────────────>│                         │                        │
        │                              │──── git commit ────────>│                        │
        │                              │                         │──── git push ─────────>│
        │<─────────────────────────────────── git pull ──────────────────────────────────│
```

## 二、常用命令

### 2.1 初始化与克隆
```bash
# 初始化新仓库
git init

# 克隆远程仓库
git clone https://github.com/用户名/仓库名.git

# 浅克隆（只下载最新版本，速度快）
git clone --depth 1 https://github.com/用户名/仓库名.git

# 克隆指定分支
git clone -b 分支名 https://github.com/用户名/仓库名.git
```

### 2.2 日常操作
```bash
# 查看状态
git status

# 添加文件到暂存区
git add 文件名           # 添加单个文件
git add .               # 添加所有变更

# 提交到本地仓库
git commit -m "提交说明"

# 推送到远程
git push
git push origin 分支名   # 推送到指定分支

# 拉取远程更新
git pull
git pull origin 分支名   # 从指定分支拉取
```

### 2.3 查看信息
```bash
# 查看提交历史
git log
git log --oneline        # 简洁模式
git log --graph          # 图形化显示分支

# 查看远程仓库信息
git remote -v

# 查看所有分支
git branch -a
```

## 三、分支管理（重点）

### 3.1 分支的本质
分支就像**平行世界**，每个分支有自己独立的代码版本，互不影响。

```
                    ┌── feature-A (新功能开发)
                    │
master ─────────────┼── feature-B (另一个功能)
(主分支)             │
                    └── bugfix (修复bug)
```

### 3.2 分支操作命令
```bash
# 查看分支
git branch              # 本地分支
git branch -a           # 所有分支（含远程）

# 创建分支
git branch 分支名        # 只创建，不切换

# 切换分支
git checkout 分支名
git switch 分支名        # Git 2.23+ 新命令

# 创建并切换
git checkout -b 分支名
git switch -c 分支名

# 删除分支
git branch -d 分支名     # 删除已合并的分支
git branch -D 分支名     # 强制删除

# 推送分支到远程
git push -u origin 分支名

# 删除远程分支
git push origin --delete 分支名
```

### 3.3 本地分支 vs 远程分支
```
本地                              远程 (GitHub)
─────                             ─────────────
master  ←───── pull/push ──────→  origin/master
feature ←───── pull/push ──────→  origin/feature (需要先 push -u 建立关联)
company       (只在本地，远程没有)
```

**关键理解：**
- 本地创建分支不会自动同步到远程
- 需要 `git push -u origin 分支名` 推送并建立关联
- `git pull` 只拉取当前分支对应的远程分支

### 3.4 合并分支
```bash
# 将 feature 合并到 master
git checkout master      # 先切换到目标分支
git merge feature        # 合并 feature 到当前分支

# 合并后删除已完成的功能分支
git branch -d feature
```

## 四、团队协作流程

### 4.1 标准工作流程
```
1. 从 master 创建功能分支
   git checkout master
   git pull
   git checkout -b feature-xxx

2. 在功能分支开发
   git add .
   git commit -m "完成xxx功能"

3. 开发完成，先更新 master
   git checkout master
   git pull

4. 合并最新 master 到功能分支（解决冲突）
   git checkout feature-xxx
   git merge master

5. 推送功能分支
   git push -u origin feature-xxx

6. 在 GitHub 上发起 Pull Request

7. 代码审核通过后合并到 master
```

### 4.2 个人项目简化流程
如果是个人项目，可以直接在 master 上开发：
```bash
git pull                 # 拉取最新
# ... 编写代码 ...
git add .
git commit -m "说明"
git push
```

## 五、常见场景处理

### 5.1 同步远程最新代码
```bash
# 场景：在家更新了代码，到公司要同步

# 方法1：直接拉取
git pull

# 方法2：如果本地有未提交的修改
git stash                # 暂存本地修改
git pull                 # 拉取远程
git stash pop            # 恢复本地修改
```

### 5.2 撤销操作
```bash
# 撤销工作区修改（未 add）
git checkout -- 文件名
git restore 文件名        # Git 2.23+

# 撤销暂存（已 add，未 commit）
git reset HEAD 文件名
git restore --staged 文件名

# 撤销提交（已 commit，未 push）
git reset --soft HEAD~1  # 保留修改
git reset --hard HEAD~1  # 丢弃修改

# 已 push 的提交，创建新提交来撤销
git revert 提交ID
```

### 5.3 解决冲突
当两个分支修改了同一文件的同一位置时会产生冲突：
```bash
# 合并时出现冲突
git merge feature
# Auto-merging file.txt
# CONFLICT (content): Merge conflict in file.txt

# 打开冲突文件，手动编辑
<<<<<<< HEAD
当前分支的内容
=======
要合并分支的内容
>>>>>>> feature

# 解决后
git add file.txt
git commit -m "解决冲突"
```

### 5.4 强制同步远程（丢弃本地所有修改）
```bash
git fetch --all
git reset --hard origin/master
```

## 六、.gitignore 文件

用于指定不需要版本控制的文件：
```gitignore
# 忽略所有 .log 文件
*.log

# 忽略 node_modules 目录
node_modules/

# 忽略 build 输出
dist/
build/

# 忽略 IDE 配置
.idea/
.vscode/

# 忽略环境配置文件
.env
.env.local
```

## 七、实用技巧

### 7.1 配置别名
```bash
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status

# 使用
git co master    # 等于 git checkout master
```

### 7.2 查看某个文件的修改历史
```bash
git log --follow -p 文件名
```

### 7.3 临时保存工作进度
```bash
git stash                # 保存当前工作
git stash list           # 查看保存列表
git stash pop            # 恢复最近一次
git stash drop           # 删除最近一次
```

### 7.4 修改最后一次提交
```bash
git commit --amend -m "新的提交信息"
```

## 八、常见问题

### Q1: git pull 提示 "Already up to date" 但文件数量不对？
检查是否在正确的分支：
```bash
git branch -a            # 查看所有分支
git checkout master      # 切换到正确分支
git pull
```

### Q2: 推送被拒绝？
远程有新提交，需要先拉取：
```bash
git pull --rebase
git push
```

### Q3: 如何查看某次提交改了什么？
```bash
git show 提交ID
```

### Q4: 如何对比两个分支的差异？
```bash
git diff master..feature
```

## 九、总结图示

```
┌─────────────────────────────────────────────────────────────────┐
│                        Git 工作流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   工作区          暂存区           本地仓库          远程仓库     │
│     │               │                │                 │        │
│     │── git add ───>│                │                 │        │
│     │               │── git commit ─>│                 │        │
│     │               │                │── git push ────>│        │
│     │<────────────── git pull ───────────────────────────       │
│     │                                                           │
│     │<── git checkout/restore ──│    (撤销工作区修改)            │
│                     │<── git reset ──│ (撤销暂存)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---
*最后更新：2024年12月*
