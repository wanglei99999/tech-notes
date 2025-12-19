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

### 3.4 分支命名规范

团队协作中约定俗成的命名方式，一眼就能看出分支用途：

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/` | 新功能开发 | `feature/user-login`、`feature/export-excel` |
| `bugfix/` | 修复 bug | `bugfix/pagination-error`、`bugfix/null-pointer` |
| `hotfix/` | 紧急线上修复 | `hotfix/security-patch`、`hotfix/data-fix` |
| `release/` | 发布版本 | `release/v1.0.0`、`release/2024-01` |
| `refactor/` | 代码重构 | `refactor/user-service`、`refactor/api-cleanup` |
| `test/` | 测试相关 | `test/unit-tests`、`test/e2e` |

**命名建议：**
- 使用小写字母和连字符 `-`
- 简短但能说明用途
- 可以加上 issue 编号：`feature/123-user-login`

### 3.5 合并分支
```bash
# 将 feature 合并到 master
git checkout master      # 先切换到目标分支
git merge feature        # 合并 feature 到当前分支

# 合并后删除已完成的功能分支
git branch -d feature
```

### 3.6 Rebase 变基（进阶）

`rebase` 可以让提交历史更整洁，变成一条直线。

**场景：你在 feature 分支开发，同时 master 也有新提交**

```
初始状态：
master:    A ── B ── E (别人的新提交)
                 \
feature:          C ── D (你的提交)
```

**用 merge 合并：**
```bash
git checkout feature
git merge master
```
```
结果（有分叉和合并提交）：
master:    A ── B ── E ────────┐
                 \              \
feature:          C ── D ── M (合并提交)
```

**用 rebase 变基：**
```bash
git checkout feature
git rebase master
```
```
结果（一条直线，更整洁）：
master:    A ── B ── E
                      \
feature:               C' ── D' (你的提交被"移动"到 E 后面)
```

**merge vs rebase 对比：**
| | merge | rebase |
|--|-------|--------|
| 历史记录 | 保留分叉，有合并提交 | 一条直线，更整洁 |
| 安全性 | 更安全，不改变历史 | 会重写提交历史 |
| 适用场景 | 团队协作、公共分支 | 个人分支整理 |

**使用建议：**
- 新手先用 `merge`，更安全
- `rebase` 只用于**未推送到远程的本地分支**
- **永远不要 rebase 公共分支**（如 master），会导致团队混乱

**常用 rebase 命令：**
```bash
# 将当前分支变基到 master
git rebase master

# 交互式 rebase（可以合并、修改、删除提交）
git rebase -i HEAD~3     # 操作最近 3 个提交

# 变基过程中出现冲突
git rebase --continue    # 解决冲突后继续
git rebase --abort       # 放弃变基，恢复原状
```

## 四、Commit 提交规范

### 4.1 Conventional Commits（约定式提交）

业界流行的提交信息规范，让提交历史清晰易读。

**格式：**
```
<类型>(<范围>): <简短描述>

[可选的详细说明]

[可选的关联 issue]
```

**常用类型：**
| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 添加用户登录功能` |
| `fix` | 修复 bug | `fix: 修复分页显示错误` |
| `docs` | 文档更新 | `docs: 更新 README` |
| `style` | 代码格式（不影响逻辑） | `style: 格式化代码` |
| `refactor` | 重构（不是新功能也不是修 bug） | `refactor: 重构用户服务` |
| `test` | 测试相关 | `test: 添加单元测试` |
| `chore` | 构建/工具/依赖等 | `chore: 升级依赖版本` |
| `perf` | 性能优化 | `perf: 优化查询速度` |

**好的例子：**
```bash
git commit -m "feat(user): 添加用户登录功能"
git commit -m "fix(table): 修复分页数据不刷新的问题"
git commit -m "docs: 添加 Git 学习笔记"
git commit -m "refactor(api): 重构文件上传接口"
git commit -m "chore: 更新 package.json 依赖"
```

**坏的例子：**
```bash
git commit -m "update"           # 太模糊
git commit -m "fix bug"          # 什么 bug？
git commit -m "修改"             # 修改了什么？
git commit -m "1"                # ???
git commit -m "."                # 完全没意义
```

### 4.2 简单原则

1. **用动词开头**：添加、修复、更新、删除、重构
2. **说清楚改了什么**：让别人（包括未来的自己）能看懂
3. **一个 commit 只做一件事**：方便回滚和追溯

**个人项目简化版：**
```bash
git commit -m "添加 Git 学习笔记"
git commit -m "修复登录页面样式问题"
git commit -m "重构文件上传组件"
```

## 五、团队协作流程

### 5.1 标准工作流程
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
*最后更新：2024年12月19日*
