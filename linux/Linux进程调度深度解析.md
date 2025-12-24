# Linux 进程调度深度解析

> 📖 参考书籍：《深入理解 Linux 进程与内存》
>
> 🤖 笔记方式：阅读书籍 + AI 辅助答疑整理
>
> 本文从零开始，系统性地介绍 Linux 内核的进程调度机制。适合想要深入理解操作系统调度原理的开发者阅读。

## 目录

1. [调度器演进历史](#1-调度器演进历史)
2. [现代调度器架构](#2-现代调度器架构)
3. [CFS 完全公平调度器](#3-cfs-完全公平调度器)
4. [调度时机](#4-调度时机)
5. [任务队列选择](#5-任务队列选择)
6. [负载均衡](#6-负载均衡)
7. [任务切换开销](#7-任务切换开销)
8. [调度相关命令](#8-调度相关命令)
9. [内核常见宏与技巧](#9-内核常见宏与技巧)

---

## 1. 调度器演进历史

### 1.1 为什么需要调度器？

CPU 是稀缺资源，而系统中有大量进程需要运行。调度器的职责就是**决定哪个进程在什么时候使用 CPU**。

想象一个银行只有一个柜台（CPU），但有很多客户（进程）在等待办理业务：
- 谁先办？
- 每个人办多久？
- 有 VIP 客户怎么处理？
- 有紧急业务怎么插队？

这就是调度器要解决的问题。

**一个好的调度器需要平衡：**

| 目标 | 说明 | 典型场景 |
|------|------|----------|
| **响应性** | 交互式任务要快速响应 | 鼠标点击、键盘输入 |
| **吞吐量** | 尽可能多地完成工作 | 批处理任务、编译 |
| **公平性** | 每个进程都能获得合理的 CPU 时间 | 多用户系统 |
| **实时性** | 关键任务必须在截止时间前完成 | 音视频播放、工业控制 |

这些目标往往是矛盾的，调度器需要在它们之间做权衡。

### 1.2 O(n) 调度器（Linux 2.4 及之前）

这是 Linux 最早期的调度器，设计简单但效率低下。

**工作原理：**
```
┌─────────────────────────────────────────────────────┐
│              全局运行队列（单链表）                   │
│  ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐            │
│  │ A │ → │ B │ → │ C │ → │ D │ → │ E │ → ...      │
│  └───┘   └───┘   └───┘   └───┘   └───┘            │
│                                                     │
│  每次调度：遍历所有进程，计算 goodness 值，选最大的   │
└─────────────────────────────────────────────────────┘
```

**goodness 值计算（简化）：**
```c
int goodness(struct task_struct *p) {
    int weight = p->counter;  // 剩余时间片
    
    if (p->policy == SCHED_NORMAL) {
        weight += 20 - p->nice;  // nice 值影响
        if (p->mm == current->mm)
            weight += 1;  // 同一地址空间加分（缓存友好）
    }
    return weight;
}
```

**问题：**
```
调度开销 = O(n)，n = 可运行进程数

100 个进程 → 每次调度遍历 100 次
10000 个进程 → 每次调度遍历 10000 次（灾难！）
```

- 进程越多，调度延迟越大
- SMP（多处理器）系统下，全局队列需要加锁，竞争严重
- 实时性差，不适合服务器场景

### 1.3 O(1) 调度器（Linux 2.6，2003年）

由 Ingo Molnar 开发，核心改进是使用**位图 + 优先级数组**实现 O(1) 查找。

**核心思想：用空间换时间**

```
┌─────────────────────────────────────────────────────────────┐
│                    O(1) 调度器数据结构                       │
│                                                             │
│  bitmap[5]  (140 bits)                                      │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐   │
│  │1│0│1│0│0│0│1│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│0│...│
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘   │
│   ↓   ↓       ↓                                             │
│  ┌───┬───┬───┬───┬───┬───┬───┐                              │
│  │ 0 │ 1 │ 2 │ 3 │ 4 │...│139│  queue[140]                  │
│  └─┬─┴───┴─┬─┴───┴───┴───┴─┬─┘                              │
│    │       │               │                                │
│    ▼       ▼               ▼                                │
│  ┌───┐   ┌───┐           ┌───┐                              │
│  │ A │   │ C │           │ F │                              │
│  └─┬─┘   └─┬─┘           └───┘                              │
│    │       │                                                │
│    ▼       ▼                                                │
│  ┌───┐   ┌───┐                                              │
│  │ B │   │ D │                                              │
│  └───┘   └───┘                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键数据结构：**
```c
struct runqueue {
    spinlock_t lock;              // 队列锁
    unsigned long nr_running;     // 可运行进程数
    struct prio_array *active;    // 活跃数组（有时间片的进程）
    struct prio_array *expired;   // 过期数组（时间片用完的进程）
    struct prio_array arrays[2];  // 两个数组，用于交换
};

struct prio_array {
    int nr_active;                    // 活跃进程数
    unsigned long bitmap[5];          // 140位的位图（BITMAP_SIZE）
    struct list_head queue[140];      // 每个优先级一个链表
};
```

**O(1) 如何实现：**

**步骤 1：位图快速查找最高优先级**
```c
// 找到第一个为 1 的位 → 最高优先级
idx = sched_find_first_bit(array->bitmap);  // 硬件指令，O(1)
```

x86 提供 `bsf`（Bit Scan Forward）指令，可以在一个时钟周期内找到第一个为 1 的位。

**步骤 2：从对应链表取进程**
```c
// 取该优先级链表的第一个进程
next = list_entry(array->queue[idx].next, task_t, run_list);  // O(1)
```

**步骤 3：双数组交换**
```c
// 当 active 数组为空时
if (array->nr_active == 0) {
    // 直接交换指针，不需要遍历！
    rq->active = rq->expired;
    rq->expired = array;
}
```

这个设计非常巧妙：
- 进程时间片用完 → 移到 expired 数组
- active 空了 → 交换指针，expired 变成新的 active
- 避免了 O(n) 的时间片重新计算

**Per-CPU 运行队列：**
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   CPU 0     │  │   CPU 1     │  │   CPU 2     │
│  runqueue   │  │  runqueue   │  │  runqueue   │
│  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │
│  │active │  │  │  │active │  │  │  │active │  │
│  │expired│  │  │  │expired│  │  │  │expired│  │
│  └───────┘  │  │  └───────┘  │  │  └───────┘  │
└─────────────┘  └─────────────┘  └─────────────┘
      ↑               ↑               ↑
   独立的锁        独立的锁        独立的锁
   无竞争          无竞争          无竞争
```

每个 CPU 有独立的运行队列和锁，大大减少了多核系统的锁竞争。

### 1.4 CFS 调度器（Linux 2.6.23+，2007年至今）

O(1) 调度器虽然快，但有公平性问题。CFS（Completely Fair Scheduler）由 Ingo Molnar 重新设计，追求"完全公平"。

**为什么要换掉 O(1)？**

O(1) 调度器的问题：
1. **交互性判断不准确**：用启发式算法判断进程是否是交互式的，经常出错
2. **复杂的优先级计算**：代码复杂，难以维护
3. **不够公平**：某些场景下进程得不到应有的 CPU 时间

**CFS 的核心思想：**
```
理想情况：N 个进程，每个同时获得 1/N 的 CPU
现实情况：CPU 不能真正"同时"运行多个进程
CFS 方案：用 vruntime（虚拟运行时间）来模拟公平
```

**CFS vs O(1) 对比：**

| 特性 | O(1) 调度器 | CFS 调度器 |
|------|------------|-----------|
| 数据结构 | 位图 + 优先级数组 | 红黑树 |
| 查找复杂度 | O(1) | O(1)（缓存最左节点） |
| 插入/删除 | O(1) | O(log n) |
| 公平性 | 一般 | 优秀 |
| 代码复杂度 | 高 | 相对简单 |
| 交互性 | 启发式判断 | 自然支持 |

---

## 2. 现代调度器架构

### 2.1 Per-CPU 运行队列（rq）

内核为每个逻辑 CPU 维护一个独立的运行队列：

```c
// 简化的 rq 结构
struct rq {
    raw_spinlock_t lock;        // 队列锁
    unsigned int nr_running;    // 总可运行进程数
    
    struct cfs_rq cfs;          // CFS 队列（普通进程）
    struct rt_rq rt;            // 实时队列
    struct dl_rq dl;            // Deadline 队列
    
    struct task_struct *curr;   // 当前运行的进程
    struct task_struct *idle;   // 该 CPU 的 idle 进程
    
    u64 clock;                  // 队列时钟
    int cpu;                    // 所属 CPU 编号
};
```

**架构图：**
```
┌─────────────────────────────────────────────────────────────────┐
│                         Linux 调度器架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CPU 0                    CPU 1                    CPU 2       │
│     │                        │                        │         │
│     ▼                        ▼                        ▼         │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │     rq       │      │     rq       │      │     rq       │  │
│  ├──────────────┤      ├──────────────┤      ├──────────────┤  │
│  │   dl_rq      │      │   dl_rq      │      │   dl_rq      │  │
│  │  (红黑树)    │      │  (红黑树)    │      │  (红黑树)    │  │
│  ├──────────────┤      ├──────────────┤      ├──────────────┤  │
│  │   rt_rq      │      │   rt_rq      │      │   rt_rq      │  │
│  │ (位图+链表)  │      │ (位图+链表)  │      │ (位图+链表)  │  │
│  ├──────────────┤      ├──────────────┤      ├──────────────┤  │
│  │   cfs_rq     │      │   cfs_rq     │      │   cfs_rq     │  │
│  │  (红黑树)    │      │  (红黑树)    │      │  (红黑树)    │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                                                                 │
│                    ↑↓ 负载均衡 ↑↓                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 调度类（Scheduling Class）

Linux 使用**调度类**来支持不同类型的调度策略。每个调度类实现一组标准接口：

```c
struct sched_class {
    const struct sched_class *next;  // 下一个调度类（优先级链）
    
    // 核心调度操作
    void (*enqueue_task)(struct rq *rq, struct task_struct *p, int flags);
    void (*dequeue_task)(struct rq *rq, struct task_struct *p, int flags);
    struct task_struct *(*pick_next_task)(struct rq *rq);
    void (*put_prev_task)(struct rq *rq, struct task_struct *p);
    
    // 时钟 tick 处理
    void (*task_tick)(struct rq *rq, struct task_struct *p, int queued);
    
    // 负载均衡相关
    int (*select_task_rq)(struct task_struct *p, int cpu, int flags);
    // ... 更多接口
};
```

**调度类优先级链：**
```
stop_sched_class      (最高优先级，用于 CPU 热插拔等)
       ↓
dl_sched_class        (Deadline 调度)
       ↓
rt_sched_class        (实时调度)
       ↓
fair_sched_class      (CFS 公平调度)
       ↓
idle_sched_class      (最低优先级，idle 进程)
```

**调度时的查找顺序：**
```c
// 简化的 pick_next_task
struct task_struct *pick_next_task(struct rq *rq) {
    struct task_struct *p;
    const struct sched_class *class;
    
    // 按优先级遍历调度类
    for_each_class(class) {
        p = class->pick_next_task(rq);
        if (p)
            return p;
    }
    
    // 永远不会到这里，因为 idle 进程总是存在
    BUG();
}
```

### 2.3 各调度类详解

#### 2.3.1 dl_sched_class（Deadline 调度）

**适用场景**：硬实时任务，有明确的截止时间要求

**调度策略**：SCHED_DEADLINE

**核心参数**：
```c
struct sched_attr {
    __u64 sched_runtime;   // 每周期需要的 CPU 时间（纳秒）
    __u64 sched_deadline;  // 相对截止时间（纳秒）
    __u64 sched_period;    // 任务周期（纳秒）
};
```

**示例：视频解码任务**
```
period = 33ms      (30fps，每帧 33ms)
runtime = 10ms     (解码一帧需要 10ms CPU 时间)
deadline = 30ms    (必须在 30ms 内完成)

时间线：
├─────────────────────────────────────────────────────┤
│                    period (33ms)                     │
├──────────┬───────────────────────┬──────────────────┤
│ runtime  │                       │                  │
│  (10ms)  │      可以休息         │   deadline 前    │
│ ████████ │                       │   必须完成       │
└──────────┴───────────────────────┴──────────────────┘
     ↑                                    ↑
   开始执行                            截止时间(30ms)
```

**调度算法**：EDF（Earliest Deadline First）
- 按截止时间排序（红黑树）
- 截止时间最近的优先运行
- 有准入控制，防止系统过载

#### 2.3.2 rt_sched_class（实时调度）

**适用场景**：软实时任务，需要快速响应

**调度策略**：
- `SCHED_FIFO`：先进先出，同优先级不抢占
- `SCHED_RR`：时间片轮转，同优先级轮流执行

**优先级范围**：0-99（数字越大优先级越高）

**数据结构**（保留了 O(1) 的设计）：
```c
struct rt_rq {
    struct rt_prio_array active;
    unsigned int rt_nr_running;     // 实时进程数
    unsigned int rr_nr_running;     // RR 进程数
    // ...
};

struct rt_prio_array {
    DECLARE_BITMAP(bitmap, MAX_RT_PRIO+1);  // 100 位的位图
    struct list_head queue[MAX_RT_PRIO];     // 100 个优先级链表
};
```

**FIFO vs RR：**
```
SCHED_FIFO：
优先级 50: [A] → [B] → [C]
           ↑
        A 一直运行，直到主动让出或被更高优先级抢占

SCHED_RR：
优先级 50: [A] → [B] → [C]
           ↑
        A 运行一个时间片 → B 运行一个时间片 → C → A → ...
```

#### 2.3.3 fair_sched_class（CFS 公平调度）

**适用场景**：普通进程（绝大多数进程）

**调度策略**：
- `SCHED_NORMAL`：普通交互式进程
- `SCHED_BATCH`：批处理进程，不需要交互
- `SCHED_IDLE`：最低优先级的后台任务

这是最复杂也最重要的调度类，后面会详细讲解。

#### 2.3.4 idle_sched_class（空闲调度）

**适用场景**：CPU 空闲时运行

每个 CPU 都有一个 idle 进程（swapper，PID 0），当没有其他进程可运行时，运行 idle 进程。

**idle 进程的作用：**
```c
void cpu_idle_loop(void) {
    while (1) {
        // 检查是否需要调度
        while (!need_resched()) {
            // 进入低功耗状态
            cpuidle_idle_call();
            // 可能执行：
            // - HLT 指令（停止 CPU，等待中断）
            // - MWAIT 指令（更深度的睡眠）
            // - 各种 C-state 节能状态
        }
        // 有进程可运行了，调度出去
        schedule_idle();
    }
}
```

**为什么需要 idle 进程？**
1. **CPU 必须有进程运行**：调度器设计假设总有进程在运行
2. **省电**：idle 进程让 CPU 进入低功耗状态
3. **统计**：`top` 命令的 `%id` 就是 idle 进程的运行时间

### 2.4 调度类优先级总结

```
┌─────────────────────────────────────────────────────────────┐
│                      调度优先级                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  最高  ┌─────────────────────────────────────────────────┐  │
│    ↑   │  SCHED_DEADLINE                                 │  │
│    │   │  硬实时，有明确截止时间                          │  │
│    │   │  优先级：按 deadline 排序                        │  │
│    │   ├─────────────────────────────────────────────────┤  │
│    │   │  SCHED_FIFO / SCHED_RR                          │  │
│    │   │  软实时，固定优先级 0-99                         │  │
│    │   │  FIFO 不抢占，RR 时间片轮转                      │  │
│    │   ├─────────────────────────────────────────────────┤  │
│    │   │  SCHED_NORMAL / SCHED_BATCH                     │  │
│    │   │  普通进程，CFS 调度                              │  │
│    │   │  nice 值 -20 到 +19                             │  │
│    │   ├─────────────────────────────────────────────────┤  │
│    ↓   │  SCHED_IDLE                                     │  │
│  最低  │  最低优先级后台任务                              │  │
│        └─────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. CFS 完全公平调度器

CFS 是 Linux 默认的调度器，处理绝大多数进程。理解 CFS 是理解 Linux 调度的关键。

### 3.1 核心思想：完全公平

**理想模型：**
```
如果有 N 个进程，每个进程应该同时获得 1/N 的 CPU

例：4 个进程，每个应该获得 25% 的 CPU

理想情况（不可能实现）：
时间 →
CPU: [A+B+C+D][A+B+C+D][A+B+C+D]...
      同时运行  同时运行  同时运行
```

**现实模型：**
```
CPU 一次只能运行一个进程，只能轮流执行

现实情况：
时间 →
CPU: [A][B][C][D][A][B][C][D][A][B][C][D]...
      轮流执行，模拟"同时"
```

**CFS 的方案：用 vruntime 追踪公平性**

### 3.2 vruntime：虚拟运行时间

vruntime 是 CFS 的核心概念，表示进程"应该"运行了多长时间。

**基本公式：**
```
vruntime = 实际运行时间 × (NICE_0_LOAD / 进程权重)
         = 实际运行时间 × (1024 / weight)
```

**核心规则：vruntime 最小的进程优先运行**

**为什么这样设计？**
```
假设 A、B 两个进程，权重相同：

初始状态：
A.vruntime = 0
B.vruntime = 0

A 运行 10ms 后：
A.vruntime = 10
B.vruntime = 0  ← 更小，应该运行 B

B 运行 10ms 后：
A.vruntime = 10
B.vruntime = 10  ← 相等，公平！

继续轮流...
```

### 3.3 权重与 nice 值

不同优先级的进程，vruntime 增长速度不同。

**nice 值范围**：-20（最高优先级）到 +19（最低优先级），默认 0

**nice 值 → 权重映射表（部分）：**
```c
static const int prio_to_weight[40] = {
 /* -20 */     88761,     71755,     56483,     46273,     36291,
 /* -15 */     29154,     23254,     18705,     14949,     11916,
 /* -10 */      9548,      7620,      6100,      4904,      3906,
 /*  -5 */      3121,      2501,      1991,      1586,      1277,
 /*   0 */      1024,       820,       655,       526,       423,
 /*   5 */       335,       272,       215,       172,       137,
 /*  10 */       110,        87,        70,        56,        45,
 /*  15 */        36,        29,        23,        18,        15,
};
```

**设计原则**：每差 1 个 nice 值，CPU 时间相差约 10%

**权重如何影响 vruntime：**
```
进程 A：nice = 0，权重 = 1024
进程 B：nice = 5，权重 = 335

两者都运行 10ms：

A 的 vruntime 增量 = 10 × (1024 / 1024) = 10
B 的 vruntime 增量 = 10 × (1024 / 335) ≈ 30.6

结果：
- A 运行 10ms，vruntime 增加 10
- B 运行 10ms，vruntime 增加 30.6
- A 的 vruntime 增长更慢，会获得更多 CPU 时间
```

**实际效果：**
```
nice 0 的进程 vs nice 5 的进程

nice 0 获得的 CPU 时间 ≈ nice 5 的 3 倍
(因为 1024 / 335 ≈ 3.06)
```

### 3.4 红黑树：高效的数据结构

CFS 使用红黑树来组织所有可运行的进程，按 vruntime 排序。

**为什么选择红黑树？**
```
需求：
1. 快速找到 vruntime 最小的进程
2. 快速插入/删除进程
3. 保持有序

红黑树特性：
- 查找最小值：O(1)（缓存最左节点）
- 插入/删除：O(log n)
- 自平衡，不会退化成链表
```

**红黑树结构示意：**
```
                    ┌─────────────────┐
                    │   cfs_rq        │
                    │                 │
                    │  rb_leftmost ───┼──────┐
                    │                 │      │
                    │  tasks_timeline │      │
                    └────────┬────────┘      │
                             │               │
                             ▼               │
                        ┌────────┐           │
                        │  105   │           │
                        │ (进程E)│           │
                        └───┬────┘           │
                       ╱         ╲           │
                 ┌────────┐   ┌────────┐     │
                 │  102   │   │  110   │     │
                 │ (进程C)│   │ (进程F)│     │
                 └───┬────┘   └────────┘     │
                ╱         ╲                  │
          ┌────────┐   ┌────────┐            │
          │  100   │   │  103   │            │
          │ (进程A)│   │ (进程D)│            │
          └────────┘   └────────┘            │
               ↑                             │
               └─────────────────────────────┘
            rb_leftmost 指向最左节点（vruntime 最小）
```

### 3.5 cfs_rq 数据结构详解

```c
struct cfs_rq {
    // 负载相关
    struct load_weight load;        // 队列总权重
    unsigned int nr_running;        // 可运行进程数
    
    // 红黑树
    struct rb_root_cached tasks_timeline;
    // rb_root_cached 包含：
    //   - rb_root：红黑树根
    //   - rb_leftmost：最左节点指针（缓存）
    
    // vruntime 相关
    u64 min_vruntime;               // 队列最小 vruntime（单调递增）
    
    // 当前运行的调度实体
    struct sched_entity *curr;
    struct sched_entity *next;      // 下一个要运行的（hint）
    struct sched_entity *skip;      // 跳过的（yield）
    
    // 组调度相关
    struct task_group *tg;          // 所属任务组
    // ...
};
```

**sched_entity：调度实体**
```c
struct sched_entity {
    struct load_weight load;        // 权重
    struct rb_node run_node;        // 红黑树节点
    unsigned int on_rq;             // 是否在运行队列上
    
    u64 exec_start;                 // 本次运行开始时间
    u64 sum_exec_runtime;           // 总运行时间（实际）
    u64 vruntime;                   // 虚拟运行时间
    u64 prev_sum_exec_runtime;      // 上次调度时的总运行时间
    
    // 组调度相关
    struct sched_entity *parent;    // 父调度实体
    struct cfs_rq *cfs_rq;          // 所属的 cfs_rq
    struct cfs_rq *my_q;            // 如果是组，指向自己的 cfs_rq
    // ...
};
```

### 3.6 核心调度流程

#### 3.6.1 选择下一个进程（pick_next_entity）

```c
struct sched_entity *pick_next_entity(struct cfs_rq *cfs_rq) {
    // 1. 获取红黑树最左节点（vruntime 最小）
    struct rb_node *left = rb_first_cached(&cfs_rq->tasks_timeline);
    
    // 2. 如果没有进程，返回 NULL
    if (!left)
        return NULL;
    
    // 3. 通过 rb_entry 宏获取 sched_entity
    struct sched_entity *se = rb_entry(left, struct sched_entity, run_node);
    
    // 4. 检查是否有 next/skip hint
    if (cfs_rq->next && wakeup_preempt_entity(cfs_rq->next, left) < 1)
        se = cfs_rq->next;
    
    return se;
}
```

**时间复杂度**：O(1)，因为最左节点被缓存了

#### 3.6.2 更新 vruntime（update_curr）

每次时钟 tick 或调度时调用：

```c
static void update_curr(struct cfs_rq *cfs_rq) {
    struct sched_entity *curr = cfs_rq->curr;
    u64 now = rq_clock_task(rq_of(cfs_rq));
    u64 delta_exec;
    
    if (unlikely(!curr))
        return;
    
    // 1. 计算本次运行的实际时间
    delta_exec = now - curr->exec_start;
    curr->exec_start = now;
    
    // 2. 累加实际运行时间
    curr->sum_exec_runtime += delta_exec;
    
    // 3. 计算并累加 vruntime
    curr->vruntime += calc_delta_fair(delta_exec, curr);
    
    // 4. 更新 cfs_rq 的 min_vruntime
    update_min_vruntime(cfs_rq);
}

// 计算 vruntime 增量
static u64 calc_delta_fair(u64 delta, struct sched_entity *se) {
    // vruntime 增量 = 实际时间 × (NICE_0_LOAD / 权重)
    if (unlikely(se->load.weight != NICE_0_LOAD))
        delta = __calc_delta(delta, NICE_0_LOAD, &se->load);
    return delta;
}
```

#### 3.6.3 进程入队（enqueue_entity）

```c
static void enqueue_entity(struct cfs_rq *cfs_rq, struct sched_entity *se, int flags) {
    // 1. 如果是新进程或长时间睡眠后唤醒，调整 vruntime
    if (!(flags & ENQUEUE_WAKEUP) || (flags & ENQUEUE_MIGRATED))
        se->vruntime += cfs_rq->min_vruntime;
    
    // 2. 更新负载统计
    update_load_add(&cfs_rq->load, se->load.weight);
    
    // 3. 插入红黑树
    __enqueue_entity(cfs_rq, se);
    
    // 4. 更新计数
    cfs_rq->nr_running++;
}

// 插入红黑树
static void __enqueue_entity(struct cfs_rq *cfs_rq, struct sched_entity *se) {
    struct rb_node **link = &cfs_rq->tasks_timeline.rb_root.rb_node;
    struct rb_node *parent = NULL;
    struct sched_entity *entry;
    bool leftmost = true;
    
    // 二叉搜索树插入
    while (*link) {
        parent = *link;
        entry = rb_entry(parent, struct sched_entity, run_node);
        
        if (se->vruntime < entry->vruntime) {
            link = &parent->rb_left;
        } else {
            link = &parent->rb_right;
            leftmost = false;  // 不是最左节点
        }
    }
    
    // 插入并平衡
    rb_link_node(&se->run_node, parent, link);
    rb_insert_color_cached(&se->run_node, &cfs_rq->tasks_timeline, leftmost);
}
```

**时间复杂度**：O(log n)

### 3.7 min_vruntime 的作用

**问题**：新进程的 vruntime 应该设为多少？

```
场景 1：设为 0
新进程 vruntime = 0
老进程 vruntime = 1000000
→ 新进程会一直抢占，老进程饿死

场景 2：设为当前最大值
新进程 vruntime = 1000000
老进程 vruntime = 500000
→ 新进程要等很久才能运行
```

**解决方案**：使用 min_vruntime 作为基准

```c
// min_vruntime 的更新
static void update_min_vruntime(struct cfs_rq *cfs_rq) {
    u64 vruntime = cfs_rq->min_vruntime;
    
    // 考虑当前运行进程的 vruntime
    if (cfs_rq->curr)
        vruntime = cfs_rq->curr->vruntime;
    
    // 考虑红黑树最左节点的 vruntime
    if (cfs_rq->rb_leftmost) {
        struct sched_entity *se = rb_entry(cfs_rq->rb_leftmost, ...);
        vruntime = min(vruntime, se->vruntime);
    }
    
    // min_vruntime 只能单调递增
    cfs_rq->min_vruntime = max(cfs_rq->min_vruntime, vruntime);
}
```

**新进程的 vruntime 初始化**：
```c
// 新进程入队时
se->vruntime = max(se->vruntime, cfs_rq->min_vruntime);
```

这样：
- 新进程不会太占便宜（vruntime 不会太小）
- 也不会等太久（从当前最小值开始）

### 3.8 时间片计算

CFS 没有固定时间片，而是动态计算。

**调度周期（sched_period）**：
```c
// 所有进程轮一遍的目标时间
sched_period = max(sysctl_sched_latency, nr_running × sysctl_sched_min_granularity)

// 默认值
sysctl_sched_latency = 6ms           // 目标延迟
sysctl_sched_min_granularity = 0.75ms // 最小时间片
```

**每个进程的时间片**：
```c
time_slice = sched_period × (进程权重 / 队列总权重)
```

**示例**：
```
4 个相同优先级进程（权重都是 1024）：
sched_period = max(6ms, 4 × 0.75ms) = 6ms
每个进程时间片 = 6ms × (1024 / 4096) = 1.5ms

100 个相同优先级进程：
sched_period = max(6ms, 100 × 0.75ms) = 75ms
每个进程时间片 = 75ms / 100 = 0.75ms（最小值）
```

### 3.9 抢占检查

CFS 在以下情况检查是否需要抢占当前进程：

```c
static void check_preempt_wakeup(struct rq *rq, struct task_struct *p, int flags) {
    struct sched_entity *se = &current->se;
    struct sched_entity *pse = &p->se;
    
    // 1. 如果被唤醒的进程 vruntime 更小，考虑抢占
    if (pse->vruntime < se->vruntime) {
        // 2. 但要超过一定阈值才抢占（避免频繁切换）
        s64 delta = se->vruntime - pse->vruntime;
        if (delta > sysctl_sched_wakeup_granularity) {
            resched_curr(rq);  // 标记需要重新调度
        }
    }
}
```

**sysctl_sched_wakeup_granularity**：默认 1ms，防止过于频繁的抢占。

---

## 4. 调度时机

### 4.1 两种调度方式

**主动调度（Voluntary）**：进程主动放弃 CPU
```c
// 直接让出 CPU
schedule();

// 条件让出（如果需要）
cond_resched();

// 等待资源时让出
mutex_lock(&lock);      // 拿不到锁，睡眠
wait_event(wq, cond);   // 等待条件满足
msleep(100);            // 主动睡眠
io_schedule();          // 等待 I/O
```

**被动调度（Involuntary/Preemption）**：进程被迫让出 CPU
```c
// 内核设置 TIF_NEED_RESCHED 标志
set_tsk_need_resched(current);

// 在特定时机检查这个标志，触发调度
if (need_resched())
    schedule();
```

### 4.2 调度时机详解

#### 4.2.1 时钟中断（scheduler_tick）

每个时钟 tick（通常 1ms 或 4ms）触发：

```c
void scheduler_tick(void) {
    struct rq *rq = this_rq();
    struct task_struct *curr = rq->curr;
    
    // 1. 更新时钟
    update_rq_clock(rq);
    
    // 2. 调用当前进程所属调度类的 tick 处理
    curr->sched_class->task_tick(rq, curr, 0);
    
    // 3. 触发负载均衡检查
    trigger_load_balance(rq);
}

// CFS 的 tick 处理
static void task_tick_fair(struct rq *rq, struct task_struct *curr, int queued) {
    struct sched_entity *se = &curr->se;
    
    // 更新 vruntime
    update_curr(cfs_rq_of(se));
    
    // 检查是否需要抢占
    if (cfs_rq->nr_running > 1)
        check_preempt_tick(cfs_rq, se);
}

// 检查时间片是否用完
static void check_preempt_tick(struct cfs_rq *cfs_rq, struct sched_entity *curr) {
    u64 ideal_runtime = sched_slice(cfs_rq, curr);  // 理想时间片
    u64 delta_exec = curr->sum_exec_runtime - curr->prev_sum_exec_runtime;
    
    // 如果运行时间超过理想时间片，标记需要调度
    if (delta_exec > ideal_runtime) {
        resched_curr(rq_of(cfs_rq));
    }
}
```

**注意**：时钟中断只是设置标志，不会立即调度！

#### 4.2.2 中断/系统调用返回

```
用户态 → 系统调用/中断 → 内核态 → 返回用户态
                                    ↓
                            检查 TIF_NEED_RESCHED
                                    ↓
                            如果设置了 → schedule()
```

```c
// 返回用户态前的检查（简化）
ret_to_user:
    // 检查是否需要调度
    if (test_thread_flag(TIF_NEED_RESCHED)) {
        schedule();
        goto ret_to_user;  // 调度后重新检查
    }
    // 返回用户态
    return_to_userspace();
```

#### 4.2.3 进程唤醒（try_to_wake_up）

```c
int try_to_wake_up(struct task_struct *p, unsigned int state, int wake_flags) {
    // 1. 选择目标 CPU
    cpu = select_task_rq(p, p->wake_cpu, wake_flags);
    
    // 2. 将进程加入运行队列
    activate_task(rq, p, flags);
    
    // 3. 检查是否应该抢占当前进程
    check_preempt_curr(rq, p, wake_flags);
    
    return success;
}

// 检查是否抢占
void check_preempt_curr(struct rq *rq, struct task_struct *p, int flags) {
    // 调用调度类的抢占检查
    p->sched_class->check_preempt_curr(rq, p, flags);
}
```

#### 4.2.4 进程创建（wake_up_new_task）

```c
void wake_up_new_task(struct task_struct *p) {
    // 1. 选择 CPU
    __set_task_cpu(p, select_task_rq(p, task_cpu(p), WF_FORK));
    
    // 2. 加入运行队列
    activate_task(rq, p, ENQUEUE_NOCLOCK);
    
    // 3. 检查是否抢占父进程
    check_preempt_curr(rq, p, WF_FORK);
}
```

#### 4.2.5 进程退出

```c
void do_exit(long code) {
    // ... 清理工作 ...
    
    // 设置状态为 TASK_DEAD
    set_special_state(TASK_DEAD);
    
    // 最后一次调度，永不返回
    do_task_dead();
}

void __noreturn do_task_dead(void) {
    // 通知调度器
    __schedule(false);
    
    // 永远不会执行到这里
    BUG();
}
```

### 4.3 TIF_NEED_RESCHED 标志

这是调度的核心机制，存储在 `thread_info` 结构中：

```c
struct thread_info {
    unsigned long flags;  // 包含 TIF_NEED_RESCHED 等标志
    // ...
};

// 标志位定义
#define TIF_NEED_RESCHED    3  // 需要重新调度

// 操作宏
#define set_tsk_need_resched(tsk)   set_tsk_thread_flag(tsk, TIF_NEED_RESCHED)
#define clear_tsk_need_resched(tsk) clear_tsk_thread_flag(tsk, TIF_NEED_RESCHED)
#define test_tsk_need_resched(tsk)  test_tsk_thread_flag(tsk, TIF_NEED_RESCHED)
#define need_resched()              test_thread_flag(TIF_NEED_RESCHED)
```

**为什么用标志而不是直接调度？**

```
中断上下文的限制：
├── 不能睡眠
├── 不能调用可能睡眠的函数
├── 必须快速返回
└── 调度可能涉及睡眠（等待锁等）

解决方案：
├── 中断中只设置标志
├── 返回安全上下文后再检查标志
└── 如果标志被设置，执行调度
```

### 4.4 内核抢占

#### 4.4.1 非抢占内核 vs 抢占内核

**非抢占内核（旧）**：
```
只有返回用户态时才检查 TIF_NEED_RESCHED
内核态代码不会被抢占
```

**抢占内核（现代 Linux）**：
```
内核态代码也可以被抢占
但需要满足条件：preempt_count == 0 且 TIF_NEED_RESCHED 被设置
```

#### 4.4.2 preempt_count

```c
// preempt_count 结构（32位）
┌────────────────────────────────────────────────────────┐
│ bit 31-24 │ bit 23-16 │ bit 15-8  │ bit 7-0           │
├───────────┼───────────┼───────────┼───────────────────┤
│  保留     │ 硬中断    │ 软中断    │ 抢占禁用计数      │
│           │ 嵌套计数  │ 嵌套计数  │                   │
└────────────────────────────────────────────────────────┘

只有 preempt_count == 0 时才能抢占
```

**抢占禁用/启用：**
```c
// 禁用抢占
preempt_disable();  // preempt_count++

// 启用抢占
preempt_enable();   // preempt_count--，如果变为 0 且需要调度，则调度

// 常见的隐式禁用
spin_lock();        // 禁用抢占
spin_unlock();      // 启用抢占，检查是否需要调度
```

#### 4.4.3 抢占点

内核代码中的抢占检查点：

```c
// spin_unlock 后检查
spin_unlock(&lock);
// 内部会调用 preempt_enable()，可能触发调度

// preempt_enable 后检查
preempt_enable();
// 如果 preempt_count 变为 0 且 TIF_NEED_RESCHED 被设置，调度

// 显式检查点
if (need_resched())
    cond_resched();
```

### 4.5 调度流程总结

```
┌─────────────────────────────────────────────────────────────────┐
│                        调度触发流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ 时钟中断    │   │ 进程唤醒    │   │ 进程创建    │           │
│  │ scheduler_ │   │ try_to_     │   │ wake_up_    │           │
│  │ tick()     │   │ wake_up()   │   │ new_task()  │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         └────────────────┬┴─────────────────┘                   │
│                          │                                      │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │ set_tsk_need_resched()│                          │
│              │ 设置 TIF_NEED_RESCHED │                          │
│              └───────────┬───────────┘                          │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ 中断返回    │  │ 系统调用    │  │ 内核抢占点  │             │
│  │ 用户态      │  │ 返回        │  │ spin_unlock │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │    need_resched()?    │                          │
│              └───────────┬───────────┘                          │
│                          │ Yes                                  │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │      schedule()       │                          │
│              │    实际执行调度       │                          │
│              └───────────────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


---

## 5. 任务队列选择

### 5.1 核心问题

当一个进程需要运行（新创建或被唤醒）时，应该放到哪个 CPU 的运行队列上？

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   进程 P 被唤醒，应该放到哪个 CPU？                              │
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │  CPU 0  │    │  CPU 1  │    │  CPU 2  │    │  CPU 3  │     │
│   │ 负载:8  │    │ 负载:2  │    │ 负载:5  │    │ 负载:3  │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│        ?              ?              ?              ?           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 select_task_rq 函数

```c
int select_task_rq(struct task_struct *p, int cpu, int flags) {
    // 调用进程所属调度类的选择函数
    return p->sched_class->select_task_rq(p, cpu, flags);
}

// CFS 的实现
static int select_task_rq_fair(struct task_struct *p, int prev_cpu, int flags) {
    struct sched_domain *sd;
    int new_cpu = prev_cpu;
    
    // 1. 检查 CPU 亲和性
    if (!cpumask_test_cpu(prev_cpu, p->cpus_ptr))
        new_cpu = cpumask_any(p->cpus_ptr);
    
    // 2. 唤醒场景的快速路径
    if (flags & WF_TTWU) {
        // 尝试选择唤醒者的 CPU 或之前的 CPU
        new_cpu = select_idle_sibling(p, prev_cpu, new_cpu);
    }
    
    // 3. 慢速路径：遍历调度域找最优 CPU
    if (flags & WF_FORK) {
        new_cpu = find_idlest_cpu(sd, p, new_cpu);
    }
    
    return new_cpu;
}
```

### 5.3 关键考虑因素

#### 5.3.1 CPU 亲和性（Affinity）

进程可以绑定到特定的 CPU 集合：

```c
struct task_struct {
    cpumask_t cpus_mask;  // 允许运行的 CPU 集合
};

// 用户空间设置
sched_setaffinity(pid, sizeof(mask), &mask);

// 命令行
taskset -c 0,1 ./program  // 只在 CPU 0,1 上运行
```

**选择时必须满足亲和性约束**：
```c
if (!cpumask_test_cpu(target_cpu, p->cpus_ptr))
    return -EINVAL;  // 不允许在这个 CPU 上运行
```

#### 5.3.2 缓存亲和性（Cache Affinity）

```
进程 A 在 CPU 0 运行了一段时间
→ A 的数据在 CPU 0 的 L1/L2/L3 缓存中（热数据）

A 睡眠后被唤醒：
├── 选择 CPU 0：缓存命中率高，性能好
└── 选择 CPU 1：缓存冷启动，性能差

结论：尽量选择之前运行过的 CPU
```

#### 5.3.3 负载均衡

```
CPU 0: 负载 80%
CPU 1: 负载 20%

新进程应该放 CPU 1 → 均衡负载
```

#### 5.3.4 NUMA 拓扑

```
┌─────────────────────────────────────────────────────────────────┐
│                         NUMA 架构                               │
│                                                                 │
│   ┌─────────────────────┐         ┌─────────────────────┐      │
│   │    NUMA Node 0      │         │    NUMA Node 1      │      │
│   │                     │         │                     │      │
│   │  ┌───────┬───────┐  │         │  ┌───────┬───────┐  │      │
│   │  │ CPU 0 │ CPU 1 │  │         │  │ CPU 2 │ CPU 3 │  │      │
│   │  └───────┴───────┘  │         │  └───────┴───────┘  │      │
│   │         ↓↑          │  互联   │         ↓↑          │      │
│   │  ┌─────────────┐    │←───────→│  ┌─────────────┐    │      │
│   │  │  本地内存   │    │  (慢)   │  │  本地内存   │    │      │
│   │  │  访问:100ns │    │         │  │  访问:100ns │    │      │
│   │  └─────────────┘    │         │  └─────────────┘    │      │
│   └─────────────────────┘         └─────────────────────┘      │
│                                                                 │
│   跨节点访问内存：~300ns（3倍延迟）                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**选择原则**：尽量选择进程内存所在 NUMA 节点的 CPU

### 5.4 select_idle_sibling：找空闲兄弟

唤醒时的快速路径，尝试找一个空闲的"近邻" CPU：

```c
static int select_idle_sibling(struct task_struct *p, int prev, int target) {
    // 1. 目标 CPU 空闲？直接用
    if (available_idle_cpu(target))
        return target;
    
    // 2. 之前的 CPU 空闲？用它（缓存热）
    if (prev != target && available_idle_cpu(prev))
        return prev;
    
    // 3. 在同一 LLC（共享缓存）内找空闲 CPU
    struct sched_domain *sd = rcu_dereference(per_cpu(sd_llc, target));
    if (sd) {
        int cpu = select_idle_cpu(p, sd, target);
        if (cpu >= 0)
            return cpu;
    }
    
    // 4. 没找到空闲的，返回目标 CPU
    return target;
}
```

### 5.5 选择优先级总结

```
┌─────────────────────────────────────────────────────────────────┐
│                     CPU 选择优先级                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 必须满足 CPU 亲和性（cpus_mask）                            │
│                    ↓                                            │
│  2. 优先选之前运行的 CPU（缓存热）                              │
│                    ↓                                            │
│  3. 如果之前的 CPU 忙，选同一缓存域的空闲 CPU                   │
│                    ↓                                            │
│  4. 如果都忙，选同一 NUMA 节点的最空闲 CPU                      │
│                    ↓                                            │
│  5. 最后才考虑跨 NUMA 节点                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 负载均衡

### 6.1 为什么需要负载均衡？

每个 CPU 有独立的运行队列，可能出现不均衡：

```
┌─────────────────────────────────────────────────────────────────┐
│                        负载不均衡                                │
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │  CPU 0  │    │  CPU 1  │    │  CPU 2  │    │  CPU 3  │     │
│   │ ┌─────┐ │    │ ┌─────┐ │    │ ┌─────┐ │    │ ┌─────┐ │     │
│   │ │ A   │ │    │ │     │ │    │ │ G   │ │    │ │     │ │     │
│   │ │ B   │ │    │ │ 空  │ │    │ │ H   │ │    │ │ 空  │ │     │
│   │ │ C   │ │    │ │     │ │    │ └─────┘ │    │ │     │ │     │
│   │ │ D   │ │    │ └─────┘ │    └─────────┘    │ └─────┘ │     │
│   │ │ E   │ │    └─────────┘                   └─────────┘     │
│   │ │ F   │ │                                                   │
│   │ └─────┘ │    CPU 1, 3 闲着                                  │
│   └─────────┘    CPU 0 累死了                                   │
│                                                                 │
│   需要把 CPU 0 的进程迁移到 CPU 1, 3                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 负载均衡的触发时机

#### 6.2.1 周期性均衡（Periodic Balancing）

```c
void scheduler_tick(void) {
    // ...
    trigger_load_balance(rq);
}

void trigger_load_balance(struct rq *rq) {
    // 检查是否到了均衡时间
    if (time_after_eq(jiffies, rq->next_balance)) {
        // 触发调度软中断
        raise_softirq(SCHED_SOFTIRQ);
    }
}

// 软中断处理
void run_rebalance_domains(struct softirq_action *h) {
    rebalance_domains(this_rq(), CPU_IDLE);
}
```

**time_after_eq 宏**：
```c
// 用于比较时间，能正确处理 jiffies 溢出
#define time_after_eq(a, b) ((long)((a) - (b)) >= 0)

// jiffies 是内核的心跳计数器，每个 tick 加 1
// 32 位系统上约 49.7 天溢出一次
```

#### 6.2.2 空闲均衡（Idle Balancing）

CPU 空闲时主动"偷"任务：

```c
static void idle_balance(struct rq *this_rq) {
    // 遍历调度域，尝试从其他 CPU 拉任务
    for_each_domain(this_cpu, sd) {
        int pulled = load_balance(this_cpu, sd, CPU_IDLE);
        if (pulled)
            break;  // 拉到了就停止
    }
}
```

#### 6.2.3 唤醒/Fork 均衡

在 `select_task_rq` 中选择负载轻的 CPU（前面已讲）。

### 6.3 调度域（sched_domain）

#### 6.3.1 什么是调度域？

调度域是一组"关系相近"的 CPU，内核按 CPU 的"亲密程度"分层组织。

```
┌─────────────────────────────────────────────────────────────────┐
│                       调度域层次结构                             │
│                                                                 │
│                    ┌─────────────────────────────┐              │
│    Level 2         │        NUMA 域              │              │
│    (最高层)        │   跨 NUMA 节点均衡          │              │
│                    │   均衡间隔: 64ms            │              │
│                    └──────────────┬──────────────┘              │
│                                   │                             │
│                    ┌──────────────┴──────────────┐              │
│                    │                             │              │
│           ┌────────┴────────┐          ┌─────────┴───────┐      │
│  Level 1  │    DIE 域       │          │     DIE 域      │      │
│           │  同一 NUMA 节点 │          │  同一 NUMA 节点 │      │
│           │  均衡间隔: 16ms │          │  均衡间隔: 16ms │      │
│           └────────┬────────┘          └─────────┬───────┘      │
│                    │                             │              │
│           ┌────────┴────────┐          ┌─────────┴───────┐      │
│  Level 0  │    MC 域        │          │     MC 域       │      │
│  (最底层) │  共享 L3 缓存   │          │  共享 L3 缓存   │      │
│           │  均衡间隔: 4ms  │          │  均衡间隔: 4ms  │      │
│           └────────┬────────┘          └─────────┬───────┘      │
│                    │                             │              │
│             ┌──────┴──────┐               ┌──────┴──────┐       │
│             │             │               │             │       │
│           CPU 0        CPU 1           CPU 2        CPU 3       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**域名称说明**：
- **MC (Multi-Core)**：共享缓存的核心
- **DIE**：同一个 CPU 封装（die）
- **NUMA**：跨 NUMA 节点

#### 6.3.2 sched_domain 结构

```c
struct sched_domain {
    // 层次关系
    struct sched_domain *parent;      // 父域（上一层）
    struct sched_domain *child;       // 子域（下一层）
    
    // 域的范围
    struct cpumask span;              // 包含的 CPU
    char *name;                       // 域名称（调试用）
    
    // 均衡参数
    unsigned int min_interval;        // 最小均衡间隔
    unsigned int max_interval;        // 最大均衡间隔
    unsigned int busy_factor;         // 忙时间隔倍数
    unsigned int imbalance_pct;       // 不均衡阈值（百分比）
    unsigned int cache_nice_tries;    // 缓存热进程保护次数
    
    // 调度组
    struct sched_group *groups;       // 域内的调度组
    
    // 统计信息
    unsigned long last_balance;       // 上次均衡时间
    // ...
};
```

**sd 是什么？**
```c
// 内核代码中 sd 是 struct sched_domain * 的惯用缩写
struct sched_domain *sd;

// 类似的缩写：
// rq = run queue
// se = sched_entity
// sg = sched_group
// p = process (task_struct)
```

#### 6.3.3 调度组（sched_group）

每个调度域内部分成若干调度组：

```c
struct sched_group {
    struct sched_group *next;     // 下一个组（环形链表）
    struct cpumask cpumask;       // 组内的 CPU
    unsigned long group_weight;   // 组的权重（CPU 数量）
    struct sched_group_capacity *sgc;  // 组的容量信息
};
```

**均衡时先比较组的负载，再在组内找最忙的 CPU**。

### 6.4 load_balance 函数

负载均衡的核心执行者：

```c
static int load_balance(int this_cpu, struct sched_domain *sd, enum cpu_idle_type idle) {
    struct rq *this_rq = cpu_rq(this_cpu);
    struct sched_group *busiest_group;
    struct rq *busiest_rq;
    unsigned long imbalance;
    int pulled = 0;
    
    // 1. 找到调度域内最忙的调度组
    busiest_group = find_busiest_group(sd, this_cpu, &imbalance);
    if (!busiest_group)
        goto out_balanced;  // 已经均衡
    
    // 2. 找到该组内最忙的 CPU
    busiest_rq = find_busiest_queue(busiest_group, this_cpu);
    if (!busiest_rq)
        goto out_balanced;
    
    // 3. 从最忙 CPU 迁移进程
    while (imbalance > 0) {
        // 锁住两个队列
        double_lock_balance(this_rq, busiest_rq);
        
        // 从 busiest_rq 取下任务
        struct task_struct *p = detach_one_task(busiest_rq, this_cpu);
        if (!p)
            break;
        
        // 放到 this_rq
        attach_one_task(this_rq, p);
        
        pulled++;
        imbalance -= task_h_load(p);
        
        double_unlock_balance(this_rq, busiest_rq);
    }
    
    return pulled;
    
out_balanced:
    return 0;
}
```

### 6.5 迁移的代价与限制

#### 6.5.1 迁移代价

```
┌─────────────────────────────────────────────────────────────────┐
│                        迁移代价                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 缓存失效                                                    │
│     进程在 CPU 0 运行 → 数据在 CPU 0 缓存                       │
│     迁移到 CPU 1 → CPU 1 缓存没有数据 → cache miss              │
│                                                                 │
│  2. TLB 失效                                                    │
│     页表翻译缓存失效，需要重新翻译                               │
│                                                                 │
│  3. NUMA 代价                                                   │
│     跨 NUMA 节点迁移 → 访问内存变慢（2-3倍延迟）                │
│                                                                 │
│  4. 锁竞争                                                      │
│     迁移需要同时锁住两个 CPU 的运行队列                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.5.2 迁移限制

```c
static int can_migrate_task(struct task_struct *p, struct rq *busiest,
                           struct rq *this_rq, int this_cpu) {
    // 1. 检查 CPU 亲和性
    if (!cpumask_test_cpu(this_cpu, p->cpus_ptr))
        return 0;  // 进程不允许在目标 CPU 运行
    
    // 2. 检查是否正在运行
    if (task_running(busiest, p))
        return 0;  // 正在运行的不能迁移
    
    // 3. 检查缓存热度
    if (task_hot(p, busiest))
        return 0;  // 缓存还热，迁移代价大
    
    return 1;
}

// 缓存热度判断
static int task_hot(struct task_struct *p, struct rq *rq) {
    s64 delta = rq_clock_task(rq) - p->se.exec_start;
    
    // 最近运行过的进程，缓存数据还在
    return delta < sysctl_sched_migration_cost;
    // 默认 500us
}
```

### 6.6 均衡策略总结

```
┌─────────────────────────────────────────────────────────────────┐
│                      负载均衡策略                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  均衡原则：                                                     │
│  ├── 先在底层域均衡（代价小，缓存友好）                         │
│  ├── 底层均衡不了，再往上层均衡                                 │
│  └── 越往上，均衡间隔越长（代价大）                             │
│                                                                 │
│  迁移原则：                                                     │
│  ├── 遵守 CPU 亲和性                                            │
│  ├── 不迁移正在运行的进程                                       │
│  ├── 尽量不迁移缓存热的进程                                     │
│  └── 不均衡到一定程度才迁移（避免频繁迁移）                     │
│                                                                 │
│  触发时机：                                                     │
│  ├── 周期性检查（定时器）                                       │
│  ├── CPU 空闲时主动拉任务                                       │
│  └── 进程唤醒/创建时选择负载轻的 CPU                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 任务切换开销

### 7.1 切换时要做什么？

```
┌─────────────────────────────────────────────────────────────────┐
│                      上下文切换步骤                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 保存当前进程状态                                            │
│     ├── 通用寄存器（RAX, RBX, RCX, ...）                        │
│     ├── 栈指针（RSP）                                           │
│     ├── 程序计数器（RIP）                                       │
│     ├── 标志寄存器（RFLAGS）                                    │
│     └── 浮点/SIMD 寄存器（如果使用了）                          │
│                                                                 │
│  2. 切换内核栈                                                  │
│     每个进程有独立的内核栈                                      │
│                                                                 │
│  3. 切换页表（进程切换时）                                      │
│     加载新进程的 CR3 寄存器（页表基地址）                       │
│                                                                 │
│  4. 刷新 TLB（进程切换时）                                      │
│     页表翻译缓存失效                                            │
│                                                                 │
│  5. 恢复新进程状态                                              │
│     与步骤 1 相反                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 进程切换 vs 线程切换

#### 7.2.1 进程切换

```
进程 A → 进程 B（不同地址空间）

需要切换：
├── CPU 寄存器          ✓
├── 内核栈              ✓
├── 页表（mm_struct）   ✓  ← 关键！
├── TLB 刷新            ✓  ← 代价大！
└── 各种上下文          ✓
```

#### 7.2.2 线程切换（同一进程内）

```
线程 A → 线程 B（同属进程 P，共享地址空间）

需要切换：
├── CPU 寄存器          ✓
├── 内核栈              ✓
└── 线程私有数据        ✓

不需要切换：
├── 页表                ✗  ← 省了！
└── TLB                 ✗  ← 省了！
```

#### 7.2.3 为什么线程切换更快？

```
┌─────────────────────────────────────────────────────────────────┐
│                         进程 P                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    共享地址空间                            │  │
│  │              （代码、堆、全局变量、页表）                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│     线程 A              线程 B              线程 C              │
│    ┌──────┐            ┌──────┐            ┌──────┐            │
│    │ 栈   │            │ 栈   │            │ 栈   │            │
│    │寄存器│            │寄存器│            │寄存器│            │
│    │ TLS  │            │ TLS  │            │ TLS  │            │
│    └──────┘            └──────┘            └──────┘            │
│                                                                 │
│    线程切换只需要换这些，地址空间不用动                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 开销分解

#### 7.3.1 直接开销

| 操作 | 进程切换 | 线程切换 | 耗时 |
|------|---------|---------|------|
| 保存/恢复寄存器 | ✓ | ✓ | ~0.5 μs |
| 切换内核栈 | ✓ | ✓ | ~0.5 μs |
| 切换页表 | ✓ | ✗ | ~1 μs |
| 调度器逻辑 | ✓ | ✓ | ~2-3 μs |
| **直接开销合计** | **~5 μs** | **~3 μs** | |

#### 7.3.2 间接开销（更大！）

```
TLB 失效（进程切换）：
├── 进程切换后，TLB 被刷新
├── 新进程的地址翻译全部 miss
├── 每次内存访问都要查页表（慢！）
└── 影响可能持续几百微秒到几毫秒

缓存污染：
├── 新任务的数据不在 L1/L2/L3 缓存
├── 大量 cache miss
└── 要从内存重新加载（慢！）
```

### 7.4 典型开销数据

**测试环境**：现代 x86-64 服务器，3GHz CPU

| 切换类型 | 直接开销 | 加上间接开销 |
|---------|---------|-------------|
| 线程切换（同进程） | ~1-2 μs | ~3-5 μs |
| 进程切换 | ~3-5 μs | ~10-30 μs |
| 跨 NUMA 节点切换 | ~5-10 μs | ~50-100 μs |

**开销构成（进程切换 ~20μs）**：
```
直接开销 (~5μs)：
├── 保存/恢复寄存器: ~0.5μs
├── 切换内核栈: ~0.5μs
├── 切换页表: ~1μs
└── 调度器逻辑: ~3μs

间接开销 (~15μs)：
├── TLB miss 惩罚: ~5-10μs
└── Cache miss 惩罚: ~5-10μs
```

### 7.5 测量方法

#### 7.5.1 使用 lmbench

```bash
# 安装
apt install lmbench

# 测量上下文切换延迟
lat_ctx -s 0 2    # 2 个进程之间切换
lat_ctx -s 0 8    # 8 个进程之间切换
lat_ctx -s 0 16   # 16 个进程之间切换
```

#### 7.5.2 使用 perf

```bash
# 统计上下文切换次数
perf stat -e context-switches,cpu-migrations ./your_program

# 详细分析调度事件
perf record -e sched:sched_switch -a sleep 1
perf report

# 调度延迟分析
perf sched record sleep 5
perf sched latency
```

#### 7.5.3 自己写测试程序

```c
// 用管道让两个进程互相唤醒，测量来回切换时间
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#define ITERATIONS 100000

int main() {
    int pipe1[2], pipe2[2];
    pipe(pipe1);
    pipe(pipe2);
    
    char buf;
    struct timespec start, end;
    
    pid_t pid = fork();
    
    if (pid == 0) {
        // 子进程
        close(pipe1[1]);
        close(pipe2[0]);
        for (int i = 0; i < ITERATIONS; i++) {
            read(pipe1[0], &buf, 1);
            write(pipe2[1], "x", 1);
        }
        exit(0);
    } else {
        // 父进程
        close(pipe1[0]);
        close(pipe2[1]);
        
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (int i = 0; i < ITERATIONS; i++) {
            write(pipe1[1], "x", 1);
            read(pipe2[0], &buf, 1);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        long ns = (end.tv_sec - start.tv_sec) * 1000000000L + 
                  (end.tv_nsec - start.tv_nsec);
        printf("总时间: %.2f ms\n", ns / 1000000.0);
        printf("每次切换: %.2f μs\n", ns / 1000.0 / ITERATIONS / 2);
    }
    
    return 0;
}
```

### 7.6 优化建议

```
┌─────────────────────────────────────────────────────────────────┐
│                      减少切换开销的方法                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 减少进程数，用线程代替                                      │
│     线程切换比进程切换快 5-10 倍                                │
│                                                                 │
│  2. 使用线程池                                                  │
│     避免频繁创建/销毁线程                                       │
│                                                                 │
│  3. CPU 亲和性绑定                                              │
│     taskset -c 0,1 ./program                                    │
│     减少跨 CPU 迁移，保持缓存热度                               │
│                                                                 │
│  4. 使用协程/用户态线程                                         │
│     切换开销可以降到 ~100ns                                     │
│     Go goroutine、Rust async、Python asyncio                    │
│                                                                 │
│  5. 批处理                                                      │
│     减少系统调用次数，减少用户态/内核态切换                     │
│                                                                 │
│  6. 使用 NUMA 感知的内存分配                                    │
│     numactl --membind=0 ./program                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 调度相关命令

### 8.1 查看进程调度信息

#### 8.1.1 ps 命令

```bash
# 查看进程的优先级和 nice 值
ps -eo pid,ni,pri,cls,rtprio,comm

# 输出说明：
# PID   - 进程 ID
# NI    - nice 值 (-20 到 19)
# PRI   - 优先级
# CLS   - 调度类 (TS=普通, FF=FIFO, RR=轮转, -=其他)
# RTPRIO - 实时优先级 (0-99，普通进程显示 -)
# COMM  - 命令名

# 查看特定进程
ps -o pid,ni,pri,cls,rtprio,comm -p $(pgrep nginx)
```

#### 8.1.2 top/htop

```bash
top
# 按 'r' 可以 renice 进程
# NI 列显示 nice 值
# PR 列显示优先级

htop
# 更友好的界面
# F7 降低优先级（增加 nice 值）
# F8 提高优先级（减少 nice 值）
```

#### 8.1.3 /proc 文件系统

```bash
# 查看进程的调度策略和参数
cat /proc/<PID>/sched

# 输出示例：
# policy                    :                    0  (SCHED_NORMAL)
# prio                      :                  120
# nr_switches               :                12345
# nr_voluntary_switches     :                10000
# nr_involuntary_switches   :                 2345

# 查看调度统计
cat /proc/<PID>/schedstat
# 输出：运行时间(ns) 等待时间(ns) 切换次数

# 查看允许运行的 CPU
cat /proc/<PID>/status | grep -E "Cpus_allowed|Mems_allowed"
```

### 8.2 调整 nice 值

#### 8.2.1 nice 命令

```bash
# 以指定 nice 值启动程序
nice -n 10 ./my_program    # nice 值 10（低优先级）
nice -n -5 ./my_program    # nice 值 -5（高优先级，需要 root）
nice -n 0 ./my_program     # 默认 nice 值

# nice 值范围：-20（最高优先级）到 +19（最低优先级）
# 默认值：0
# 只有 root 可以设置负 nice 值
```

#### 8.2.2 renice 命令

```bash
# 修改运行中进程的 nice 值
renice -n 5 -p <PID>       # 设置 nice 值为 5
renice -n -10 -p <PID>     # 设置为 -10（需要 root）

# 修改某用户所有进程
renice -n 5 -u username

# 修改某进程组
renice -n 5 -g <PGID>

# 示例：降低编译任务的优先级
renice -n 19 -p $(pgrep make)
```

### 8.3 设置实时调度策略

#### 8.3.1 chrt 命令

```bash
# 查看进程的调度策略
chrt -p <PID>
# 输出示例：
# pid 1234's current scheduling policy: SCHED_OTHER
# pid 1234's current scheduling priority: 0

# 以 SCHED_FIFO 策略运行（实时，优先级 50）
chrt -f 50 ./my_program

# 以 SCHED_RR 策略运行（实时轮转，优先级 50）
chrt -r 50 ./my_program

# 以 SCHED_OTHER 运行（普通 CFS）
chrt -o 0 ./my_program

# 以 SCHED_BATCH 运行（批处理）
chrt -b 0 ./my_program

# 以 SCHED_IDLE 运行（最低优先级）
chrt -i 0 ./my_program

# 以 SCHED_DEADLINE 运行
chrt -d --sched-runtime 5000000 \
        --sched-deadline 10000000 \
        --sched-period 16666666 \
        0 ./my_program

# 修改运行中进程的策略
chrt -f -p 50 <PID>    # 改为 FIFO，优先级 50
chrt -o -p 0 <PID>     # 改回普通调度
```

#### 8.3.2 调度策略说明

| 策略 | chrt 参数 | 说明 | 优先级范围 |
|------|----------|------|-----------|
| SCHED_OTHER | -o | 普通进程，CFS 调度 | 0（使用 nice 值） |
| SCHED_FIFO | -f | 实时，先进先出 | 1-99 |
| SCHED_RR | -r | 实时，时间片轮转 | 1-99 |
| SCHED_BATCH | -b | 批处理，CPU 密集型 | 0 |
| SCHED_IDLE | -i | 最低优先级 | 0 |
| SCHED_DEADLINE | -d | 截止时间调度 | 0 |

### 8.4 CPU 亲和性设置

#### 8.4.1 taskset 命令

```bash
# 查看进程的 CPU 亲和性
taskset -p <PID>
# 输出：pid 1234's current affinity mask: f
# f = 1111 (二进制) = CPU 0,1,2,3

# 绑定进程到特定 CPU
taskset -c 0 ./my_program        # 只在 CPU 0 运行
taskset -c 0,2 ./my_program      # 在 CPU 0 和 2 运行
taskset -c 0-3 ./my_program      # 在 CPU 0-3 运行
taskset -c 0-3,8-11 ./my_program # 在 CPU 0-3 和 8-11 运行

# 修改运行中进程的亲和性
taskset -pc 0,1 <PID>

# 使用掩码方式
taskset 0x3 ./my_program         # 0x3 = 0011，CPU 0 和 1
taskset 0xf ./my_program         # 0xf = 1111，CPU 0-3
```

#### 8.4.2 numactl 命令

```bash
# 查看 NUMA 拓扑
numactl --hardware
# 输出示例：
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 3
# node 0 size: 32768 MB
# node 1 cpus: 4 5 6 7
# node 1 size: 32768 MB
# node distances:
# node   0   1
#   0:  10  21
#   1:  21  10

# 在指定 NUMA 节点运行
numactl --cpunodebind=0 ./my_program    # 绑定到节点 0 的 CPU
numactl --membind=0 ./my_program        # 内存分配在节点 0
numactl --cpunodebind=0 --membind=0 ./my_program  # 两者都绑定

# 交叉分配内存（适合内存密集型应用）
numactl --interleave=all ./my_program

# 查看进程的 NUMA 统计
numastat -p <PID>
```

### 8.5 查看调度器参数

#### 8.5.1 sysctl 命令

```bash
# 查看 CFS 调度器参数
sysctl kernel.sched_latency_ns           # 调度周期（默认 6ms）
sysctl kernel.sched_min_granularity_ns   # 最小时间片（默认 0.75ms）
sysctl kernel.sched_wakeup_granularity_ns # 唤醒抢占阈值（默认 1ms）
sysctl kernel.sched_migration_cost_ns    # 迁移代价阈值（默认 0.5ms）

# 修改参数（临时）
sysctl -w kernel.sched_latency_ns=6000000

# 永久修改：编辑 /etc/sysctl.conf
echo "kernel.sched_latency_ns = 6000000" >> /etc/sysctl.conf
sysctl -p
```

#### 8.5.2 调度域信息

```bash
# 查看 CPU 0 的调度域
ls /proc/sys/kernel/sched_domain/cpu0/
# 输出：domain0  domain1  domain2

# 查看域的详细参数
cat /proc/sys/kernel/sched_domain/cpu0/domain0/name
# 输出：MC（Multi-Core，共享缓存）

cat /proc/sys/kernel/sched_domain/cpu0/domain0/min_interval
# 输出：4（毫秒）

cat /proc/sys/kernel/sched_domain/cpu0/domain0/max_interval
# 输出：8（毫秒）

# 查看所有域
for i in 0 1 2; do
    echo "=== domain$i ==="
    cat /proc/sys/kernel/sched_domain/cpu0/domain$i/name 2>/dev/null
done
```

### 8.6 性能分析工具

#### 8.6.1 perf sched

```bash
# 记录调度事件
perf sched record sleep 5

# 查看调度延迟
perf sched latency
# 输出示例：
#  Task                  |   Runtime ms  | Switches | Avg delay ms |
# -----------------------|---------------|----------|--------------|
#  nginx:1234            |     50.123    |    1000  |    0.050     |

# 查看调度时间线
perf sched map

# 查看调度统计
perf sched timehist
```

#### 8.6.2 trace-cmd / ftrace

```bash
# 跟踪调度事件
trace-cmd record -e sched_switch -e sched_wakeup sleep 5
trace-cmd report

# 直接使用 ftrace
cd /sys/kernel/debug/tracing
echo 1 > events/sched/sched_switch/enable
cat trace
echo 0 > events/sched/sched_switch/enable
```

#### 8.6.3 其他工具

```bash
# schedtool（需安装）
schedtool <PID>              # 查看调度信息
schedtool -F -p 50 <PID>     # 设置 FIFO
schedtool -R -p 50 <PID>     # 设置 RR
schedtool -N <PID>           # 恢复普通调度

# mpstat（CPU 统计）
mpstat -P ALL 1              # 每秒显示所有 CPU 的统计

# pidstat（进程统计）
pidstat -w 1                 # 每秒显示上下文切换统计
```

### 8.7 命令速查表

| 目的 | 命令 |
|------|------|
| 查看 nice 值 | `ps -eo pid,ni,comm` |
| 启动时设置 nice | `nice -n 10 ./prog` |
| 修改 nice 值 | `renice -n 5 -p PID` |
| 查看调度策略 | `chrt -p PID` |
| 设置 FIFO 调度 | `chrt -f 50 ./prog` |
| 设置 RR 调度 | `chrt -r 50 ./prog` |
| 查看 CPU 亲和性 | `taskset -p PID` |
| 绑定 CPU | `taskset -c 0,1 ./prog` |
| 查看 NUMA 拓扑 | `numactl --hardware` |
| 绑定 NUMA 节点 | `numactl --cpunodebind=0 ./prog` |
| 查看调度参数 | `sysctl kernel.sched_latency_ns` |
| 调度延迟分析 | `perf sched latency` |
| 上下文切换统计 | `perf stat -e context-switches ./prog` |
| 查看进程调度信息 | `cat /proc/PID/sched` |

---

## 9. 内核常见宏与技巧

在阅读内核代码时，会遇到很多宏和技巧，这里总结常见的几个。

### 9.1 container_of 宏

**作用**：已知结构体某个成员的指针，反推出整个结构体的指针。

```c
#define container_of(ptr, type, member) ({                  \
    const typeof(((type *)0)->member) *__mptr = (ptr);      \
    (type *)((char *)__mptr - offsetof(type, member));      \
})

// 使用示例
struct sched_entity {
    struct rb_node run_node;  // 红黑树节点
    u64 vruntime;
};

// 已知 run_node 的指针，获取 sched_entity 的指针
struct rb_node *node = ...;
struct sched_entity *se = container_of(node, struct sched_entity, run_node);
```

**原理**：
```
结构体起始地址 = 成员地址 - 成员在结构体中的偏移量

内存布局：
┌─────────────────────────────────────┐
│         sched_entity                │
├──────────────┬──────────────────────┤
│    其他成员   │     run_node         │  ← 已知这个地址
├──────────────┴──────────────────────┤
│    vruntime                         │
└─────────────────────────────────────┘
↑                ↑
结构体起始        run_node 的位置
(要求的)         (已知的)
```

### 9.2 rb_entry 宏

**作用**：`container_of` 的别名，用于红黑树。

```c
#define rb_entry(ptr, type, member) container_of(ptr, type, member)

// 使用示例
struct rb_node *left = cfs_rq->tasks_timeline.rb_leftmost;
struct sched_entity *se = rb_entry(left, struct sched_entity, run_node);
```

### 9.3 likely / unlikely 宏

**作用**：给编译器提示分支预测信息，优化性能。

```c
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// 使用示例
if (likely(condition)) {
    // 这个分支更可能执行
}

if (unlikely(error)) {
    // 这个分支很少执行
}
```

### 9.4 时间比较宏

**作用**：正确处理 jiffies 溢出的时间比较。

```c
#define time_after(a, b)     ((long)((b) - (a)) < 0)
#define time_before(a, b)    time_after(b, a)
#define time_after_eq(a, b)  ((long)((a) - (b)) >= 0)
#define time_before_eq(a, b) time_after_eq(b, a)

// 使用示例
if (time_after_eq(jiffies, rq->next_balance)) {
    // 到了均衡时间
}
```

### 9.5 常见缩写

| 缩写 | 全称 | 说明 |
|------|------|------|
| rq | run queue | 运行队列 |
| cfs_rq | CFS run queue | CFS 运行队列 |
| rt_rq | RT run queue | 实时运行队列 |
| dl_rq | Deadline run queue | Deadline 运行队列 |
| se | sched_entity | 调度实体 |
| sd | sched_domain | 调度域 |
| sg | sched_group | 调度组 |
| p | process | 进程（task_struct） |
| curr | current | 当前进程 |
| prev | previous | 前一个进程 |
| next | next | 下一个进程 |

---

## 总结

Linux 进程调度是一个复杂而精妙的系统，本文涵盖了以下核心内容：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Linux 进程调度知识图谱                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 调度器演进                                                  │
│     O(n) → O(1) → CFS                                          │
│     从简单遍历到位图优化到红黑树公平调度                         │
│                                                                 │
│  2. 调度器架构                                                  │
│     Per-CPU rq → dl_rq / rt_rq / cfs_rq                        │
│     多级队列满足不同场景需求                                    │
│                                                                 │
│  3. CFS 核心                                                    │
│     vruntime + 红黑树 + 权重                                    │
│     追求完全公平的调度                                          │
│                                                                 │
│  4. 调度时机                                                    │
│     主动让出 + 被动抢占                                         │
│     TIF_NEED_RESCHED 标志协调                                   │
│                                                                 │
│  5. 任务队列选择                                                │
│     亲和性 + 缓存 + 负载 + NUMA                                 │
│     多因素权衡选择最优 CPU                                      │
│                                                                 │
│  6. 负载均衡                                                    │
│     调度域分层 + 周期性/空闲均衡                                │
│     权衡均衡收益和迁移代价                                      │
│                                                                 │
│  7. 切换开销                                                    │
│     线程切换 < 进程切换 < 跨 NUMA 切换                          │
│     间接开销（TLB/缓存）往往更大                                │
│                                                                 │
│  8. 调度命令                                                    │
│     nice/renice/chrt/taskset/numactl/perf                      │
│     查看和调整调度行为的工具箱                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

理解这些原理，对于：
- **系统性能优化**：知道瓶颈在哪里，如何调整
- **问题排查**：理解调度延迟、CPU 不均衡等问题的根因
- **架构设计**：选择合适的并发模型（进程/线程/协程）

都有很大帮助。

---

*本文基于《深入理解 Linux 进程与内存》学习整理，结合 AI 辅助答疑完成。如有错误欢迎指正。*

*最后更新：2025-12-24*
