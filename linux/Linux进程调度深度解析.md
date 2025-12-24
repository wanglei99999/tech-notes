# Linux 进程调度深度解析

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

---

## 1. 调度器演进历史

### 1.1 为什么需要调度器？

CPU 是稀缺资源，而系统中有大量进程需要运行。调度器的职责就是**决定哪个进程在什么时候使用 CPU**。

一个好的调度器需要平衡：
- **响应性**：交互式任务要快速响应
- **吞吐量**：尽可能多地完成工作
- **公平性**：每个进程都能获得合理的 CPU 时间

### 1.2 O(n) 调度器（Linux 2.4 及之前）

**工作原理：**
- 所有可运行进程放在一个全局运行队列
- 每次调度时，遍历整个队列计算每个进程的优先级（goodness 值）
- 选择优先级最高的进程运行

**问题：**
```
调度开销 = O(n)，n = 可运行进程数
```
- 进程越多，调度延迟越大
- SMP（多处理器）系统下，全局队列需要加锁，竞争严重
- 实时性差，不适合服务器场景

### 1.3 O(1) 调度器（Linux 2.6，2003年）

由 Ingo Molnar 开发，核心改进是使用**位图 + 优先级数组**实现 O(1) 查找。

**关键数据结构：**
```c
struct runqueue {
    struct prio_array *active;   // 活跃数组
    struct prio_array *expired;  // 过期数组
};

struct prio_array {
    unsigned long bitmap[5];          // 140位的位图
    struct list_head queue[140];      // 每个优先级一个链表
};
```

**O(1) 如何实现：**

1. **位图快速查找**：140 个优先级，用 bitmap 标记哪些优先级有进程，`find_first_bit()` 是硬件指令，O(1)
2. **双数组交换**：active 数组存放有时间片的进程，expired 存放时间片用完的。当 active 为空时，直接交换指针，O(1)
3. **Per-CPU 运行队列**：每个 CPU 维护独立队列，减少锁竞争

### 1.4 CFS 调度器（Linux 2.6.23+，2007年至今）

O(1) 调度器后来被 CFS（Completely Fair Scheduler）取代：
- 使用红黑树，复杂度 O(log n)
- 但实际性能更好，公平性更强
- 这是目前现代内核使用的调度器

---

## 2. 现代调度器架构

### 2.1 Per-CPU 运行队列

内核为每个逻辑 CPU 维护一个独立的运行队列（rq）：

```
CPU 0                          CPU 1
  │                              │
  ▼                              ▼
┌─────────────────┐        ┌─────────────────┐
│      rq         │        │      rq         │
├─────────────────┤        ├─────────────────┤
│ dl_rq (红黑树)  │        │ dl_rq (红黑树)  │
├─────────────────┤        ├─────────────────┤
│ rt_rq (位图)    │        │ rt_rq (位图)    │
├─────────────────┤        ├─────────────────┤
│ cfs_rq (红黑树) │        │ cfs_rq (红黑树) │
└─────────────────┘        └─────────────────┘
```

### 2.2 调度类层次

Linux 支持多种调度类，不同类型的进程有不同的调度需求：

| 调度类 | 对应队列 | 适用场景 | 调度策略 |
|--------|----------|----------|----------|
| dl_sched_class | dl_rq | 硬实时任务 | SCHED_DEADLINE |
| rt_sched_class | rt_rq | 软实时任务 | SCHED_FIFO, SCHED_RR |
| fair_sched_class | cfs_rq | 普通进程 | SCHED_NORMAL, SCHED_BATCH |
| idle_sched_class | - | 空闲任务 | SCHED_IDLE |

**调度优先级顺序：Deadline > 实时 > CFS > Idle**

### 2.3 各队列的数据结构

```c
struct rq {
    struct cfs_rq cfs;          // CFS 队列（红黑树）
    struct rt_rq rt;            // 实时队列（位图+链表）
    struct dl_rq dl;            // Deadline 队列（红黑树）
    struct task_struct *curr;   // 当前运行的进程
    struct task_struct *idle;   // idle 进程
};
```

**rt_rq 保留了 O(1) 的设计**：
```c
struct rt_rq {
    struct rt_prio_array active;  // 优先级数组
};

struct rt_prio_array {
    DECLARE_BITMAP(bitmap, MAX_RT_PRIO+1);  // 100 个优先级的位图
    struct list_head queue[MAX_RT_PRIO];     // 每个优先级一个链表
};
```

### 2.4 Idle 进程

每个 CPU 都有一个 idle 进程（PID 0），作用：
- **占位符**：CPU 必须始终有进程在运行
- **省电**：让 CPU 进入低功耗状态（执行 HLT/MWAIT 指令）

```c
void cpu_idle_loop() {
    while (1) {
        while (!need_resched()) {
            cpuidle_idle_call();  // 进入低功耗状态
        }
        schedule();  // 有进程可运行了，切换出去
    }
}
```

### 2.5 Deadline 进程

SCHED_DEADLINE 是给**硬实时任务**设计的，特点是有明确的截止时间：

```
"我必须在 X 时间内完成 Y 工作量，周期是 Z"
```

三个关键参数：
- `sched_runtime`：每周期需要的 CPU 时间
- `sched_deadline`：截止时间
- `sched_period`：任务周期

调度器使用 EDF（Earliest Deadline First）算法，按 deadline 最近的优先调度。

---

## 3. CFS 完全公平调度器

### 3.1 核心思想

CFS 的目标：**让每个进程获得相等的 CPU 时间份额**

理想情况下，如果有 N 个进程，每个进程应该同时获得 1/N 的 CPU。但 CPU 不能真正"同时"运行多个进程，所以 CFS 用**虚拟运行时间（vruntime）**来模拟公平。

### 3.2 vruntime：公平的度量

```
vruntime = 进程实际运行时间 × (NICE_0_LOAD / 进程权重)
```

**核心规则：vruntime 最小的进程优先运行**

```
进程 A：vruntime = 10
进程 B：vruntime = 5
进程 C：vruntime = 8

红黑树排列（按 vruntime）：
        B(5)
       /    \
     C(8)   A(10)

下一个运行：B（最左节点，vruntime 最小）
```

### 3.3 权重与 nice 值

不同优先级的进程，vruntime 增长速度不同：

```c
// nice 值 → 权重 映射（部分）
static const int prio_to_weight[40] = {
    /* -20 */ 88761, 71755, 56483, ...
    /*   0 */ 1024,   // nice 0 的基准权重
    /*  19 */ 15,     // 最低优先级
};
```

**权重越高，vruntime 增长越慢，获得更多 CPU 时间**

```
delta_vruntime = delta_exec × (1024 / weight)

例：
nice 0 进程运行 10ms：delta_vruntime = 10 × (1024/1024) = 10
nice -5 进程运行 10ms：delta_vruntime = 10 × (1024/3121) ≈ 3.3
```

### 3.4 cfs_rq 数据结构

```c
struct cfs_rq {
    struct load_weight load;              // 队列总权重
    unsigned int nr_running;              // 可运行进程数
    struct rb_root_cached tasks_timeline; // 红黑树（按 vruntime 排序）
    u64 min_vruntime;                     // 队列中最小的 vruntime
    struct sched_entity *curr;            // 当前运行的调度实体
};

struct sched_entity {
    struct load_weight load;        // 权重
    struct rb_node run_node;        // 红黑树节点
    u64 vruntime;                   // 虚拟运行时间
};
```

### 3.5 调度流程

**选择下一个进程（O(1)）：**
```c
pick_next_entity(struct cfs_rq *cfs_rq) {
    // 直接取红黑树最左节点（已缓存）
    struct rb_node *left = cfs_rq->tasks_timeline.rb_leftmost;
    return rb_entry(left, struct sched_entity, run_node);
}
```

> **关于 rb_entry 宏**：这是 `container_of` 的别名，作用是已知结构体某个成员的指针，反推出整个结构体的指针。这是 Linux 内核的常见技巧。

**进程入队/出队**：O(log n)，红黑树插入/删除

### 3.6 min_vruntime 的作用

新进程的 vruntime 设为多少？
- 设为 0？新进程会一直抢占
- 设为当前最大？新进程要等很久

**解决**：用 `min_vruntime` 作为基准，它是队列中最小 vruntime 的单调递增版本。

### 3.7 时间片

CFS 没有固定时间片，而是动态计算：

```c
sched_period = max(sysctl_sched_latency, nr_running × sysctl_sched_min_granularity)
time_slice = sched_period × (进程权重 / 队列总权重)
```

---

## 4. 调度时机

### 4.1 两种调度方式

**主动调度（自愿）**：进程主动放弃 CPU
```c
schedule();          // 直接让出
mutex_lock();        // 拿不到锁，睡眠
wait_event();        // 等待事件
msleep();            // 主动睡眠
```

**被动调度（抢占）**：进程被迫让出 CPU
```c
// 内核设置 TIF_NEED_RESCHED 标志
set_tsk_need_resched(current);
// 在特定时机检查这个标志，触发调度
```

### 4.2 调度时机详解

| 时机 | 类型 | 说明 |
|------|------|------|
| 时钟中断 | 被动 | 每个 tick 检查时间片是否用完 |
| 中断/系统调用返回 | 被动 | 返回用户态前检查 TIF_NEED_RESCHED |
| 进程唤醒 | 被动 | 唤醒的进程可能抢占当前进程 |
| 进程创建 | 被动 | 新进程可能抢占父进程 |
| 等待资源 | 主动 | 锁、I/O、睡眠 |
| 进程退出 | 主动 | exit() 后最后一次调度 |
| 内核抢占点 | 被动 | spin_unlock/preempt_enable 后检查 |

### 4.3 TIF_NEED_RESCHED 标志

这是调度的核心机制：

```c
// 设置标志（标记需要调度）
set_tsk_need_resched(task);

// 检查标志
need_resched();

// 清除标志
clear_tsk_need_resched(task);
```

**为什么用标志而不是直接调度？**
- 中断上下文不能直接调度
- 必须等到安全的时机才能调度

### 4.4 内核抢占

现代 Linux 是抢占内核：

```c
// preempt_count == 0 且 TIF_NEED_RESCHED 被设置 → 可以抢占

// 关键位置会检查
spin_unlock();       // 释放锁后检查
preempt_enable();    // 开启抢占后检查
```

---

## 5. 任务队列选择

### 5.1 核心问题

一个进程要运行/唤醒时，应该放到哪个 CPU 的 rq 上？

### 5.2 关键考虑因素

1. **CPU 亲和性**：进程可能绑定了特定 CPU
2. **缓存亲和性**：尽量选之前运行过的 CPU（缓存热）
3. **负载均衡**：选负载最轻的 CPU
4. **NUMA 拓扑**：尽量选同一 NUMA 节点的 CPU

### 5.3 选择优先级

```
1. 必须满足 CPU 亲和性（cpus_mask）
          ↓
2. 优先选之前运行的 CPU（缓存热）
          ↓
3. 如果之前的 CPU 忙，选同一缓存域的空闲 CPU
          ↓
4. 如果都忙，选同一 NUMA 节点的最空闲 CPU
          ↓
5. 最后才考虑跨 NUMA 节点
```

---

## 6. 负载均衡

### 6.1 为什么需要负载均衡？

每个 CPU 有独立的运行队列，可能出现：
```
CPU 0: 10 个进程排队
CPU 1: 0 个进程（空闲）
```

需要把进程从忙的 CPU 迁移到闲的 CPU。

### 6.2 均衡触发时机

1. **周期性均衡**：定时器触发，通过 `trigger_load_balance()` 发起软中断
2. **空闲均衡**：CPU 空闲时主动"偷"任务
3. **唤醒/fork 均衡**：选择负载轻的 CPU

### 6.3 trigger_load_balance 函数

```c
void trigger_load_balance(struct rq *rq) {
    // 检查是否到了均衡时间
    if (time_after_eq(jiffies, rq->next_balance)) {
        raise_softirq(SCHED_SOFTIRQ);  // 触发软中断
    }
}
```

> **time_after_eq 宏**：用于比较时间，能正确处理 jiffies 溢出的情况。
> `time_after_eq(a, b)` 表示 `a >= b`。

### 6.4 调度域（sched_domain）

内核把 CPU 按"亲密程度"分层：

```
                    ┌─────────────────────────────┐
    Level 2         │        NUMA 域              │
                    │   跨 NUMA 节点均衡          │
                    └──────────────┬──────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           │                                               │
  ┌────────┴────────┐                          ┌───────────┴───────┐
  │    DIE 域       │                          │      DIE 域       │
  │  同一 NUMA 节点 │                          │   同一 NUMA 节点  │
  └────────┬────────┘                          └───────────┬───────┘
           │                                               │
  ┌────────┴────────┐                          ┌───────────┴───────┐
  │    MC 域        │                          │      MC 域        │
  │  共享 L3 缓存   │                          │   共享 L3 缓存    │
  └─────────────────┘                          └───────────────────┘
```

**均衡原则**：先在底层域均衡（代价小），再往上层均衡。

### 6.5 sched_domain 结构

```c
struct sched_domain {
    struct sched_domain *parent;      // 父域
    struct sched_domain *child;       // 子域
    struct cpumask span;              // 包含的 CPU
    unsigned int min_interval;        // 最小均衡间隔
    unsigned int max_interval;        // 最大均衡间隔
    struct sched_group *groups;       // 域内的调度组
};
```

> **sd 是什么？** 内核代码中 `sd` 是 `struct sched_domain *` 的惯用缩写。

### 6.6 load_balance 函数

负载均衡的核心执行者：

```
load_balance(this_cpu, sd)
│
├── 1. find_busiest_group(sd)     // 找最忙的调度组
├── 2. find_busiest_queue(group)  // 找最忙的 CPU
├── 3. calculate_imbalance()      // 计算迁移量
├── 4. detach_tasks()             // 从忙 CPU 取下任务
└── 5. attach_tasks()             // 放到当前 CPU
```

### 6.7 迁移的代价

不是所有进程都适合迁移：
- **缓存失效**：迁移后缓存数据要重新加载
- **NUMA 代价**：跨节点访问内存变慢
- **限制条件**：CPU 亲和性、正在运行的进程、缓存热的进程

---

## 7. 任务切换开销

### 7.1 切换时要做什么？

1. 保存当前任务的状态（寄存器、栈指针等）
2. 切换内核栈
3. 切换页表（如果是进程切换）
4. 恢复下一个任务的状态
5. 刷新各种缓存（TLB、流水线等）

### 7.2 进程切换 vs 线程切换

| 操作 | 进程切换 | 线程切换 |
|------|---------|---------|
| 保存/恢复寄存器 | ✓ | ✓ |
| 切换内核栈 | ✓ | ✓ |
| 切换页表 | ✓ | ✗ |
| TLB 刷新 | ✓ | ✗ |

**线程切换更快**：同一进程的线程共享地址空间，不需要切换页表和刷新 TLB。

### 7.3 典型开销

| 切换类型 | 直接开销 | 加上间接开销 |
|---------|---------|-------------|
| 线程切换（同进程） | ~1-2 μs | ~3-5 μs |
| 进程切换 | ~3-5 μs | ~10-30 μs |
| 跨 NUMA 节点切换 | ~5-10 μs | ~50-100 μs |

### 7.4 间接开销更大

- **TLB 失效**：进程切换后地址翻译全部 miss
- **缓存污染**：新任务的数据不在缓存，大量 cache miss

### 7.5 测量方法

```bash
# 使用 lmbench
lat_ctx -s 0 2

# 使用 perf
perf stat -e context-switches,cpu-migrations ./program
```

---

## 8. 调度相关命令

### 8.1 查看进程调度信息

```bash
# 查看优先级和 nice 值
ps -eo pid,ni,pri,cls,comm

# 查看调度统计
cat /proc/<PID>/schedstat
```

### 8.2 调整 nice 值

```bash
# 启动时设置
nice -n 10 ./program

# 修改运行中进程
renice -n 5 -p <PID>
```

### 8.3 设置实时调度策略

```bash
# 查看调度策略
chrt -p <PID>

# 设置 SCHED_FIFO
chrt -f 50 ./program

# 设置 SCHED_RR
chrt -r 50 ./program
```

### 8.4 CPU 亲和性

```bash
# 查看亲和性
taskset -p <PID>

# 绑定到特定 CPU
taskset -c 0,1 ./program
```

### 8.5 NUMA 相关

```bash
# 查看 NUMA 拓扑
numactl --hardware

# 绑定到 NUMA 节点
numactl --cpunodebind=0 ./program
```

### 8.6 调度器参数

```bash
# 查看 CFS 参数
sysctl kernel.sched_latency_ns
sysctl kernel.sched_min_granularity_ns

# 查看调度域
cat /proc/sys/kernel/sched_domain/cpu0/domain0/name
```

### 8.7 性能分析

```bash
# 调度延迟分析
perf sched record sleep 5
perf sched latency

# 跟踪调度事件
trace-cmd record -e sched_switch sleep 5
```

### 8.8 命令速查表

| 目的 | 命令 |
|------|------|
| 查看 nice 值 | `ps -eo pid,ni,comm` |
| 设置 nice | `nice -n 10 ./prog` |
| 修改 nice | `renice -n 5 -p PID` |
| 查看调度策略 | `chrt -p PID` |
| 设置实时调度 | `chrt -f 50 ./prog` |
| 查看 CPU 亲和性 | `taskset -p PID` |
| 绑定 CPU | `taskset -c 0,1 ./prog` |
| 查看 NUMA | `numactl --hardware` |
| 调度参数 | `sysctl kernel.sched_latency_ns` |
| 性能分析 | `perf sched latency` |

---

## 总结

Linux 进程调度是一个复杂而精妙的系统：

1. **调度器演进**：从 O(n) → O(1) → CFS，不断优化
2. **多级队列**：dl_rq、rt_rq、cfs_rq 满足不同场景需求
3. **CFS 核心**：vruntime + 红黑树实现公平调度
4. **调度时机**：主动让出 + 被动抢占，通过 TIF_NEED_RESCHED 标志协调
5. **负载均衡**：调度域分层，权衡均衡收益和迁移代价
6. **切换开销**：线程切换比进程切换快 5-10 倍

理解这些原理，对于系统性能优化、问题排查都有很大帮助。

---

*本文基于 Linux 内核学习整理，如有错误欢迎指正。*
