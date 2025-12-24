# Linux 性能统计原理

> 📖 参考书籍：《深入理解 Linux 进程与内存》
>
> 🤖 笔记方式：阅读书籍 + AI 辅助答疑整理
>
> 本文深入分析 Linux 系统中各种性能指标的统计原理，理解 top、vmstat、perf 等工具背后的数据来源。

## 目录

0. [时间子系统基础](#0-时间子系统基础)
1. [负载（Load Average）计算原理](#1-负载load-average计算原理)
2. [CPU 利用率统计原理](#2-cpu-利用率统计原理)
3. [CPI 与 perf 性能分析](#3-cpi-与-perf-性能分析)
4. [总结：时间子系统与性能统计的关系](#4-总结时间子系统与性能统计的关系)

---

## 0. 时间子系统基础

时间子系统是 Linux 内核的基础设施，**几乎所有性能统计都依赖它**。在学习具体的性能指标之前，先理解时间子系统的工作原理。

### 0.1 为什么需要时间子系统？

```
┌─────────────────────────────────────────────────────────────────┐
│                 时间子系统的作用                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 提供时间度量                                                │
│     系统时间、进程运行时间、定时器                              │
│                                                                 │
│  2. 驱动周期性任务                                              │
│     调度器 tick、负载统计、CPU 时间统计                         │
│                                                                 │
│  3. 支持定时器和延迟                                            │
│     sleep()、定时任务、超时处理                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 0.2 核心概念

#### HZ 和 tick

```c
// HZ = 每秒的时钟中断次数（内核配置）
// 常见值：100、250、1000

#define HZ 1000  // 每秒 1000 次中断

// tick = 一次时钟中断的时间间隔
// HZ=1000 → 1 tick = 1ms
// HZ=250  → 1 tick = 4ms
// HZ=100  → 1 tick = 10ms
```

**HZ 的选择是权衡**：
| HZ 值 | 优点 | 缺点 |
|-------|------|------|
| 高（1000） | 响应快、统计精确 | 中断开销大 |
| 低（100） | 中断开销小 | 响应慢、统计粗糙 |

#### jiffies

```c
// jiffies = 系统启动以来的 tick 计数
extern unsigned long volatile jiffies;

// 每次时钟中断：jiffies++

// 常用转换
jiffies_to_msecs(j)   // jiffies → 毫秒
msecs_to_jiffies(m)   // 毫秒 → jiffies
time_after(a, b)      // 时间比较（处理溢出）
```

**查看当前系统的 HZ 值**：
```bash
# 方法 1：通过 /proc/config.gz（如果内核编译时启用）
zcat /proc/config.gz | grep CONFIG_HZ

# 方法 2：通过 getconf
getconf CLK_TCK  # 返回 USER_HZ，通常是 100

# 方法 3：计算（jiffies 增长速度）
# 内核 HZ 和用户态 USER_HZ 可能不同
# 内核常用 1000，用户态接口统一用 100
```

#### 时钟源（Clocksource）

```
硬件时钟源（精度从高到低）：
├── TSC（Time Stamp Counter）
│   CPU 内部计数器，每个时钟周期 +1，最精确
│
├── HPET（High Precision Event Timer）
│   独立硬件定时器，精度高
│
├── ACPI PM Timer
│   电源管理定时器，精度中等
│
└── PIT（Programmable Interval Timer）
    传统 8254 芯片，精度最低
```

### 0.3 时钟中断处理流程

这是理解性能统计的关键：

```
┌─────────────────────────────────────────────────────────────────┐
│                    时钟中断处理流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   硬件定时器产生中断                                            │
│            ↓                                                    │
│   tick_periodic() 或 tick_sched_handle()                        │
│            ↓                                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  do_timer(1)                                             │  │
│   │       └── jiffies++  (更新系统 tick 计数)               │  │
│   └─────────────────────────────────────────────────────────┘  │
│            ↓                                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  update_process_times(user_tick)                         │  │
│   │       │                                                  │  │
│   │       ├── account_process_tick()                         │  │
│   │       │       └── 【CPU 时间统计】                       │  │
│   │       │           us/sy/ni/id/wa/hi/si 累加              │  │
│   │       │                                                  │  │
│   │       ├── run_local_timers()                             │  │
│   │       │       └── 处理本地定时器                         │  │
│   │       │                                                  │  │
│   │       └── scheduler_tick()                               │  │
│   │               │                                          │  │
│   │               ├── 更新当前进程的 vruntime                │  │
│   │               ├── 检查是否需要调度                       │  │
│   │               ├── trigger_load_balance()                 │  │
│   │               └── calc_global_load_tick()                │  │
│   │                       └── 【负载统计】                   │  │
│   │                           累加活跃任务数                 │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 0.4 关键代码路径

```c
// 时钟中断入口
void tick_periodic(int cpu) {
    if (tick_do_timer_cpu == cpu) {
        do_timer(1);  // 更新 jiffies
    }
    update_process_times(user_mode(get_irq_regs()));
}

// 更新进程时间（每个 tick 调用）
void update_process_times(int user_tick) {
    struct task_struct *p = current;
    
    // 1. CPU 时间统计（本章第 2 节详解）
    account_process_tick(p, user_tick);
    
    // 2. 运行本地定时器
    run_local_timers();
    
    // 3. 调度器 tick
    scheduler_tick();
}

// 调度器 tick
void scheduler_tick(void) {
    struct rq *rq = this_rq();
    
    // 更新运行队列时钟
    update_rq_clock(rq);
    
    // 调用调度类的 tick 处理（更新 vruntime 等）
    curr->sched_class->task_tick(rq, curr, 0);
    
    // 触发负载均衡检查
    trigger_load_balance(rq);
    
    // 负载统计（本章第 1 节详解）
    calc_global_load_tick(rq);
}
```

### 0.5 tickless 模式（NO_HZ）

现代内核支持 tickless 模式，CPU 空闲时停止周期性 tick：

```
┌─────────────────────────────────────────────────────────────────┐
│              周期性 tick vs tickless（NO_HZ）                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  周期性 tick（传统）：                                           │
│  │ tick │ tick │ tick │ tick │ tick │ tick │                   │
│  ├──────┼──────┼──────┼──────┼──────┼──────┤                   │
│  │ 运行 │ 运行 │ 空闲 │ 空闲 │ 空闲 │ 运行 │                   │
│                  ↑      ↑      ↑                                │
│              空闲时也有 tick（浪费电）                           │
│                                                                 │
│  tickless（NO_HZ）：                                             │
│  │ tick │ tick │              │ tick │                         │
│  ├──────┼──────┼──────────────┼──────┤                         │
│  │ 运行 │ 运行 │ 空闲（无tick）│ 运行 │                         │
│                  ↑                                              │
│              空闲时停止 tick（省电）                             │
│                                                                 │
│  影响：                                                         │
│  ├── 省电，适合笔记本、移动设备                                 │
│  └── 统计需要特殊处理（补偿空闲期间的时间）                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 0.6 时间子系统与性能统计的关系

```
┌─────────────────────────────────────────────────────────────────┐
│              时间子系统驱动的性能统计                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   时钟中断（tick）                                              │
│        │                                                        │
│        ├──→ 【负载统计】                                        │
│        │    每个 tick: calc_global_load_tick()                  │
│        │    每 5 秒: calc_global_load() 计算平均负载            │
│        │                                                        │
│        ├──→ 【CPU 时间统计】                                    │
│        │    每个 tick: account_process_tick()                   │
│        │    判断 user/system/idle/iowait 并累加                 │
│        │                                                        │
│        ├──→ 【进程时间统计】                                    │
│        │    每个 tick: 更新 p->utime / p->stime                 │
│        │    /proc/<pid>/stat 中的时间数据                       │
│        │                                                        │
│        └──→ 【调度统计】                                        │
│             每个 tick: 更新 vruntime、检查时间片                │
│             perf sched 的数据来源                               │
│                                                                 │
│   注意：perf 的硬件计数器（cycles、instructions）               │
│         不依赖 tick，而是直接读取 CPU 的 PMU                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 负载（Load Average）计算原理

### 1.1 从 top 命令说起

当我们执行 `top` 或 `uptime` 命令时，会看到负载信息：

```bash
$ uptime
 10:30:45 up 5 days,  3:22,  2 users,  load average: 1.25, 0.98, 0.75
                                                     ↑     ↑     ↑
                                                   1分钟 5分钟 15分钟

$ cat /proc/loadavg
1.25 0.98 0.75 2/150 12345
↑    ↑    ↑    ↑     ↑
1min 5min 15min 运行/总进程 最近PID
```

**数据来源**：内核全局变量 `avenrun[3]`，通过 `/proc/loadavg` 暴露给用户空间。

```c
// fs/proc/loadavg.c
static int loadavg_proc_show(struct seq_file *m, void *v) {
    unsigned long avnrun[3];
    
    get_avenrun(avnrun, FIXED_1/200, 0);
    
    seq_printf(m, "%lu.%02lu %lu.%02lu %lu.%02lu %ld/%d %d\n",
        LOAD_INT(avnrun[0]), LOAD_FRAC(avnrun[0]),
        LOAD_INT(avnrun[1]), LOAD_FRAC(avnrun[1]),
        LOAD_INT(avnrun[2]), LOAD_FRAC(avnrun[2]),
        nr_running(), nr_threads,
        idr_get_cursor(...));
    return 0;
}
```

### 1.2 负载的含义

**负载 = 正在运行 + 等待运行 + 等待不可中断 I/O 的任务数**

```
┌─────────────────────────────────────────────────────────────────┐
│                      负载的组成                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  负载 = R 状态任务数 + D 状态任务数                              │
│                                                                 │
│  R (TASK_RUNNING)：                                             │
│  ├── 正在 CPU 上运行的进程                                      │
│  └── 在运行队列中等待 CPU 的进程                                │
│                                                                 │
│  D (TASK_UNINTERRUPTIBLE)：                                     │
│  └── 等待不可中断 I/O 的进程（如磁盘 I/O）                      │
│                                                                 │
│  注意：S 状态（可中断睡眠）不计入负载！                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**如何解读负载值？**

| 负载值 | 4核CPU含义 | 系统状态 |
|--------|-----------|----------|
| 1.0 | 平均 1 个任务在运行/等待 | 空闲（25%利用） |
| 4.0 | 平均 4 个任务在运行/等待 | 满载（100%利用） |
| 8.0 | 平均 8 个任务在运行/等待 | 过载（有任务排队） |

**经验法则**：负载 < CPU 核心数 × 0.7 为健康状态。

### 1.3 时间子系统与瞬时负载

#### 1.3.1 时钟中断触发统计

Linux 时间子系统通过时钟中断驱动负载统计：

```
时钟中断
    ↓
tick_periodic() / tick_sched_handle()
    ↓
update_process_times()
    ↓
scheduler_tick()
    ↓
calc_global_load_tick()  ← 统计瞬时负载
```

#### 1.3.2 瞬时负载的统计

```c
// kernel/sched/loadavg.c

// 全局变量：所有 CPU 的活跃任务数之和
atomic_long_t calc_load_tasks;

// 每个 CPU 在 scheduler_tick 中调用
void calc_global_load_tick(struct rq *this_rq) {
    long delta = calc_load_fold_active(this_rq, 0);
    if (delta)
        atomic_long_add(delta, &calc_load_tasks);
}

// 计算活跃任务数
static long calc_load_fold_active(struct rq *this_rq, long adjust) {
    long nr_active, delta = 0;
    
    // 活跃 = 正在运行/等待运行 + 不可中断睡眠
    nr_active = this_rq->nr_running - adjust;
    nr_active += (long)this_rq->nr_uninterruptible;
    
    if (nr_active != this_rq->calc_load_active) {
        delta = nr_active - this_rq->calc_load_active;
        this_rq->calc_load_active = nr_active;
    }
    
    return delta;
}
```

**瞬时负载计算图示**：

```
┌─────────────────────────────────────────────────────────────────┐
│                     瞬时负载的组成                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CPU 0          CPU 1          CPU 2          CPU 3            │
│  ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐         │
│  │R: 3  │       │R: 2  │       │R: 1  │       │R: 2  │         │
│  │D: 1  │       │D: 0  │       │D: 2  │       │D: 0  │         │
│  └──────┘       └──────┘       └──────┘       └──────┘         │
│     4              2              3              2              │
│                                                                 │
│   瞬时负载 = 4 + 2 + 3 + 2 = 11                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3.3 进程状态与负载的关系

```c
// 进程状态定义
#define TASK_RUNNING           0x0000  // R：运行或就绪 → 计入负载 ✓
#define TASK_INTERRUPTIBLE     0x0001  // S：可中断睡眠 → 不计入负载 ✗
#define TASK_UNINTERRUPTIBLE   0x0002  // D：不可中断睡眠 → 计入负载 ✓
#define TASK_STOPPED           0x0004  // T：停止 → 不计入负载 ✗
#define TASK_ZOMBIE            0x0020  // Z：僵尸 → 不计入负载 ✗
```

### 1.4 平均负载的计算

#### 1.4.1 为什么需要平均？

瞬时负载波动太大，没有参考价值：

```
时刻 1：负载 = 2
时刻 2：负载 = 15
时刻 3：负载 = 3
时刻 4：负载 = 20
```

需要使用**指数加权移动平均（EWMA）**进行平滑处理。

#### 1.4.2 EWMA 算法

```
公式：load(t) = load(t-1) × exp + active × (1 - exp)

其中 exp 是衰减因子：
- 1 分钟：exp = e^(-5/60)  ≈ 0.9200
- 5 分钟：exp = e^(-5/300) ≈ 0.9835
- 15分钟：exp = e^(-5/900) ≈ 0.9945
```

**衰减因子的含义**：

| 时间窗口 | 衰减因子 | 含义 |
|---------|---------|------|
| 1 分钟 | 0.9200 | 每 5 秒旧数据权重变为 92%，约 1 分钟后影响降到 37% |
| 5 分钟 | 0.9835 | 衰减更慢，更平滑 |
| 15 分钟 | 0.9945 | 衰减最慢，最平滑 |

#### 1.4.3 内核实现

```c
// 每 5 秒计算一次
#define LOAD_FREQ  (5*HZ + 1)

// 衰减因子（定点数表示）
#define FSHIFT   11
#define FIXED_1  (1 << FSHIFT)  // 2048，表示 1.0
#define EXP_1    1884           // 1884/2048 ≈ 0.9200
#define EXP_5    2014           // 2014/2048 ≈ 0.9835
#define EXP_15   2037           // 2037/2048 ≈ 0.9945

void calc_global_load(unsigned long ticks) {
    long active = atomic_long_read(&calc_load_tasks);
    active = active > 0 ? active * FIXED_1 : 0;
    
    // EWMA 更新
    avenrun[0] = calc_load(avenrun[0], EXP_1, active);
    avenrun[1] = calc_load(avenrun[1], EXP_5, active);
    avenrun[2] = calc_load(avenrun[2], EXP_15, active);
}

static unsigned long calc_load(unsigned long load, unsigned long exp, unsigned long active) {
    unsigned long newload;
    
    // newload = load × exp + active × (1 - exp)
    newload = load * exp + active * (FIXED_1 - exp);
    
    if (active >= load)
        newload += FIXED_1 - 1;
    
    return newload / FIXED_1;
}
```

#### 1.4.4 定点数运算

内核不能使用浮点运算，所以用定点数：

```c
#define FSHIFT   11                    // 小数位数
#define FIXED_1  (1 << FSHIFT)         // 2048，表示 1.0

// 转换为显示值
#define LOAD_INT(x)  ((x) >> FSHIFT)                          // 整数部分
#define LOAD_FRAC(x) LOAD_INT(((x) & (FIXED_1-1)) * 100)      // 小数部分

// 例：avenrun[0] = 2560
// LOAD_INT(2560) = 2560 >> 11 = 1
// LOAD_FRAC(2560) = ((2560 & 2047) * 100) >> 11 = (512 * 100) >> 11 = 25
// 显示为 1.25
```

### 1.5 负载 ≠ CPU 使用率

#### 1.5.1 关键区别

| 指标 | 统计内容 | 含义 |
|------|---------|------|
| CPU 使用率 | CPU 实际工作时间比例 | CPU 有多忙 |
| 负载 | 需要 CPU 的任务数量 | 系统有多繁忙 |

#### 1.5.2 场景分析

**场景 1：高负载，低 CPU 使用率**

```
原因：大量进程在等待 I/O（D 状态）

┌─────────────────────────────────────────┐
│  进程 A-F: D 状态（等待磁盘 I/O）       │
│  进程 G: R 状态（正在运行）             │
│                                         │
│  负载 = 7（很高）                       │
│  CPU 使用率 = 很低（CPU 在等待 I/O）    │
│                                         │
│  瓶颈：磁盘 I/O，不是 CPU               │
└─────────────────────────────────────────┘
```

**场景 2：低负载，高 CPU 使用率**

```
原因：单个 CPU 密集型任务

┌─────────────────────────────────────────┐
│  4 核 CPU                               │
│                                         │
│  CPU 0: 进程 A（100% 使用）             │
│  CPU 1-3: idle                          │
│                                         │
│  负载 ≈ 1.0（低）                       │
│  CPU 0 使用率 = 100%                    │
│  整体 CPU 使用率 = 25%                  │
└─────────────────────────────────────────┘
```

**场景 3：高负载，高 CPU 使用率**

```
原因：大量 CPU 密集型任务

┌─────────────────────────────────────────┐
│  4 核 CPU，8 个 CPU 密集型进程          │
│                                         │
│  每个 CPU 运行 1 个，等待 1 个          │
│                                         │
│  负载 ≈ 8.0（高）                       │
│  CPU 使用率 ≈ 100%                      │
│                                         │
│  瓶颈：CPU                              │
└─────────────────────────────────────────┘
```

#### 1.5.3 为什么 D 状态计入负载？

1. **历史原因**：早期 Unix 设计，负载反映"系统繁忙程度"
2. **实际意义**：D 状态进程虽不占 CPU，但占用系统资源，反映整体压力
3. **争议**：有人认为不应计入，但 Linux 保持了这个设计

### 1.6 负载计算完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    负载计算完整流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 时钟中断触发 scheduler_tick()                               │
│                    ↓                                            │
│  2. 每个 CPU 统计 nr_running + nr_uninterruptible               │
│                    ↓                                            │
│  3. 累加到全局 calc_load_tasks（瞬时负载）                      │
│                    ↓                                            │
│  4. 每 5 秒，用 EWMA 更新 avenrun[0/1/2]（平均负载）            │
│                    ↓                                            │
│  5. top/uptime 读取 /proc/loadavg 显示                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.7 实用技巧

#### 查看负载

```bash
# 实时查看
uptime
top
cat /proc/loadavg

# 历史趋势（需要安装 sysstat）
sar -q 1 10
```

#### 解读负载趋势

```
1min > 5min > 15min：负载在上升 📈
1min < 5min < 15min：负载在下降 📉
三者接近：负载稳定 ➡️
```

#### 判断瓶颈

```bash
# 高负载时，检查是 CPU 还是 I/O 瓶颈
vmstat 1
# 看 r 列（等待 CPU）和 b 列（等待 I/O）

# 或者
top
# 按 1 查看各 CPU 使用率
# 看 %wa（I/O 等待）是否很高
```

---

## 2. CPU 利用率统计原理

### 2.1 CPU 时间的分类

top 命令显示的 CPU 时间分为三大类：

```
┌─────────────────────────────────────────────────────────────────┐
│                    CPU 时间 = 100%                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────┐                                           │
│   │    用户态时间    │  us + ni                                  │
│   │   (User Mode)   │  进程在用户空间执行代码                    │
│   └─────────────────┘                                           │
│            +                                                    │
│   ┌─────────────────┐                                           │
│   │    内核态时间    │  sy + hi + si                             │
│   │  (Kernel Mode)  │  进程在内核空间执行代码                    │
│   │                 │  （系统调用、中断处理等）                  │
│   └─────────────────┘                                           │
│            +                                                    │
│   ┌─────────────────┐                                           │
│   │    空闲时间      │  id + wa                                  │
│   │     (Idle)      │  CPU 没有任务执行                         │
│   └─────────────────┘                                           │
│                                                                 │
│   三者之和 = 100%                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**top 显示的各字段含义**：

```
%Cpu(s):  5.2 us,  2.1 sy,  1.0 ni, 89.0 id,  0.5 wa,  0.1 hi,  0.1 si,  0.0 st
```

| 字段 | 全称 | 含义 |
|------|------|------|
| us | user | 用户态时间（nice ≤ 0 的进程） |
| ni | nice | 低优先级用户态时间（nice > 0 的进程） |
| sy | system | 内核态时间（系统调用等） |
| id | idle | 空闲时间 |
| wa | iowait | I/O 等待时间（CPU 空闲但有进程等待 I/O） |
| hi | irq | 硬中断处理时间 |
| si | softirq | 软中断处理时间 |
| st | steal | 虚拟机被宿主机偷走的时间 |

### 2.2 数据来源：/proc/stat

top 命令读取 `/proc/stat` 获取 CPU 时间数据：

```bash
$ cat /proc/stat
cpu  74608 2520 57292 1362404 10860 0 912 0 0 0
cpu0 18894 638 14292 340451 2714 0 456 0 0 0
cpu1 18590 620 14311 340478 2709 0 228 0 0 0
...
```

**各列含义**：
```
cpu  user nice system idle iowait irq softirq steal guest guest_nice
```

**注意**：这些值是**累计值**（从系统启动开始），单位是 USER_HZ（通常是 1/100 秒）。

### 2.3 内核如何统计 CPU 时间

#### 2.3.1 统计时机：时钟中断

CPU 时间统计发生在每个时钟中断（tick）：

```
时钟中断
    ↓
tick_periodic() / tick_sched_handle()
    ↓
update_process_times()
    ↓
account_process_tick()  ← 统计 CPU 时间
```

#### 2.3.2 核心统计函数

```c
// kernel/sched/cputime.c
void account_process_tick(struct task_struct *p, int user_tick) {
    u64 cputime = TICK_NSEC;  // 一个 tick 的时间
    
    if (in_irq()) {
        // 硬中断上下文 → hi
        account_irq_time();
    } 
    else if (in_softirq()) {
        // 软中断上下文 → si
        account_softirq_time();
    }
    else if (user_tick) {
        // 用户态
        if (task_nice(p) > 0)
            // nice > 0 → ni
            account_nice_time();
        else
            // nice ≤ 0 → us
            account_user_time();
    }
    else if (p == idle_task) {
        // idle 进程
        if (atomic_read(&rq->nr_iowait) > 0)
            // 有进程等待 I/O → wa
            account_iowait_time();
        else
            // 真正空闲 → id
            account_idle_time();
    }
    else {
        // 内核态 → sy
        account_system_time();
    }
}
```

#### 2.3.3 为什么区分 user 和 nice？

```c
void account_user_time(struct task_struct *p, u64 cputime) {
    int index = CPUTIME_USER;
    
    // nice > 0 的进程单独统计
    if (task_nice(p) > 0)
        index = CPUTIME_NICE;
    
    cpustat[index] += cputime;
}
```

**目的**：区分"重要任务"和"后台任务"的 CPU 消耗

| 场景 | 含义 |
|------|------|
| us 高，ni 低 | CPU 花在正常/重要任务上 |
| us 低，ni 高 | CPU 花在低优先级后台任务上（如编译、备份） |

### 2.4 数据存储结构

#### 2.4.1 Per-CPU 统计结构

```c
// 每个 CPU 的统计数据
struct kernel_cpustat {
    u64 cpustat[NR_STATS];
};

// 索引定义
enum cpu_usage_stat {
    CPUTIME_USER,       // 用户态
    CPUTIME_NICE,       // nice 用户态
    CPUTIME_SYSTEM,     // 内核态
    CPUTIME_SOFTIRQ,    // 软中断
    CPUTIME_IRQ,        // 硬中断
    CPUTIME_IDLE,       // 空闲
    CPUTIME_IOWAIT,     // I/O 等待
    CPUTIME_STEAL,      // 虚拟机 steal
    CPUTIME_GUEST,      // 虚拟机 guest
    CPUTIME_GUEST_NICE,
    NR_STATS,
};

// Per-CPU 变量
DEFINE_PER_CPU(struct kernel_cpustat, kernel_cpustat);
```

#### 2.4.2 统计函数更新数据

```c
// 累加用户态时间
void account_user_time(struct task_struct *p, u64 cputime) {
    int index = (task_nice(p) > 0) ? CPUTIME_NICE : CPUTIME_USER;
    
    // 累加到 Per-CPU 统计
    cpustat[index] += cputime;
    
    // 同时更新进程自己的统计
    p->utime += cputime;
}

// 累加空闲时间
void account_idle_time(u64 cputime) {
    struct rq *rq = this_rq();
    
    if (atomic_read(&rq->nr_iowait) > 0)
        cpustat[CPUTIME_IOWAIT] += cputime;  // 有 I/O 等待
    else
        cpustat[CPUTIME_IDLE] += cputime;    // 真正空闲
}
```

### 2.5 /proc/stat 如何生成

```c
// fs/proc/stat.c
static int show_stat(struct seq_file *p, void *v) {
    u64 user, nice, system, idle, iowait, irq, softirq, steal;
    
    // 汇总所有 CPU 的数据
    for_each_possible_cpu(i) {
        struct kernel_cpustat *kcs = &kcpustat_cpu(i);
        
        user += kcs->cpustat[CPUTIME_USER];
        nice += kcs->cpustat[CPUTIME_NICE];
        system += kcs->cpustat[CPUTIME_SYSTEM];
        idle += kcs->cpustat[CPUTIME_IDLE];
        iowait += kcs->cpustat[CPUTIME_IOWAIT];
        irq += kcs->cpustat[CPUTIME_IRQ];
        softirq += kcs->cpustat[CPUTIME_SOFTIRQ];
        steal += kcs->cpustat[CPUTIME_STEAL];
    }
    
    // 转换单位：内核用纳秒，用户态用 USER_HZ (1/100秒)
    seq_printf(p, "cpu  %llu %llu %llu %llu %llu %llu %llu %llu ...\n",
        nsec_to_clock_t(user),
        nsec_to_clock_t(nice),
        nsec_to_clock_t(system),
        nsec_to_clock_t(idle),
        nsec_to_clock_t(iowait),
        nsec_to_clock_t(irq),
        nsec_to_clock_t(softirq),
        nsec_to_clock_t(steal));
    
    return 0;
}
```

### 2.6 top 如何计算 CPU 利用率

#### 2.6.1 两次采样计算差值

/proc/stat 的值是**累计值**，top 需要两次采样计算差值：

```c
// 第一次采样
read_proc_stat(&t1_user, &t1_nice, &t1_system, &t1_idle, ...);
t1_total = t1_user + t1_nice + t1_system + t1_idle + t1_iowait + ...;

// 等待一段时间（如 1 秒）
sleep(interval);

// 第二次采样
read_proc_stat(&t2_user, &t2_nice, &t2_system, &t2_idle, ...);
t2_total = t2_user + t2_nice + t2_system + t2_idle + t2_iowait + ...;

// 计算差值
total_diff = t2_total - t1_total;
user_diff = t2_user - t1_user;
system_diff = t2_system - t1_system;
idle_diff = t2_idle - t1_idle;
// ...

// 计算百分比
user_pct = user_diff * 100.0 / total_diff;
system_pct = system_diff * 100.0 / total_diff;
idle_pct = idle_diff * 100.0 / total_diff;
// ...
```

#### 2.6.2 CPU 利用率公式

```
CPU 利用率 = (用户态 + 内核态) / 总时间 × 100%
           = (us + ni + sy + hi + si) / total × 100%
           = 100% - (id + wa + st)
```

### 2.7 完整流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                  CPU 利用率统计完整流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. 时钟中断触发（每个 tick）                            │   │
│  │     tick_periodic() → update_process_times()            │   │
│  │                           ↓                             │   │
│  │     account_process_tick() 判断当前状态                 │   │
│  │     ├── 用户态 + nice≤0 → us                            │   │
│  │     ├── 用户态 + nice>0 → ni                            │   │
│  │     ├── 内核态 → sy                                     │   │
│  │     ├── 硬中断 → hi                                     │   │
│  │     ├── 软中断 → si                                     │   │
│  │     ├── 空闲 + 有I/O等待 → wa                           │   │
│  │     └── 空闲 + 无I/O等待 → id                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. 累加到 Per-CPU 统计结构                              │   │
│  │     kernel_cpustat.cpustat[CPUTIME_xxx] += cputime      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. /proc/stat 读取时汇总                                │   │
│  │     show_stat() 遍历所有 CPU，汇总各项时间              │   │
│  │     转换单位（纳秒 → USER_HZ），格式化输出              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  4. top 读取并计算                                       │   │
│  │     两次采样 /proc/stat，计算差值                       │   │
│  │     差值 / 总差值 × 100 = 百分比                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.8 注意事项

#### 2.8.1 采样误差

时钟中断是**采样统计**，不是精确统计：

```
例：一个进程在两次 tick 之间：
    用户态运行 3ms → 进入内核态 → 内核态运行 7ms
    
    如果 tick 发生在内核态：
    → 整个 10ms 都被计为内核态时间（不准确！）
```

#### 2.8.2 iowait 的含义

```
iowait = CPU 空闲，但有进程在等待 I/O

注意：
├── iowait 高不一定是坏事
├── 可能只是 I/O 密集型任务的正常表现
└── 需要结合磁盘 I/O 指标一起分析
```

#### 2.8.3 steal 时间（虚拟化环境）

```
在虚拟机中：
├── 虚拟机想运行，但宿主机没给 CPU 时间
├── 这段时间被计为 steal
└── steal 高说明宿主机资源紧张，需要调整
```

### 2.9 实用命令

```bash
# 查看 CPU 利用率
top                    # 实时查看
htop                   # 更友好的界面
mpstat -P ALL 1        # 每秒显示所有 CPU

# 查看原始数据
cat /proc/stat

# 历史趋势
sar -u 1 10            # 每秒采样，共 10 次

# 按 CPU 查看
top 然后按 1           # 显示每个 CPU 的利用率
```

---

## 3. CPI 与 perf 性能分析

### 3.1 什么是 CPI？

**CPI（Cycles Per Instruction）= CPU 周期数 / 指令数**

表示平均执行一条指令需要多少个 CPU 时钟周期。

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPI 的含义                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CPI = 1.0  → 平均每条指令用 1 个周期（理想情况）               │
│  CPI = 2.0  → 平均每条指令用 2 个周期（有等待/停顿）            │
│  CPI = 0.5  → 平均每条指令用 0.5 个周期（超标量/流水线并行）    │
│                                                                 │
│  CPI 越低，CPU 效率越高                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**CPI 参考值**：

| CPI 范围 | 含义 |
|---------|------|
| < 1.0 | 非常好，充分利用超标量/并行 |
| 1.0 - 2.0 | 正常范围 |
| 2.0 - 4.0 | 有优化空间，可能有 cache miss |
| > 4.0 | 性能问题，大量停顿 |

### 3.2 导致 CPI 升高的原因

```
┌─────────────────────────────────────────────────────────────────┐
│                    导致 CPI 升高的原因                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Cache Miss（缓存未命中）                                    │
│     L1 miss → 等待 L2（~10 周期）                               │
│     L2 miss → 等待 L3（~40 周期）                               │
│     L3 miss → 等待内存（~200 周期）                             │
│                                                                 │
│  2. 分支预测失败                                                │
│     预测错误 → 流水线清空 → 重新取指（~15-20 周期）             │
│                                                                 │
│  3. 数据依赖                                                    │
│     后一条指令依赖前一条的结果 → 等待                           │
│                                                                 │
│  4. 资源冲突                                                    │
│     多条指令竞争同一执行单元                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 perf 工具简介

perf 是 Linux 内核自带的性能分析工具，可以访问 CPU 的**硬件性能计数器（PMU）**。

```
┌─────────────────────────────────────────────────────────────────┐
│                        perf 架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   用户态                                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  perf 命令行工具                                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           ↓ 系统调用                            │
│   内核态                                                        │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  perf_event 子系统                                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           ↓                                     │
│   硬件                                                          │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  PMU（Performance Monitoring Unit）                      │  │
│   │  硬件性能计数器                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 perf stat：统计性能指标

```bash
# 统计程序的基本性能指标
$ perf stat ./my_program

 Performance counter stats for './my_program':

          1,234.56 msec task-clock                #    0.998 CPUs utilized
               123      context-switches          #    0.100 K/sec
                 5      cpu-migrations            #    0.004 K/sec
            12,345      page-faults               #    0.010 M/sec
     3,456,789,012      cycles                    #    2.800 GHz
     1,234,567,890      instructions              #    0.36  insn per cycle (IPC)
       234,567,890      branches                  #  190.000 M/sec
        12,345,678      branch-misses             #    5.26% of all branches

       1.236789012 seconds time elapsed
```

**关键指标**：
- `cycles`：CPU 周期数
- `instructions`：执行的指令数
- `insn per cycle`：IPC = 1/CPI，每周期执行的指令数
- `branch-misses`：分支预测失败率

**计算 CPI**：
```bash
CPI = cycles / instructions
    = 3,456,789,012 / 1,234,567,890
    ≈ 2.8

# 或者：CPI = 1 / IPC = 1 / 0.36 ≈ 2.8
```

### 3.5 perf 常用命令

```bash
# 统计基本指标
perf stat ./program

# 指定统计事件
perf stat -e cycles,instructions,cache-misses,branch-misses ./program

# 统计 cache 相关
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./program

# 采样记录
perf record -g ./program

# 查看采样报告
perf report

# 实时查看系统热点
perf top

# 查看特定进程
perf top -p <PID>
```

### 3.6 常用性能事件

| 事件名 | 含义 |
|--------|------|
| cycles | CPU 周期数 |
| instructions | 执行的指令数 |
| cache-references | 缓存访问次数 |
| cache-misses | 缓存未命中次数 |
| branch-instructions | 分支指令数 |
| branch-misses | 分支预测失败数 |
| L1-dcache-load-misses | L1 数据缓存未命中 |
| LLC-load-misses | 最后一级缓存未命中 |
| context-switches | 上下文切换次数 |
| cpu-migrations | CPU 迁移次数 |

### 3.7 CPI 分析示例

```bash
# CPU 密集型程序
$ perf stat -e cycles,instructions ./cpu_bound

   cycles:       10,000,000,000
   instructions:  8,000,000,000
   CPI = 1.25  ← 还不错

# 内存密集型程序
$ perf stat -e cycles,instructions ./memory_bound

   cycles:       10,000,000,000
   instructions:  2,000,000,000
   CPI = 5.0   ← 很高，可能有大量 cache miss
```

**进一步分析高 CPI**：
```bash
# 检查 cache miss
perf stat -e cycles,instructions,cache-misses,LLC-load-misses ./program

# 如果 cache-misses 很高 → 内存访问模式问题
# 如果 branch-misses 很高 → 分支预测问题
```

### 3.8 性能分析流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    性能分析流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. perf stat 看整体指标                                        │
│     └── 关注 CPI (或 IPC)                                       │
│                                                                 │
│  2. CPI 高？检查原因                                            │
│     ├── cache-misses 高 → 内存访问问题                          │
│     ├── branch-misses 高 → 分支预测问题                         │
│     └── 都不高 → 可能是数据依赖或资源冲突                       │
│                                                                 │
│  3. perf record + report 定位热点                               │
│     └── 找到消耗 CPU 最多的函数                                 │
│                                                                 │
│  4. 针对性优化                                                  │
│     ├── cache miss → 优化数据结构、访问模式                     │
│     ├── branch miss → 减少分支、使用 likely/unlikely            │
│     └── 热点函数 → 算法优化、向量化                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.9 火焰图（Flame Graph）

火焰图是可视化 perf 采样数据的利器，能直观看到 CPU 时间花在哪里：

```bash
# 1. 采样记录（-g 表示记录调用栈）
perf record -g -p <PID> -- sleep 30
# 或者直接运行程序
perf record -g ./my_program

# 2. 生成火焰图（需要 FlameGraph 工具）
# 下载：git clone https://github.com/brendangregg/FlameGraph

# 3. 转换数据
perf script > out.perf
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded
./FlameGraph/flamegraph.pl out.folded > flamegraph.svg

# 4. 用浏览器打开 flamegraph.svg
```

**火焰图解读**：
```
┌─────────────────────────────────────────────────────────────────┐
│                      火焰图解读                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    main()                                │   │
│  ├─────────────────────────┬───────────────────────────────┤   │
│  │       func_a()          │          func_b()             │   │
│  ├───────────┬─────────────┼───────────────────────────────┤   │
│  │ helper1() │  helper2()  │          func_c()             │   │
│  └───────────┴─────────────┴───────────────────────────────┘   │
│                                                                 │
│  - 横轴：函数占用的 CPU 时间比例（越宽越耗时）                  │
│  - 纵轴：调用栈深度（从下往上是调用关系）                       │
│  - 颜色：随机，无特殊含义                                       │
│  - 平顶：CPU 时间消耗的地方（优化重点）                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 总结：时间子系统与性能统计的关系

### 4.1 时间子系统是性能统计的基础设施

通过前面的学习，我们可以看到时间子系统贯穿了整个性能统计体系：

```
┌─────────────────────────────────────────────────────────────────┐
│              时间子系统与性能统计的关系总览                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────┐                          │
│                    │   时钟中断      │                          │
│                    │  (tick_periodic)│                          │
│                    └────────┬────────┘                          │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              ↓              ↓              ↓                    │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│     │ jiffies++  │  │ 调度器tick │  │ CPU时间统计│             │
│     │ 系统时间   │  │ scheduler_ │  │ account_   │             │
│     │            │  │ tick()     │  │ process_   │             │
│     │            │  │            │  │ tick()     │             │
│     └────────────┘  └─────┬──────┘  └─────┬──────┘             │
│                           │               │                     │
│              ┌────────────┴───┐           │                     │
│              ↓                ↓           ↓                     │
│     ┌────────────┐   ┌────────────┐  ┌────────────┐            │
│     │ 负载统计   │   │ vruntime   │  │ us/sy/id   │            │
│     │ calc_      │   │ 更新       │  │ wa/hi/si   │            │
│     │ global_    │   │            │  │ 累加       │            │
│     │ load_tick()│   │            │  │            │            │
│     └─────┬──────┘   └────────────┘  └─────┬──────┘            │
│           │                                │                    │
│           ↓                                ↓                    │
│     ┌────────────┐                   ┌────────────┐            │
│     │ avenrun[]  │                   │ /proc/stat │            │
│     │ 平均负载   │                   │ CPU利用率  │            │
│     └─────┬──────┘                   └─────┬──────┘            │
│           │                                │                    │
│           ↓                                ↓                    │
│     ┌────────────┐                   ┌────────────┐            │
│     │ uptime/top │                   │ top/mpstat │            │
│     │ 负载显示   │                   │ CPU%显示   │            │
│     └────────────┘                   └────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 各性能指标的统计时机

| 性能指标 | 统计时机 | 统计函数 | 数据来源 |
|---------|---------|---------|---------|
| 负载（Load Average） | 每个 tick + 每 5 秒 | calc_global_load_tick() + calc_global_load() | /proc/loadavg |
| CPU 利用率 | 每个 tick | account_process_tick() | /proc/stat |
| 进程 CPU 时间 | 每个 tick | account_user_time() / account_system_time() | /proc/[pid]/stat |
| CPI/硬件计数器 | 硬件 PMU | perf_event 子系统 | perf 命令 |

### 4.3 tick 驱动 vs 硬件计数器

```
┌─────────────────────────────────────────────────────────────────┐
│                 两种统计方式的对比                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【tick 驱动的统计】                                            │
│  ├── 负载、CPU 利用率、进程时间                                 │
│  ├── 精度受 HZ 限制（如 HZ=1000 → 1ms 精度）                   │
│  ├── 采样统计，有误差                                           │
│  └── 开销低，始终运行                                           │
│                                                                 │
│  【硬件计数器统计】                                              │
│  ├── cycles、instructions、cache-miss、branch-miss             │
│  ├── 精度极高（CPU 周期级别）                                   │
│  ├── 精确统计，无采样误差                                       │
│  └── 需要显式启用（perf）                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 性能分析工具与数据来源

```
┌─────────────────────────────────────────────────────────────────┐
│                 工具与数据来源对应关系                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  工具              数据来源              统计方式               │
│  ─────────────────────────────────────────────────────────────  │
│  uptime            /proc/loadavg         tick 驱动              │
│  top               /proc/stat            tick 驱动              │
│  vmstat            /proc/stat + meminfo  tick 驱动              │
│  mpstat            /proc/stat            tick 驱动              │
│  pidstat           /proc/[pid]/stat      tick 驱动              │
│  perf stat         PMU 硬件计数器        硬件计数器             │
│  perf record       PMU + 采样            硬件计数器 + 采样      │
│  perf top          PMU + 采样            硬件计数器 + 采样      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 理解时间子系统的意义

1. **知道数据从哪来**：看到 top 的 CPU 利用率，知道它是 tick 采样统计的，有精度限制
2. **理解统计误差**：短时间的 CPU 突刺可能被 tick 采样错过
3. **选择合适工具**：需要精确分析用 perf，日常监控用 top/vmstat
4. **排查问题**：高负载低 CPU 时，知道去看 D 状态进程和 I/O

### 4.6 本章知识点回顾

```
┌─────────────────────────────────────────────────────────────────┐
│                    本章知识点总结                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  第 0 节：时间子系统基础                                        │
│  ├── HZ、tick、jiffies 的概念                                   │
│  ├── 时钟中断处理流程                                           │
│  └── tickless 模式                                              │
│                                                                 │
│  第 1 节：负载计算原理                                          │
│  ├── 负载 = R 状态 + D 状态任务数                               │
│  ├── 瞬时负载：calc_global_load_tick()                         │
│  ├── 平均负载：EWMA 算法，每 5 秒更新                           │
│  └── 负载 ≠ CPU 使用率                                          │
│                                                                 │
│  第 2 节：CPU 利用率统计                                        │
│  ├── us/sy/ni/id/wa/hi/si/st 各字段含义                        │
│  ├── account_process_tick() 每 tick 统计                       │
│  ├── /proc/stat 累计值                                          │
│  └── top 两次采样计算差值                                       │
│                                                                 │
│  第 3 节：CPI 与 perf                                           │
│  ├── CPI = cycles / instructions                                │
│  ├── perf stat 统计硬件计数器                                   │
│  ├── perf record/report 采样分析                                │
│  └── 火焰图可视化                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 实战：性能问题排查思路

当遇到性能问题时，可以按以下思路排查：

```
┌─────────────────────────────────────────────────────────────────┐
│                 性能问题排查流程                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 先看负载和 CPU                                              │
│     $ uptime                    # 看负载趋势                    │
│     $ top                       # 看 CPU 各项指标               │
│                                                                 │
│  2. 判断瓶颈类型                                                │
│     ┌──────────────────┬──────────────────────────────────┐    │
│     │ 现象             │ 可能原因                          │    │
│     ├──────────────────┼──────────────────────────────────┤    │
│     │ 高负载 + 高CPU   │ CPU 密集型任务多                  │    │
│     │ 高负载 + 低CPU   │ I/O 瓶颈（看 wa 和 D 状态进程）   │    │
│     │ 低负载 + 高CPU   │ 单进程 CPU 密集                   │    │
│     │ 高 sy%           │ 系统调用频繁或内核问题            │    │
│     │ 高 wa%           │ 磁盘 I/O 慢                       │    │
│     │ 高 si%           │ 网络流量大或软中断问题            │    │
│     └──────────────────┴──────────────────────────────────┘    │
│                                                                 │
│  3. 深入分析                                                    │
│     $ vmstat 1              # 看 r/b 列，判断 CPU/IO 等待       │
│     $ pidstat -u 1          # 看哪个进程消耗 CPU                │
│     $ perf top              # 看哪个函数是热点                  │
│     $ perf stat -p <PID>    # 看进程的 CPI                      │
│                                                                 │
│  4. 针对性优化                                                  │
│     CPU 瓶颈 → 算法优化、并行化、减少锁竞争                     │
│     I/O 瓶颈 → 异步 I/O、缓存、更快的存储                       │
│     内存瓶颈 → 减少分配、优化数据结构、增加缓存                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**常用排查命令速查**：

```bash
# 整体概览
uptime                    # 负载
top                       # CPU、内存、进程
vmstat 1                  # 系统整体状态

# CPU 分析
mpstat -P ALL 1           # 每个 CPU 的利用率
pidstat -u 1              # 每个进程的 CPU
perf top                  # 实时热点函数
perf stat -p <PID>        # 进程的硬件计数器

# 进程状态
ps aux                    # 所有进程
ps -eo pid,stat,comm      # 看进程状态（R/S/D/Z）

# 查找 D 状态进程（高负载低 CPU 时）
ps aux | awk '$8 ~ /D/'
```

---

*本文基于《深入理解 Linux 进程与内存》学习整理*

*最后更新：2024-12-25*
