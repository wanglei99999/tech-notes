# Embedding 原理详解

## 一、什么是 Embedding？

Embedding 是一种将离散符号（如词、句子）映射到连续向量空间的技术。数学上，这是一个映射函数：

$$f: \mathcal{V} \rightarrow \mathbb{R}^d$$

其中 $\mathcal{V}$ 是词汇表（离散空间），$\mathbb{R}^d$ 是 d 维实数向量空间（连续空间）。

### 为什么需要 Embedding？

1. **计算机无法直接处理文本**：需要数值化表示
2. **One-hot 编码的缺陷**：
   - 维度爆炸：词汇表大小 = 向量维度（通常 10万+）
   - 稀疏表示：每个向量只有一个 1
   - 无法表达语义相似性：任意两个词的距离相同

```
One-hot 示例（词汇表大小=5）：
"猫" = [1, 0, 0, 0, 0]
"狗" = [0, 1, 0, 0, 0]
"汽车" = [0, 0, 1, 0, 0]

问题：cos("猫", "狗") = cos("猫", "汽车") = 0
但语义上，"猫"和"狗"应该更接近！
```

---

## 二、Word2Vec：词向量的开山之作

### 2.1 分布式假设（Distributional Hypothesis）

> "You shall know a word by the company it keeps." — J.R. Firth, 1957

核心思想：**一个词的含义由其上下文决定**。

```
"我养了一只___，它会抓老鼠" → 猫
"我养了一只___，它会看门" → 狗
```

如果两个词经常出现在相似的上下文中，它们的语义就相似。

### 2.2 Skip-gram 模型

**目标**：给定中心词，预测其上下文词。

#### 模型结构

```
输入层        隐藏层(无激活)      输出层(softmax)
[V×1]    →    [d×V] × [V×1]   →   [V×d] × [d×1]   →   [V×1]
one-hot        W (输入矩阵)         W' (输出矩阵)      概率分布
```

#### 数学形式

给定中心词 $w_c$，预测上下文词 $w_o$ 的概率：

$$P(w_o | w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

其中：
- $\mathbf{v}_{w_c} \in \mathbb{R}^d$：中心词 $w_c$ 的输入向量（从矩阵 W 中取出）
- $\mathbf{u}_{w_o} \in \mathbb{R}^d$：上下文词 $w_o$ 的输出向量（从矩阵 W' 中取出）

#### 训练目标

最大化整个语料库的对数似然：

$$\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

其中：
- $T$：语料库总词数
- $c$：上下文窗口大小

#### 梯度推导

对于单个训练样本 $(w_c, w_o)$，损失函数：

$$\mathcal{L} = -\log P(w_o | w_c) = -\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c} + \log \sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})$$

对 $\mathbf{v}_{w_c}$ 求梯度：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_c}} = -\mathbf{u}_{w_o} + \sum_{w \in \mathcal{V}} P(w|w_c) \cdot \mathbf{u}_w$$

直觉理解：
- $-\mathbf{u}_{w_o}$：拉近中心词和真实上下文词
- $\sum P(w|w_c) \cdot \mathbf{u}_w$：推远中心词和所有词的期望

### 2.3 CBOW 模型

**目标**：给定上下文词，预测中心词。

与 Skip-gram 相反：
- Skip-gram：$P(context | center)$
- CBOW：$P(center | context)$

CBOW 将上下文词向量取平均：

$$\mathbf{h} = \frac{1}{2c} \sum_{-c \leq j \leq c, j \neq 0} \mathbf{v}_{w_{t+j}}$$

然后预测中心词：

$$P(w_c | context) = \frac{\exp(\mathbf{u}_{w_c}^\top \mathbf{h})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{h})}$$

### 2.4 计算优化：负采样（Negative Sampling）

#### 问题

Softmax 分母需要遍历整个词汇表（通常 10万+），计算量巨大：

$$\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})$$

#### 负采样的思想

将多分类问题转化为二分类问题：
- 正样本：真实的 (中心词, 上下文词) 对
- 负样本：随机采样的 (中心词, 随机词) 对

#### 新的目标函数

$$\mathcal{L} = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_c})]$$

其中：
- $\sigma(x) = \frac{1}{1+e^{-x}}$：sigmoid 函数
- $k$：负样本数量（通常 5-20）
- $P_n(w)$：负采样分布，通常用 $P_n(w) \propto f(w)^{0.75}$（$f(w)$ 是词频）

#### 为什么用 $f(w)^{0.75}$？

- 如果用均匀分布：高频词采样不足
- 如果用词频分布：高频词采样过多
- $0.75$ 次幂是一个折中，让低频词有更多机会被采样

### 2.5 词向量的神奇性质

训练完成后，词向量展现出语义算术性质：

$$\vec{v}_{king} - \vec{v}_{man} + \vec{v}_{woman} \approx \vec{v}_{queen}$$

这是因为：
- $\vec{v}_{king} - \vec{v}_{man}$ 捕捉了"皇室"的语义
- 加上 $\vec{v}_{woman}$ 得到"女性皇室成员"

---

## 三、从词向量到句向量

### 3.1 简单方法：词向量聚合

**平均池化**：

$$\mathbf{s} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{v}_{w_i}$$

**TF-IDF 加权平均**：

$$\mathbf{s} = \frac{\sum_{i=1}^{n} \text{tfidf}(w_i) \cdot \mathbf{v}_{w_i}}{\sum_{i=1}^{n} \text{tfidf}(w_i)}$$

**问题**：丢失词序信息，"狗咬人" 和 "人咬狗" 向量相同。

### 3.2 基于 Transformer 的句向量

#### BERT 的句向量

BERT 输出每个 token 的向量，如何得到句向量？

1. **[CLS] token**：取第一个特殊 token 的向量
   - 问题：[CLS] 是为 NSP 任务设计的，不一定适合相似度计算

2. **Mean Pooling**：所有 token 向量取平均
   - 通常效果更好

3. **Max Pooling**：每个维度取最大值

#### BERT 句向量的问题

直接用 BERT 做句子相似度效果不好，原因：
- BERT 的预训练目标（MLM、NSP）不是为相似度设计的
- 句向量分布各向异性（anisotropic），集中在一个狭窄的锥形区域

---

## 四、Sentence-BERT：专为相似度设计

### 4.1 动机

Cross-encoder 方式计算句子相似度：

```
[CLS] 句子A [SEP] 句子B [SEP] → BERT → 相似度分数
```

问题：n 个句子两两比较需要 $O(n^2)$ 次 BERT 前向传播，10000 个句子需要约 65 小时！

### 4.2 Sentence-BERT 架构

使用 Siamese 网络结构：

```
句子A → BERT → Pooling → 向量A
                              ↘
                               相似度
                              ↗
句子B → BERT → Pooling → 向量B
```

两个 BERT 共享参数，句子独立编码后计算相似度。

### 4.3 训练目标

#### 分类目标（用于 NLI 数据）

将句子对的关系分为：蕴含、矛盾、中立

$$\mathbf{o} = \text{softmax}(W_t [\mathbf{u}; \mathbf{v}; |\mathbf{u} - \mathbf{v}|])$$

其中 $[\cdot;\cdot;\cdot]$ 表示拼接。

#### 回归目标（用于 STS 数据）

直接预测相似度分数：

$$\hat{y} = \cos(\mathbf{u}, \mathbf{v})$$

损失函数：MSE

$$\mathcal{L} = (y - \hat{y})^2$$

#### 对比学习目标

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{u}, \mathbf{v}^+)/\tau)}{\sum_{j} \exp(\text{sim}(\mathbf{u}, \mathbf{v}_j)/\tau)}$$

---

## 五、现代 Embedding 模型

### 5.1 对比学习框架

现代 Embedding 模型（BGE、M3E、E5 等）主要基于对比学习：

#### InfoNCE Loss

$$\mathcal{L}_q = -\log \frac{\exp(\text{sim}(q, d^+)/\tau)}{\exp(\text{sim}(q, d^+)/\tau) + \sum_{d^- \in \mathcal{N}} \exp(\text{sim}(q, d^-)/\tau)}$$

其中：
- $q$：query 向量
- $d^+$：正样本文档向量
- $\mathcal{N}$：负样本集合
- $\tau$：温度参数（通常 0.01-0.1）

#### 温度参数的作用

$$\text{softmax}(z_i/\tau) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

- $\tau \rightarrow 0$：分布趋向 one-hot，只关注最相似的
- $\tau \rightarrow \infty$：分布趋向均匀，所有样本权重相同
- 较小的 $\tau$ 让模型更关注 hard negatives

### 5.2 负样本策略

#### In-batch Negatives

同一 batch 内其他样本作为负样本：

```
Batch:
  (q1, d1+)  → 负样本: d2+, d3+, d4+
  (q2, d2+)  → 负样本: d1+, d3+, d4+
  (q3, d3+)  → 负样本: d1+, d2+, d4+
  (q4, d4+)  → 负样本: d1+, d2+, d3+
```

优点：高效，不需要额外采样
缺点：负样本可能太简单

#### Hard Negatives

精心挑选的难负样本：

1. **BM25 Hard Negatives**：用 BM25 检索的 top-k 非相关文档
2. **Cross-batch Negatives**：其他 query 的正样本
3. **模型自挖掘**：用当前模型检索的 false positives

### 5.3 训练数据

| 数据类型 | 示例 | 特点 |
|----------|------|------|
| 搜索日志 | (query, 点击文档) | 大规模，有噪声 |
| 问答对 | (问题, 答案) | 高质量 |
| NLI | (前提, 蕴含假设) | 细粒度语义 |
| 平行语料 | (英文句, 中文句) | 跨语言对齐 |

### 5.4 多阶段训练

现代模型通常采用多阶段训练：

```
阶段1：大规模弱监督预训练
  - 数据：网页标题-正文、锚文本-目标页面
  - 目标：学习基础语义表示

阶段2：高质量数据微调
  - 数据：人工标注的相似度数据
  - 目标：提升相似度判断能力

阶段3：Hard Negative 微调
  - 数据：挖掘的 hard negatives
  - 目标：提升区分能力
```

---

## 六、Embedding 质量评估

### 6.1 内在评估

**词类比任务**：

$$\text{accuracy} = \frac{\#(\arg\max_w \cos(\mathbf{v}_w, \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c) = d)}{N}$$

例如：$a$=man, $b$=king, $c$=woman, $d$=queen

**句子相似度任务（STS）**：

计算模型预测相似度与人工标注的 Spearman 相关系数。

### 6.2 外在评估

在下游任务上评估：
- 检索任务：Recall@k, MRR, NDCG
- 分类任务：Accuracy, F1
- 聚类任务：NMI, ARI

### 6.3 各向异性问题

**问题**：Embedding 向量分布在一个狭窄的锥形区域，导致任意两个向量的余弦相似度都很高。

**检测方法**：

$$\text{Avg Cosine} = \frac{2}{n(n-1)} \sum_{i<j} \cos(\mathbf{v}_i, \mathbf{v}_j)$$

如果平均余弦相似度过高（如 > 0.8），说明存在各向异性问题。

**解决方法**：
1. 白化（Whitening）：$\mathbf{v}' = (\mathbf{v} - \mu) \Sigma^{-1/2}$
2. 对比学习训练
3. 后处理归一化

---

## 七、实践建议

### 7.1 模型选择

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 中文通用 | BGE-large-zh | 效果好，社区活跃 |
| 英文通用 | E5-large-v2 | MTEB 榜单领先 |
| 多语言 | BGE-M3 | 支持 100+ 语言 |
| 低延迟 | BGE-small | 速度快，效果可接受 |
| OpenAI 生态 | text-embedding-3-large | API 调用简单 |

### 7.2 向量维度选择

- 384 维：轻量级，适合资源受限场景
- 768 维：平衡选择
- 1024/1536 维：追求最佳效果

### 7.3 归一化

大多数模型输出已归一化，但建议显式归一化：

```python
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
```

归一化后，点积 = 余弦相似度，计算更快。
