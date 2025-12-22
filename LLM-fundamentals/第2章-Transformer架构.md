# 第2章 Transformer 架构

Transformer 是现代大语言模型（GPT、BERT、LLaMA 等）的核心架构，理解它是理解 LLM 的关键。

## 2.1 为什么需要 Transformer？

### 2.1.1 Transformer 之前：RNN/LSTM

在 Transformer（2017年）出现之前，NLP 领域主要使用 **RNN（循环神经网络）** 和它的变体 **LSTM**。

**RNN 处理序列的方式**：
```
输入序列：我 → 喜欢 → 吃 → 苹果

处理过程（必须按顺序）：
我 → [RNN] → h1
         ↓
喜欢 → [RNN] → h2（依赖 h1）
          ↓
吃 → [RNN] → h3（依赖 h2）
        ↓
苹果 → [RNN] → h4（依赖 h3）
```

每一步的计算都依赖上一步的结果，必须**串行计算**。

### 2.1.2 RNN 的两大致命问题

**问题1：无法并行计算**

```
RNN：h1 → h2 → h3 → h4（必须按顺序）
     ↓
GPU 的并行能力完全被浪费！
训练一个大模型可能需要几个月
```

**问题2：长距离依赖问题**

```
句子："那个昨天在公园里遛狗的男人是我的邻居"
       ↑                              ↑
      主语                           谓语

"男人"和"邻居"相隔很远（中间隔了很多词）
信息在 RNN 中传递时会逐渐衰减
到最后，模型很难记住前面的"男人"
```

虽然 LSTM 通过"门机制"缓解了这个问题，但并没有根本解决。

### 2.1.3 Transformer 的解决方案

**核心思想**：用**注意力机制**完全替代循环结构！

```
Transformer：
- 每个词可以直接关注任意其他词（不管距离多远）
- 所有位置可以并行计算
- 训练速度大幅提升
```

**论文标题**：《Attention Is All You Need》（2017，Google）
- 意思是：只需要注意力机制就够了，不需要 RNN！

## 2.2 注意力机制（Attention）

注意力机制是 Transformer 的核心，必须彻底理解。

### 2.2.1 什么是注意力？

**人类的注意力**：
- 看一张图片时，不会均匀地看每个像素
- 而是聚焦在重要的部分（如人脸、文字）

**机器的注意力**：
- 处理一个词时，不是平等对待所有词
- 而是给不同的词分配不同的**权重**（注意力）

```
句子："我 喜欢 吃 苹果"

当模型处理"吃"这个词时：
- "苹果"很重要（吃什么？）→ 权重 0.5
- "喜欢"比较重要 → 权重 0.3
- "我"不太相关 → 权重 0.2

最终"吃"的表示 = 0.5×苹果 + 0.3×喜欢 + 0.2×我
```

### 2.2.2 Query、Key、Value（QKV）

注意力机制有三个核心概念，这是最重要的部分！

| 概念 | 简写 | 含义 | 类比 |
|------|------|------|------|
| Query | Q | 我想找什么信息 | 搜索关键词 |
| Key | K | 每个词的"标签" | 网页标题 |
| Value | V | 每个词的实际内容 | 网页内容 |

#### Q、K、V 是怎么来的？（重要！）

**Q、K、V 都来自同一个输入，通过不同的线性变换得到：**

```
输入：X = 词的 Embedding 矩阵

Q = X × W_Q   （W_Q 是可学习的参数矩阵）
K = X × W_K   （W_K 是可学习的参数矩阵）
V = X × W_V   （W_V 是可学习的参数矩阵）

所以 Q、K、V 本质上都是 Embedding 的线性变换！
```

**为什么要分成三个不同的矩阵？**

```
如果直接用 X × X^T，所有词的"查询"和"被查询"用的是同一个表示
这样不够灵活

分成 Q、K、V 后，模型可以学习：
- W_Q：这个词"想要查什么信息"
- W_K：这个词"能被什么查到"（像标签/索引）
- W_V：这个词"实际要传递什么信息"

三个矩阵各司其职，大大增加了模型的表达能力
```

**V 是不是就是 Embedding？**

```
不完全是！V = X × W_V，是 Embedding 经过线性变换后的结果

为什么不直接用 Embedding？
- 线性变换让模型可以学习"应该传递什么信息"
- 原始 Embedding 包含所有信息，但不是所有信息都需要传递
- W_V 让模型学会筛选和转换信息
```

**Q × K^T 得到的是注意力吗？**

```
Q × K^T = Attention Score（注意力分数）

这个分数表示每个词对其他词的"关注程度"
但还不是最终的注意力权重！

还需要两步：
1. 除以 √d_k（缩放，防止数值太大）
2. Softmax（归一化成概率，所有权重加起来=1）

Attention Weights = Softmax(Q × K^T / √d_k)  ← 这才是注意力权重
```

**完整流程图：**

```
输入 Embedding (X)
        │
        ├──× W_Q ──→ Q (Query: 我想查什么)
        │
        ├──× W_K ──→ K (Key: 我的标签是什么)
        │
        └──× W_V ──→ V (Value: 我要传递的信息)


Q × K^T  →  分数矩阵（谁和谁相关，还不是注意力）
    │
    ÷ √d_k  →  缩放（防止数值太大导致 Softmax 极端化）
    │
    Softmax  →  注意力权重（概率分布，每行加起来=1）
    │
    × V  →  输出（用注意力权重对 Value 加权求和）
```

**类比：搜索引擎**

```
你在 Google 搜索"如何学习机器学习"

Query = "如何学习机器学习"（你的搜索词）
Key = 每个网页的标题/关键词
Value = 每个网页的实际内容

搜索过程：
1. 用 Query 和每个网页的 Key 比较
2. 计算相关度分数
3. 相关度高的网页排在前面
4. 返回这些网页的 Value（内容）
```

**类比：图书馆找书**

```
Query = "我想找关于深度学习的书"
Key = 每本书的书名、标签
Value = 书的实际内容

过程：
1. 图书管理员用你的 Query 去匹配每本书的 Key
2. 找到相关的书（Key 匹配度高）
3. 把这些书的内容（Value）给你
```

### 2.2.3 注意力计算公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**别被公式吓到！我们一步步拆解：**

#### 步骤1：计算相关度分数（Q × K^T）

```
Q = Query 向量（我想找什么）
K = 所有词的 Key 向量

Q × K^T = 点积，得到相关度分数

例子：
Q = [1, 0, 1]（"吃"的 Query）
K1 = [0, 1, 0]（"我"的 Key）
K2 = [1, 0, 0]（"喜欢"的 Key）
K3 = [1, 0, 1]（"苹果"的 Key）

Q · K1 = 1×0 + 0×1 + 1×0 = 0（不相关）
Q · K2 = 1×1 + 0×0 + 1×0 = 1（有点相关）
Q · K3 = 1×1 + 0×0 + 1×1 = 2（很相关！）

分数：[0, 1, 2]
```

**点积越大，说明两个向量越相似，相关度越高。**

#### 步骤2：缩放（除以 √d_k）

```
d_k = Key 的维度（如 64）
√d_k = 8

为什么要除以 √d_k？
- 点积结果可能很大（特别是维度高时）
- 太大的值经过 Softmax 后会变得极端（接近 0 或 1）
- 导致梯度消失，训练困难
- 除以 √d_k 让数值保持在合理范围

缩放后：[0, 1, 2] / 8 = [0, 0.125, 0.25]
```

#### 步骤3：Softmax 归一化

```
Softmax 把分数转成概率（所有值加起来 = 1）

公式：softmax(x_i) = exp(x_i) / Σexp(x_j)

例子：
输入：[0, 0.125, 0.25]
输出：[0.29, 0.33, 0.38]（加起来 = 1）

现在我们有了注意力权重！
- "我"的权重：0.29
- "喜欢"的权重：0.33
- "苹果"的权重：0.38（最高，因为和"吃"最相关）
```

#### 步骤4：加权求和（× V）

```
用注意力权重对 Value 加权求和

V1 = "我"的 Value
V2 = "喜欢"的 Value
V3 = "苹果"的 Value

输出 = 0.29 × V1 + 0.33 × V2 + 0.38 × V3

这个输出就是"吃"这个词经过注意力机制后的新表示
它融合了整个句子的信息，特别是和它相关的词
```

### 2.2.4 完整计算示例

```
句子："我 喜欢 苹果"（3个词）
词向量维度：4

输入矩阵 X（3×4）：
X = [[0.1, 0.2, 0.3, 0.4],   # "我"
     [0.5, 0.6, 0.7, 0.8],   # "喜欢"
     [0.2, 0.3, 0.4, 0.5]]   # "苹果"

权重矩阵（可学习参数）：
W_Q, W_K, W_V 都是 4×4 的矩阵

步骤1：生成 Q, K, V
Q = X × W_Q  # 3×4
K = X × W_K  # 3×4
V = X × W_V  # 3×4

步骤2：计算注意力分数
scores = Q × K^T  # 3×3 矩阵
# 每个位置 (i,j) 表示第 i 个词对第 j 个词的关注度

步骤3：缩放 + Softmax
attention_weights = softmax(scores / √4)  # 3×3

步骤4：加权求和
output = attention_weights × V  # 3×4
# 每个词都得到了融合全局信息的新表示
```

### 2.2.5 注意力的直观理解

```
注意力矩阵可视化（3×3）：

        我    喜欢   苹果
我     [0.6   0.2   0.2]   ← "我"主要关注自己
喜欢   [0.2   0.4   0.4]   ← "喜欢"关注自己和"苹果"
苹果   [0.1   0.3   0.6]   ← "苹果"主要关注自己

每一行是一个词的注意力分布，加起来 = 1
```

## 2.3 自注意力（Self-Attention）

### 2.3.1 为什么需要自注意力？

**核心问题：一个词的含义往往取决于上下文**

```
例子："苹果"这个词

"我吃了一个苹果" → 水果
"我买了一个苹果手机" → 品牌

单独看"苹果"无法确定含义
必须看上下文才能理解正确含义

自注意力的作用：
让"苹果"能关注到"吃"或"手机"
从而获得正确的语义表示
```

**自注意力的设计目标**：让每个词都能"看到"整个句子，根据上下文动态调整自己的表示。

### 2.3.2 为什么用 Q、K、V 三个矩阵？

**设计思想：模拟"查询-匹配-获取"的过程**

```
方案1：只用一个矩阵（X × X^T）
- 每个词用同一个表示去查询和被查询
- 太死板，表达能力弱
- 无法区分"我想问什么"和"我能回答什么"

方案2：用三个不同的矩阵（Transformer 的选择）
- Q（Query）：学习"我想问什么问题"
- K（Key）：学习"我能回答什么问题"  
- V（Value）：学习"我的答案是什么"

好处：同一个词可以扮演不同的"角色"
- 作为查询者时，用 Q
- 作为被查询者时，用 K
- 作为信息提供者时，用 V

大大增加了模型的灵活性和表达能力
```

### 2.3.3 为什么 K 和 V 要分开？

```
K 和 V 的职责不同：

K（Key）：用于"匹配"，决定注意力权重
- 像书的标题/标签，用于被搜索
- 决定"应该关注谁"

V（Value）：用于"输出"，是实际传递的信息
- 像书的内容，是真正要获取的东西
- 决定"获取什么信息"

例子：
一本书的标题是"Python入门"（K）→ 用于匹配搜索
但内容包含具体代码和解释（V）→ 是你真正要的信息

如果 K=V，那匹配用的信息和传递的信息就一样了
分开后，模型可以学习：
- 用什么特征来匹配（K）
- 匹配后传递什么信息（V）
```

### 2.3.4 Self-Attention vs Cross-Attention

**Self-Attention（自注意力）**：
- Q、K、V 都来自**同一个序列**
- 序列内部的词相互关注

```
场景：理解句子 "我 喜欢 苹果"

输入 X = "我 喜欢 苹果" 的 Embedding

Q = X × W_Q  →  来自 "我 喜欢 苹果"
K = X × W_K  →  来自 "我 喜欢 苹果"（同一个序列）
V = X × W_V  →  来自 "我 喜欢 苹果"

每个词用自己的 Q 去查句子内所有词的 K
结果：理解词与词之间的关系
```

**Cross-Attention（交叉注意力）**：
- Q 来自一个序列，K/V 来自另一个序列
- 两个序列之间交互

```
场景：机器翻译 "I love you" → "我 爱 你"

Encoder 处理源语言：
"I love you" → encoder_output

Decoder 生成目标语言时：
Q = 来自 Decoder（"我 爱 你"）
K = 来自 Encoder（"I love you"）← 不同序列！
V = 来自 Encoder（"I love you"）

"我" 的 Q 去查 "I, love, you" 的 K → 发现和 "I" 最相关
"爱" 的 Q 去查 "I, love, you" 的 K → 发现和 "love" 最相关

结果：实现源语言和目标语言的"对齐"
```

**图示对比**：

```
Self-Attention（自己看自己）：

    "我 喜欢 苹果"
         │
    ┌────┴────┐
    ↓         ↓
    Q    K, V 都来自同一个序列
    │         │
    └────┬────┘
         ↓
    句子内部词相互关注


Cross-Attention（A 看 B）：

    "我 爱 你"              "I love you"
    (Decoder)               (Encoder)
         │                       │
         ↓                       ↓
         Q ──────────────→ K, V
                  │
                  ↓
         中文词去关注英文词
```

**总结对比**：

| 类型 | Q 来自 | K/V 来自 | 作用 | 使用场景 |
|------|--------|----------|------|---------|
| Self-Attention | 序列 A | 序列 A | 序列内部建模 | BERT, GPT |
| Cross-Attention | 序列 A | 序列 B | 两个序列交互 | 翻译, Encoder-Decoder |

### 2.3.5 为什么叫"自"注意力？

因为是**自己关注自己**——序列内部的词相互关注。

```
"我 喜欢 苹果" 的自注意力：

我 ──→ 关注 [我, 喜欢, 苹果] ──→ 新的"我"表示
喜欢 ──→ 关注 [我, 喜欢, 苹果] ──→ 新的"喜欢"表示
苹果 ──→ 关注 [我, 喜欢, 苹果] ──→ 新的"苹果"表示
```

### 2.3.6 自注意力的优势

**1. 捕捉长距离依赖**

```
RNN：信息需要一步步传递，距离越远越难
     我 → 喜欢 → 在 → 公园 → 里 → 吃 → 苹果
     ↑_________________________________↑
     信息传递了 6 步，可能衰减

自注意力：任意两个词可以直接交互
     我 ←──────────────────────→ 苹果
     只需要 1 步！
```

**2. 并行计算**

```
RNN：必须串行
     h1 → h2 → h3 → h4

自注意力：可以并行
     所有词同时计算注意力
     GPU 利用率大幅提升
```

**3. 可解释性**

```
注意力权重可以可视化
可以看到模型在关注什么
有助于理解和调试模型
```

### 2.3.7 Q、K、V 的生成（代码视角）

Q、K、V 通过**三个不同的线性变换**得到：

```python
# 输入：词向量矩阵 X，形状 (seq_len, d_model)
# 例如：(3, 512) 表示 3 个词，每个词 512 维

# 三个权重矩阵（可学习参数）
W_Q = 随机初始化，形状 (d_model, d_k)  # (512, 64)
W_K = 随机初始化，形状 (d_model, d_k)  # (512, 64)
W_V = 随机初始化，形状 (d_model, d_v)  # (512, 64)

# 生成 Q, K, V
Q = X @ W_Q  # (3, 64)
K = X @ W_K  # (3, 64)
V = X @ W_V  # (3, 64)
```

**为什么要用不同的矩阵？**

- 如果 Q=K=V=X，那所有词的查询和键都一样，没有区分度
- 不同的矩阵让模型学习：
  - W_Q：这个词想查询什么信息
  - W_K：这个词能提供什么信息
  - W_V：这个词的实际内容是什么

### 2.3.8 自注意力的代码实现

```python
import torch
import torch.nn as nn
import math

def self_attention(X, W_Q, W_K, W_V):
    """
    X: 输入，形状 (batch_size, seq_len, d_model)
    W_Q, W_K, W_V: 权重矩阵
    """
    # 1. 生成 Q, K, V
    Q = X @ W_Q  # (batch, seq_len, d_k)
    K = X @ W_K
    V = X @ W_V
    
    # 2. 计算注意力分数
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)
    
    # 3. 缩放
    scores = scores / math.sqrt(d_k)
    
    # 4. Softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 5. 加权求和
    output = attention_weights @ V  # (batch, seq_len, d_v)
    
    return output, attention_weights
```

## 2.4 多头注意力（Multi-Head Attention）

### 2.4.1 为什么需要多头？

单个注意力只能学习一种关注模式，但语言中的关系是多样的：

```
句子："小明昨天在北京吃了一碗热腾腾的牛肉面"

不同的关注角度：
- 语法关系："小明" → "吃了"（主谓关系）
- 时间关系："昨天" → "吃了"（时间状语）
- 地点关系："北京" → "吃了"（地点状语）
- 修饰关系："热腾腾" → "牛肉面"（形容词修饰名词）
- 动宾关系："吃了" → "牛肉面"（动词和宾语）
```

**一个注意力头很难同时学会所有这些关系！**

**解决方案**：用多个注意力头，每个头学习不同的关注模式。

### 2.4.2 多头注意力的原理

```
多头注意力 = 多个独立的注意力并行计算，然后拼接

输入 X
   ↓
┌──────────────────────────────────────┐
│  头1: Q1,K1,V1 → Attention → output1 │
│  头2: Q2,K2,V2 → Attention → output2 │
│  头3: Q3,K3,V3 → Attention → output3 │
│  ...                                  │
│  头h: Qh,Kh,Vh → Attention → outputh │
└──────────────────────────────────────┘
   ↓
拼接 [output1, output2, ..., outputh]
   ↓
线性变换 W_O
   ↓
最终输出
```

### 2.4.3 维度计算

```
假设：
- d_model = 512（模型维度）
- h = 8（头数）
- d_k = d_v = d_model / h = 64（每个头的维度）

输入 X: (seq_len, 512)

每个头：
- Q_i, K_i, V_i: (seq_len, 64)
- output_i: (seq_len, 64)

拼接后：(seq_len, 64×8) = (seq_len, 512)

最终输出：(seq_len, 512)  # 和输入维度相同！
```

**关键点**：多头注意力的输入输出维度相同，可以堆叠多层。

### 2.4.4 多头注意力公式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

其中每个头：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 2.4.5 多头注意力的代码实现

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 四个线性变换
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性变换
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 2. 拆分成多头
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # 4. 拼接多头
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终线性变换
        output = self.W_O(output)
        
        return output
```

### 2.4.6 多头的可视化理解

```
原论文中的实验发现，不同的头确实学到了不同的模式：

头1：可能学习语法依赖（主语-谓语）
头2：可能学习位置关系（相邻词）
头3：可能学习指代关系（代词-名词）
头4：可能学习修饰关系（形容词-名词）
...

这就是为什么多头比单头效果好！
```

## 2.5 掩码自注意力（Masked Self-Attention）

### 2.5.1 为什么需要掩码？

在**生成任务**（如 GPT）中，模型需要一个词一个词地生成：

```
生成过程：
输入：<BOS>
预测：我

输入：<BOS> 我
预测：喜欢

输入：<BOS> 我 喜欢
预测：苹果

输入：<BOS> 我 喜欢 苹果
预测：<EOS>
```

**问题**：在预测"喜欢"时，模型不应该看到后面的"苹果"！

如果模型能看到未来的词，那就是"作弊"了，训练时学不到真正的预测能力。

### 2.5.2 掩码的作用

掩码（Mask）用来**遮蔽未来的信息**：

```
句子："我 喜欢 苹果"

不使用掩码的注意力矩阵：
        我    喜欢   苹果
我     [0.6   0.2   0.2]   ← 可以看到所有词
喜欢   [0.2   0.4   0.4]   ← 可以看到所有词
苹果   [0.1   0.3   0.6]   ← 可以看到所有词

使用掩码后：
        我    喜欢   苹果
我     [1.0   0     0   ]   ← 只能看到"我"
喜欢   [0.4   0.6   0   ]   ← 只能看到"我"和"喜欢"
苹果   [0.1   0.3   0.6]   ← 可以看到所有词
```

### 2.5.3 掩码矩阵

掩码是一个**下三角矩阵**：

```python
# 序列长度为 4 的掩码矩阵
mask = [
    [1, 0, 0, 0],  # 位置0只能看位置0
    [1, 1, 0, 0],  # 位置1能看位置0,1
    [1, 1, 1, 0],  # 位置2能看位置0,1,2
    [1, 1, 1, 1],  # 位置3能看所有位置
]

# 1 表示可以看，0 表示不能看
```

### 2.5.4 掩码的实现

```python
def create_causal_mask(seq_len):
    """创建因果掩码（下三角矩阵）"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

# 在注意力计算中使用掩码
def masked_attention(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    
    # 把掩码为 0 的位置设为负无穷
    # softmax 后这些位置的权重就变成 0
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = attention_weights @ V
    return output
```

### 2.5.5 Encoder vs Decoder 的注意力

| 类型 | 使用掩码？ | 原因 |
|------|-----------|------|
| Encoder 自注意力 | ❌ 不使用 | 编码时可以看到整个输入 |
| Decoder 自注意力 | ✅ 使用 | 生成时不能看到未来 |
| Cross-Attention | ❌ 不使用 | Decoder 可以看到整个 Encoder 输出 |

## 2.6 位置编码（Positional Encoding）

### 2.6.1 为什么需要位置编码？

自注意力有个问题：**它不知道词的顺序！**

```
"我 喜欢 你" 和 "你 喜欢 我"

对于自注意力来说，这两个句子的计算过程完全一样
因为注意力只看词之间的关系，不看位置
但这两个句子的意思完全不同！
```

**解决方案**：给每个位置加上一个**位置编码**，让模型知道词的顺序。

### 2.6.2 位置编码的方式

**方式1：可学习的位置编码（BERT、GPT 使用）**

$\mathbf{E}_{pos} = \text{Embedding}(pos), \quad pos \in \{0, 1, ..., L_{max}-1\}$

其中 $L_{max}$ 是最大序列长度，位置编码矩阵 $\mathbf{P} \in \mathbb{R}^{L_{max} \times d}$ 是可学习参数。

**方式2：正弦余弦位置编码（原始 Transformer 使用）**

$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

其中：
- $pos$ = 位置索引（0, 1, 2, ...）
- $i$ = 维度索引（0, 1, ..., d/2-1）
- $d_{model}$ = 模型维度

### 2.6.3 正弦余弦编码的特点

```
为什么用 sin/cos？

1. 可以处理任意长度的序列（不像可学习编码有最大长度限制）

2. 相对位置可以通过线性变换得到
   PE(pos+k) 可以表示为 PE(pos) 的线性函数
   这让模型更容易学习相对位置关系

3. 不同维度有不同的周期
   低维度：周期短，捕捉局部位置
   高维度：周期长，捕捉全局位置
```

**相对位置的线性表示**：

正弦余弦编码的一个重要性质是，位置 $pos + k$ 的编码可以表示为位置 $pos$ 编码的线性变换：

$\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}$

其中 $\omega_i = \frac{1}{10000^{2i/d}}$。这意味着模型可以通过学习这个线性变换来捕捉相对位置关系。

### 2.6.4 位置编码的代码实现

```python
import torch
import math

def sinusoidal_position_encoding(seq_len, d_model):
    """正弦余弦位置编码"""
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
    
    return pe

# 使用位置编码
# 输入 = 词向量 + 位置编码
input_with_position = word_embedding + position_encoding
```

### 2.6.5 位置编码可视化

**注意**：sin/cos 是计算方法，实际存储的是具体数值！

```
位置编码的计算规则：
- 偶数维度（0, 2, 4, ...）：用 sin 函数计算
- 奇数维度（1, 3, 5, ...）：用 cos 函数计算

实际的位置编码矩阵（具体数值，假设 seq_len=50, d_model=128）：

           维度0    维度1    维度2    维度3   ...  维度127
          (sin)    (cos)    (sin)    (cos)        (cos)
位置0    [0.000   1.000    0.000   1.000   ...   1.000 ]
位置1    [0.841   0.540    0.010   0.999   ...   1.000 ]
位置2    [0.909  -0.416    0.020   0.999   ...   1.000 ]
位置3    [0.141  -0.990    0.030   0.999   ...   1.000 ]
...
位置49   [0.954   0.297    0.479   0.878   ...   1.000 ]

观察规律：
- 每一行是一个位置的编码向量（128维）
- 低维度（左边）：数值变化快，周期短 → 捕捉局部位置差异
- 高维度（右边）：数值变化慢，周期长 → 捕捉全局位置信息
- 不同位置的编码向量是不同的，模型可以区分位置
```

## 2.7 前馈神经网络（Feed-Forward Network）

### 2.7.1 FFN 的作用

Transformer 中，每个注意力层后面都跟着一个**前馈神经网络（FFN）**：

```
输入
  ↓
自注意力层
  ↓
FFN ← 这个！
  ↓
输出
```

**FFN 的作用**：
- 对注意力的输出做非线性变换
- 增加模型的表达能力
- 每个位置独立处理（不像注意力那样位置之间交互）

### 2.7.2 FFN 的结构

```
FFN = 两层全连接 + 激活函数

FFN(x) = ReLU(x × W1 + b1) × W2 + b2

或者用 GELU 激活（更常用）：
FFN(x) = GELU(x × W1 + b1) × W2 + b2
```

### 2.7.3 FFN 的维度变化

```
输入：(seq_len, d_model)，如 (100, 512)
     ↓
第一层：扩展到 4 倍
     (seq_len, d_ff)，如 (100, 2048)
     ↓
激活函数（ReLU/GELU）
     ↓
第二层：压缩回原维度
     (seq_len, d_model)，如 (100, 512)

通常 d_ff = 4 × d_model
```

### 2.7.4 FFN 的代码实现

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

## 2.8 残差连接和层归一化

### 2.8.1 残差连接（Residual Connection）

**问题**：深层网络容易出现梯度消失/爆炸，难以训练。

**解决方案**：残差连接，让梯度可以直接流过。

```
残差连接：output = x + SubLayer(x)

         ┌─────────────────┐
         │                 │
x ───────┼──→ SubLayer ────┼──→ x + SubLayer(x)
         │                 │
         └─────────────────┘
              跳跃连接
```

**好处**：
- 梯度可以直接通过跳跃连接传播
- 即使 SubLayer 学不好，至少还有原始输入
- 让深层网络更容易训练

### 2.8.2 层归一化（Layer Normalization）

**问题**：神经网络中间层的输出分布可能变化很大，影响训练稳定性。

**解决方案**：层归一化，把每一层的输出标准化。

**Layer Norm 公式**：

$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

其中：
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$（均值）
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$（方差）
- $\gamma, \beta \in \mathbb{R}^d$ 是可学习的缩放和偏移参数
- $\epsilon$ 是防止除零的小常数（如 $10^{-6}$）
- $\odot$ 表示逐元素乘法

**Layer Norm vs Batch Norm**：

| 特性 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 归一化维度 | 跨 batch | 跨特征 |
| 依赖 batch size | 是 | 否 |
| 适用场景 | CNN | Transformer, RNN |
| 训练/推理一致性 | 需要 running mean | 一致 |

### 2.8.3 Pre-LN vs Post-LN

**Post-LN（原始 Transformer）**：
```
x → SubLayer → Add → LayerNorm → output
```

**Pre-LN（现代常用）**：
```
x → LayerNorm → SubLayer → Add → output
```

Pre-LN 训练更稳定，现在更常用。

### 2.8.4 完整的 Transformer 层

```
一个完整的 Transformer 层（Pre-LN 版本）：

输入 x
    ↓
LayerNorm
    ↓
Multi-Head Attention
    ↓
残差连接（+ x）
    ↓
LayerNorm
    ↓
Feed-Forward Network
    ↓
残差连接
    ↓
输出
```

代码实现：

```python
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # FFN + 残差
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x
```

## 2.9 Encoder-Decoder 结构

### 2.9.1 原始 Transformer 的结构

原始 Transformer 是为**机器翻译**设计的，包含 Encoder 和 Decoder：

```
源语言 "I love you"
        ↓
    [Encoder]
        ↓
    编码表示
        ↓
    [Decoder] ← 目标语言 "我 爱 你"（训练时）
        ↓
    "我 爱 你"（输出）
```

### 2.9.2 Encoder 结构

```
Encoder = N 个相同的层堆叠（通常 N=6 或 12）

每一层：
输入
  ↓
Multi-Head Self-Attention（无掩码）
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
输出
```

**特点**：
- 可以看到整个输入序列
- 不使用掩码
- 输出是输入的编码表示

### 2.9.3 Decoder 结构

```
Decoder = N 个相同的层堆叠

每一层：
输入（已生成的词）
  ↓
Masked Multi-Head Self-Attention（有掩码！）
  ↓
Add & Norm
  ↓
Multi-Head Cross-Attention（Q来自Decoder，K/V来自Encoder）
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
输出
```

**特点**：
- 自注意力使用掩码（不能看未来）
- 有 Cross-Attention 连接 Encoder 的输出
- 自回归生成（一个词一个词生成）

### 2.9.4 三种 Transformer 变体

| 架构 | 代表模型 | 特点 | 适用任务 |
|------|---------|------|---------|
| Encoder-only | BERT | 双向编码，理解任务 | 分类、NER、问答 |
| Decoder-only | GPT | 单向生成，自回归 | 文本生成、对话 |
| Encoder-Decoder | T5, BART | 完整结构 | 翻译、摘要 |

```
Encoder-only (BERT)：
输入 → [Encoder] → 编码表示 → 分类/标注

Decoder-only (GPT)：
输入 → [Decoder] → 下一个词 → 下一个词 → ...

Encoder-Decoder (T5)：
输入 → [Encoder] → [Decoder] → 输出序列
```

## 2.10 完整的 Transformer 架构图

```
                    Transformer 架构
                    
输入序列                                    输出序列
   ↓                                           ↑
词嵌入 + 位置编码                          线性层 + Softmax
   ↓                                           ↑
┌─────────────────┐                    ┌─────────────────┐
│    Encoder      │                    │    Decoder      │
│  ┌───────────┐  │                    │  ┌───────────┐  │
│  │Self-Attn  │  │                    │  │Masked     │  │
│  │(无掩码)   │  │                    │  │Self-Attn  │  │
│  └───────────┘  │                    │  └───────────┘  │
│       ↓         │                    │       ↓         │
│  ┌───────────┐  │    Encoder输出     │  ┌───────────┐  │
│  │Add & Norm │  │ ──────────────────→│  │Cross-Attn │  │
│  └───────────┘  │                    │  └───────────┘  │
│       ↓         │                    │       ↓         │
│  ┌───────────┐  │                    │  ┌───────────┐  │
│  │    FFN    │  │                    │  │Add & Norm │  │
│  └───────────┘  │                    │  └───────────┘  │
│       ↓         │                    │       ↓         │
│  ┌───────────┐  │                    │  ┌───────────┐  │
│  │Add & Norm │  │                    │  │    FFN    │  │
│  └───────────┘  │                    │  └───────────┘  │
│       ↓         │                    │       ↓         │
│    × N 层       │                    │  ┌───────────┐  │
└─────────────────┘                    │  │Add & Norm │  │
                                       │  └───────────┘  │
                                       │       ↓         │
                                       │    × N 层       │
                                       └─────────────────┘
```

## 2.11 Transformer 的计算复杂度（重要！）

理解 Transformer 的计算复杂度对于理解后续的优化技术非常重要。

### 2.11.1 自注意力的复杂度分析

```
假设：
- 序列长度：n
- 模型维度：d

自注意力的计算步骤：
1. Q = X × W_Q  →  O(n × d × d) = O(nd²)
2. K = X × W_K  →  O(nd²)
3. V = X × W_V  →  O(nd²)
4. Q × K^T      →  O(n × d × n) = O(n²d)  ← 关键！
5. Softmax     →  O(n²)
6. × V         →  O(n² × d) = O(n²d)

总复杂度：O(n²d + nd²)
```

**关键发现**：复杂度和序列长度 n 的**平方**成正比！

**详细复杂度分析**：

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 线性投影 (Q,K,V) | $O(nd^2)$ | $O(nd)$ |
| 注意力分数 $QK^T$ | $O(n^2d)$ | $O(n^2)$ |
| Softmax | $O(n^2)$ | $O(n^2)$ |
| 加权求和 | $O(n^2d)$ | $O(nd)$ |
| **总计** | $O(n^2d + nd^2)$ | $O(n^2 + nd)$ |

当 $n > d$ 时（长序列），$O(n^2d)$ 主导；当 $n < d$ 时，$O(nd^2)$ 主导。

### 2.11.2 为什么 O(n²) 是个大问题？

```
实际例子：

序列长度 n = 512（短文本）
注意力矩阵大小：512 × 512 = 262,144

序列长度 n = 2048（中等文本）
注意力矩阵大小：2048 × 2048 = 4,194,304（增长 16 倍！）

序列长度 n = 8192（长文本）
注意力矩阵大小：8192 × 8192 = 67,108,864（增长 256 倍！）

序列长度 n = 32768（GPT-4 级别）
注意力矩阵大小：32768 × 32768 = 1,073,741,824（10亿！）
```

**问题**：
- 内存占用随 n² 增长，GPU 显存很快不够用
- 计算时间也随 n² 增长，处理长文本非常慢
- 这就是为什么早期模型的上下文长度都很短（512、1024）

### 2.11.3 内存占用计算

```
注意力矩阵的内存占用：

假设使用 float32（4 字节）：
- n = 512:   512² × 4 = 1 MB
- n = 2048:  2048² × 4 = 16 MB
- n = 8192:  8192² × 4 = 256 MB
- n = 32768: 32768² × 4 = 4 GB

注意：这只是一个注意力头！
如果有 32 个头，12 层，还要乘以 32 × 12 = 384
还要考虑梯度、中间结果等，实际占用更多
```

### 2.11.4 后续的优化方向（预告）

为了解决 O(n²) 的问题，研究者提出了很多优化方法：

| 方法 | 思路 | 代表模型 |
|------|------|---------|
| Sparse Attention | 只计算部分注意力 | Longformer, BigBird |
| Linear Attention | 用核函数近似，降到 O(n) | Performer, Linear Transformer |
| Flash Attention | 优化内存访问，不改变结果 | Flash Attention 1/2 |
| 滑动窗口 | 只关注局部 + 全局 token | Mistral, Longformer |

这些会在后面的章节详细介绍。

## 2.12 Cross-Attention 详解

### 2.12.1 什么是 Cross-Attention？

Cross-Attention（交叉注意力）是 Encoder-Decoder 架构中连接两个部分的桥梁。

```
Self-Attention：Q、K、V 来自同一个序列
Cross-Attention：Q 来自一个序列，K、V 来自另一个序列
```

### 2.12.2 机器翻译中的 Cross-Attention

```
源语言（英文）："I love you"
目标语言（中文）："我 爱 你"

Encoder 处理源语言：
"I love you" → [Encoder] → 编码表示 (encoder_output)

Decoder 生成目标语言时：
当前要生成"爱"这个词

Cross-Attention：
- Q = "我"的表示（来自 Decoder，已生成的部分）
- K = encoder_output（来自 Encoder）
- V = encoder_output（来自 Encoder）

作用：让 Decoder 知道应该关注源语言的哪个部分
生成"爱"时，应该重点关注"love"
```

### 2.12.3 Cross-Attention 的计算过程

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q 来自 Decoder
        self.W_Q = nn.Linear(d_model, d_model)
        # K, V 来自 Encoder
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, decoder_hidden, encoder_output):
        """
        decoder_hidden: Decoder 的隐藏状态 (batch, tgt_len, d_model)
        encoder_output: Encoder 的输出 (batch, src_len, d_model)
        """
        # Q 来自 Decoder
        Q = self.W_Q(decoder_hidden)  # (batch, tgt_len, d_model)
        
        # K, V 来自 Encoder
        K = self.W_K(encoder_output)  # (batch, src_len, d_model)
        V = self.W_V(encoder_output)  # (batch, src_len, d_model)
        
        # 注意力计算
        # scores: (batch, tgt_len, src_len)
        # 每个目标词对每个源词的注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 输出: (batch, tgt_len, d_model)
        output = attention_weights @ V
        return self.W_O(output)
```

### 2.12.4 Cross-Attention 的可视化

```
翻译 "I love you" → "我 爱 你"

Cross-Attention 矩阵：
              I     love    you
    我      [0.8    0.1    0.1]   ← "我"主要关注"I"
    爱      [0.1    0.8    0.1]   ← "爱"主要关注"love"
    你      [0.1    0.1    0.8]   ← "你"主要关注"you"

这就是注意力机制的"对齐"能力！
模型自动学会了词与词之间的对应关系
```

### 2.12.5 Self-Attention vs Cross-Attention 对比

| 特性 | Self-Attention | Cross-Attention |
|------|---------------|-----------------|
| Q 来源 | 当前序列 | Decoder |
| K/V 来源 | 当前序列 | Encoder |
| 作用 | 序列内部建模 | 序列之间交互 |
| 使用位置 | Encoder、Decoder | 只在 Decoder |
| 是否用掩码 | Decoder 中用 | 不用 |

## 2.13 训练技巧详解

### 2.13.1 Dropout 的作用

Dropout 是一种正则化技术，**核心目的是防止过拟合，增强泛化能力**。

#### Dropout 要解决什么问题？

```
问题：神经元之间会产生"共适应"（Co-adaptation）

什么是共适应？
- 某些神经元总是一起工作，形成固定的"小团体"
- 它们互相依赖，没有独立学习能力
- 一旦某个神经元出错，整个团体都会出错
- 模型变得脆弱，泛化能力差

例子：
神经元 A 和 B 总是一起激活
A 学会了：只要 B 说"是"，我就说"是"
B 学会了：只要 A 说"是"，我就说"是"
它们互相依赖，都没有真正学会独立判断
```

#### Dropout 怎么解决？

```
训练时随机"关闭"一部分神经元：

第1次训练：[A, -, C, D, -]  （B、E 被关闭）
第2次训练：[-, B, C, -, E]  （A、D 被关闭）
第3次训练：[A, B, -, D, -]  （C、E 被关闭）
...

效果：
- 每个神经元不能依赖其他特定神经元
- 必须独立学习有用的特征
- 相当于训练了很多个"子网络"
- 最终模型 = 这些子网络的集成（ensemble）
```

#### 类比理解

```
类比：团队考试

没有 Dropout：
- 5个人组队，每次都一起答题
- 小明只负责数学，小红只负责语文
- 一旦小明缺席，整个团队数学就完蛋

有 Dropout：
- 每次考试随机抽3个人参加
- 每个人都必须学会所有科目
- 任何人缺席，团队都能应对

结果：每个人都更全面，团队更稳健
```

#### Dropout 的具体操作

```
训练时：随机将一部分神经元的输出置为 0
测试时：使用所有神经元（不 dropout）

例子（dropout_rate = 0.1，即 10% 的神经元被关闭）：
原始输出：[0.5, 0.3, 0.2, 0.8, 0.2, 0.1, 0.6, 0.4, 0.1, 0.7]
Dropout后：[0.5, 0.3,  0 , 0.8, 0.2,  0 , 0.6, 0.4, 0.1,  0 ]
                      ↑              ↑                    ↑
                   被随机关闭的神经元
```

#### 常见误解

| 误解 | 正确理解 |
|------|---------|
| 让剩下的神经元学习更充分 | 让每个神经元独立学习，不依赖其他神经元 |
| 重点在"学更多" | 重点在"防止过度依赖"和"增强泛化" |
| 关闭神经元是为了减少计算 | 关闭神经元是为了打破共适应 |

**核心思想**：不是让神经元"学更多"，而是让神经元"不要偷懒依赖别人"。

#### Transformer 中 Dropout 的位置

```
1. 注意力权重后：attention_weights = dropout(softmax(scores))
2. 子层输出后：output = dropout(sublayer(x))
3. 词嵌入后：embedding = dropout(word_emb + pos_emb)
```

### 2.13.2 学习率预热（Warmup）

Transformer 训练时使用特殊的学习率策略。

```
问题：
- 训练初期，参数是随机的，梯度可能很大
- 如果学习率太大，可能导致训练不稳定

解决方案：Warmup
- 训练开始时用很小的学习率
- 逐渐增大到目标学习率
- 然后再逐渐减小

学习率变化曲线：
     ^
学习率|      /\
     |     /  \
     |    /    \
     |   /      \____
     |  /
     | /
     +-------------------> 训练步数
       warmup   decay
```

**原始 Transformer 的学习率公式**：

$lr = d_{model}^{-0.5} \times \min(step^{-0.5}, step \times warmup\_steps^{-1.5})$

```python
def get_lr(step, d_model, warmup_steps=4000):
    """Transformer 原始论文的学习率策略"""
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

# warmup_steps = 4000 步
# 前 4000 步：学习率线性增加
# 4000 步后：学习率按 step^(-0.5) 衰减
```

### 2.13.3 Label Smoothing（标签平滑）

```
普通的标签（one-hot）：
正确答案是"苹果"（词表中第 100 个词）
标签 = [0, 0, ..., 1, ..., 0]  # 只有位置 100 是 1
              位置100

问题：
- 模型被迫输出极端的概率分布（接近 0 或 1）
- 可能导致过拟合
- 模型过于自信

Label Smoothing（ε = 0.1）：
标签 = [0.001, 0.001, ..., 0.9, ..., 0.001]
                        位置100

把一部分概率（ε）均匀分给其他词
让模型不要过于自信
```

```python
def label_smoothing(target, num_classes, smoothing=0.1):
    """
    target: 真实标签的索引
    num_classes: 词表大小
    smoothing: 平滑系数
    """
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    
    # 创建平滑后的标签
    smooth_target = torch.full((num_classes,), smooth_value)
    smooth_target[target] = confidence
    
    return smooth_target
```

### 2.13.4 梯度裁剪（Gradient Clipping）

```
问题：梯度可能变得非常大（梯度爆炸）

解决方案：限制梯度的最大值

# PyTorch 中的实现
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 如果梯度的范数超过 max_norm，就按比例缩小
```

## 2.14 Transformer 的局限性

### 2.14.1 计算复杂度 O(n²)

```
问题：
- 自注意力的复杂度是 O(n²)
- 处理长文本时，计算量和内存都会爆炸
- 限制了上下文长度

影响：
- 早期模型上下文长度只有 512（BERT）或 1024（GPT）
- 无法处理长文档、长对话
- 需要截断或分块处理

后续改进：
- Sparse Attention（稀疏注意力）
- Linear Attention（线性注意力）
- Flash Attention（优化内存访问）
```

### 2.14.2 位置编码的局限

**问题1：绝对位置 vs 相对位置**

```
原始的位置编码是绝对位置：
位置 0 的编码、位置 1 的编码、位置 2 的编码...

但语言中，相对位置往往更重要：
"我 喜欢 你" 中，"喜欢"和"你"的关系
不管它们在句子的什么位置，关系都是一样的

改进：相对位置编码
- Relative Position Encoding
- RoPE（Rotary Position Embedding）← 现在最流行
- ALiBi（Attention with Linear Biases）
```

**问题2：长度外推性**

```
问题：
训练时最长序列是 2048
测试时遇到 4096 的序列，效果会变差

原因：
- 可学习的位置编码：没见过位置 2048 以后的
- 正弦余弦编码：理论上可以外推，但实际效果也会下降

改进：
- RoPE 有更好的外推性
- ALiBi 专门设计来解决这个问题
- 位置插值（Position Interpolation）
```

### 2.14.3 缺乏显式的层次结构

```
问题：
Transformer 把所有 token 平等对待
没有显式的层次结构（词 → 短语 → 句子 → 段落）

例子：
"北京是中国的首都"

人类理解：
- "北京" 是一个实体
- "中国的首都" 是一个短语
- 整句话表达一个事实

Transformer：
- 只看到 6 个 token
- 通过注意力隐式学习结构
- 但不如显式建模那么高效
```

### 2.14.4 训练不稳定

```
问题：
- Transformer 训练时容易出现 loss 突然变大（spike）
- 深层 Transformer 更难训练
- 需要精心调整超参数

原因：
- 注意力分数可能变得很大或很小
- 残差连接的累积效应
- LayerNorm 的位置影响稳定性

解决方案：
- Pre-LN（把 LayerNorm 放在子层之前）
- 学习率 Warmup
- 梯度裁剪
- 更好的初始化方法
```

### 2.14.5 推理效率问题

```
问题：自回归生成时，每生成一个 token 都要重新计算

生成 "我 喜欢 吃 苹果"：
第1步：输入 <BOS>，计算注意力，输出 "我"
第2步：输入 <BOS> 我，重新计算注意力，输出 "喜欢"
第3步：输入 <BOS> 我 喜欢，重新计算注意力，输出 "吃"
...

每一步都要重新计算之前所有 token 的注意力！
非常浪费

解决方案：KV Cache
- 缓存之前计算过的 K 和 V
- 新 token 只需要计算自己的 Q
- 大幅提升推理速度
```

```python
# KV Cache 的基本思想
class AttentionWithKVCache:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, use_cache=True):
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)
        
        if use_cache and self.k_cache is not None:
            # 拼接缓存的 K, V
            k = torch.cat([self.k_cache, k], dim=1)
            v = torch.cat([self.v_cache, v], dim=1)
        
        # 更新缓存
        self.k_cache = k
        self.v_cache = v
        
        # 计算注意力
        # q 只有当前 token，k/v 包含所有历史 token
        output = attention(q, k, v)
        return output
```

## 2.15 Transformer 的历史意义

### 2.15.1 为什么 Transformer 如此重要？

```
1. 开创了新范式
   - 证明了不需要 RNN 也能处理序列
   - "Attention Is All You Need" 成为经典

2. 催生了预训练语言模型
   - BERT（2018）：Encoder-only
   - GPT（2018）：Decoder-only
   - 开启了 NLP 的预训练时代

3. 统一了多个领域
   - NLP：BERT, GPT, T5
   - CV：ViT, Swin Transformer
   - 多模态：CLIP, DALL-E
   - 语音：Whisper
   
4. 奠定了大模型的基础
   - GPT-3, GPT-4
   - LLaMA, Claude
   - 所有现代 LLM 都基于 Transformer
```

### 2.15.2 Transformer 的演进时间线

```
2017: Transformer 诞生（Google，机器翻译）
      ↓
2018: GPT-1（OpenAI，Decoder-only）
      BERT（Google，Encoder-only）
      ↓
2019: GPT-2（OpenAI，更大的 GPT）
      T5（Google，Encoder-Decoder）
      ↓
2020: GPT-3（OpenAI，1750亿参数）
      ViT（Google，Transformer 进入 CV）
      ↓
2021: CLIP（OpenAI，多模态）
      Codex（OpenAI，代码生成）
      ↓
2022: ChatGPT（OpenAI，对话）
      InstructGPT（RLHF）
      ↓
2023: GPT-4（OpenAI，多模态）
      LLaMA（Meta，开源）
      ↓
2024: Claude 3, Gemini, LLaMA 3...
      持续演进中...
```

## 本章小结

### 核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| Attention | 让模型关注重要的部分，忽略不重要的 |
| Q, K, V | Query、Key、Value，注意力的三要素 |
| 自注意力 | Q/K/V 来自同一序列，序列内部相互关注 |
| 多头注意力 | 多个注意力并行，学习不同的关注模式 |
| 掩码注意力 | 遮蔽未来信息，用于生成任务 |
| 位置编码 | 给模型提供位置信息 |
| FFN | 前馈网络，增加非线性 |
| 残差连接 | 跳跃连接，帮助梯度流动 |
| 层归一化 | 标准化，稳定训练 |

### 注意力计算流程

```
1. 生成 Q, K, V（线性变换）
2. 计算分数：Q × K^T
3. 缩放：÷ √d_k
4. Softmax：得到注意力权重
5. 加权求和：权重 × V
```

### Transformer 的优势

| 对比项 | RNN | Transformer |
|--------|-----|-------------|
| 并行计算 | ❌ 串行 | ✅ 并行 |
| 长距离依赖 | 难以捕捉 | 直接建模 |
| 训练速度 | 慢 | 快 |
| 可解释性 | 差 | 好（注意力可视化） |

### Transformer 的局限性速查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| O(n²) 复杂度 | 注意力矩阵是 n×n | Sparse/Linear Attention |
| 位置外推差 | 位置编码的限制 | RoPE, ALiBi |
| 推理慢 | 自回归重复计算 | KV Cache |
| 训练不稳定 | 梯度问题 | Pre-LN, Warmup |

### 本章公式汇总

```
1. 注意力公式：
   Attention(Q,K,V) = softmax(QK^T / √d_k) × V

2. 多头注意力：
   MultiHead = Concat(head_1, ..., head_h) × W_O
   head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

3. 位置编码：
   PE(pos, 2i) = sin(pos / 10000^(2i/d))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

4. FFN：
   FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

5. Layer Norm：
   LN(x) = γ × (x - μ) / σ + β
```

### 下一章预告

第3章将学习**预训练语言模型（PLM）**：
- BERT（Encoder-only）：双向编码，理解任务
- GPT（Decoder-only）：单向生成，生成任务
- T5（Encoder-Decoder）：统一框架
- 预训练任务：MLM、NSP、CLM
- 微调（Fine-tuning）技术

这些模型都是基于 Transformer 构建的，理解了 Transformer，学习它们就会轻松很多！

---

## 思考题

1. 为什么 Transformer 要用多头注意力而不是单头？如果只用一个头会怎样？

2. 如果不使用位置编码，"我喜欢你"和"你喜欢我"对模型来说有区别吗？

3. 为什么 Decoder 的自注意力要用掩码，而 Encoder 不用？

4. 假设序列长度从 1024 增加到 4096，注意力的计算量会增加多少倍？

5. KV Cache 为什么能加速推理？它的代价是什么？

---

## 推荐阅读

1. **原始论文**：《Attention Is All You Need》(2017)
   - 必读！Transformer 的开山之作

2. **图解 Transformer**：Jay Alammar 的博客
   - http://jalammar.github.io/illustrated-transformer/
   - 非常直观的可视化解释

3. **The Annotated Transformer**：Harvard NLP
   - 带代码注释的 Transformer 实现
   - 适合想深入理解代码的同学

4. **李宏毅老师的课程**：
   - B站搜索"李宏毅 Transformer"
   - 中文讲解，通俗易懂

---
学习日期：2025年12月
