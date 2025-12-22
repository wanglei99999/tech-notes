# 第3章 预训练语言模型（PLM）

预训练语言模型（Pre-trained Language Model, PLM）是连接 Transformer 和大语言模型（LLM）的桥梁。理解 PLM 是理解 ChatGPT、LLaMA 等现代 LLM 的基础。

## 3.0 本章概览

```
Transformer（2017）
      ↓
   三种变体
      ↓
┌─────────────┬─────────────┬─────────────┐
│ Encoder-only│Encoder-Decoder│Decoder-only │
│   (BERT)    │    (T5)     │   (GPT)     │
│   理解任务   │   翻译/摘要  │   生成任务   │
└─────────────┴─────────────┴─────────────┘
                                  ↓
                              LLM 时代
                          (GPT-3, LLaMA, ChatGPT)
```

**本章将学习**：
- 什么是预训练？为什么需要预训练？
- 三种架构：Encoder-only、Encoder-Decoder、Decoder-only
- 代表模型：BERT、T5、GPT、LLaMA
- 预训练任务：MLM、NSP、CLM
- 从 PLM 到 LLM 的演进

---

## 3.1 什么是预训练？

### 3.1.1 传统 NLP 的困境

在预训练时代之前，NLP 任务是这样做的：

```
传统方式：每个任务从零开始

任务1（情感分类）：
  收集标注数据 → 训练模型A → 部署

任务2（命名实体识别）：
  收集标注数据 → 训练模型B → 部署

任务3（机器翻译）：
  收集标注数据 → 训练模型C → 部署

问题：
- 每个任务都需要大量标注数据（人工标注成本高）
- 每个任务都要从零训练（计算资源浪费）
- 模型之间不能共享知识
```

### 3.1.2 预训练的核心思想

**核心思想**：先在海量无标注文本上学习通用的语言知识，再针对具体任务微调。

```
预训练 + 微调范式：

第一阶段：预训练（Pre-training）
┌─────────────────────────────────────────┐
│  海量无标注文本（维基百科、新闻、书籍...）  │
│         ↓                               │
│    预训练任务（如预测下一个词）            │
│         ↓                               │
│    通用语言模型（学会了语言的规律）        │
└─────────────────────────────────────────┘

第二阶段：微调（Fine-tuning）
┌─────────────────────────────────────────┐
│    预训练好的模型                         │
│         ↓                               │
│    少量标注数据（几百~几千条）             │
│         ↓                               │
│    针对具体任务的模型                     │
└─────────────────────────────────────────┘
```

### 3.1.3 为什么预训练有效？

```
类比：学习外语

传统方式（从零学习）：
- 学情感分析：背"好/坏/喜欢/讨厌"等词
- 学机器翻译：背"你好=Hello"等对照
- 学问答系统：背"什么是...=..."等模式
- 每个任务都要重新学基础词汇和语法

预训练方式（先学通用知识）：
- 预训练：先花大量时间学习语法、词汇、常识
- 微调：学会基础后，学具体任务就很快了

人类学习也是这样！
- 我们先学会语言的基本规律（预训练）
- 然后学习具体技能就很快（微调）
```

### 3.1.4 预训练的优势

| 对比项 | 传统方式 | 预训练+微调 |
|--------|---------|------------|
| 标注数据需求 | 大量（万级） | 少量（百级） |
| 训练时间 | 每个任务都长 | 微调很快 |
| 知识共享 | 不能 | 可以 |
| 效果 | 一般 | 更好 |


## 3.2 三种 Transformer 架构

基于 Transformer，研究者发展出了三种不同的架构：

### 3.2.1 架构对比总览

```
原始 Transformer = Encoder + Decoder

三种变体：

1. Encoder-only（只用 Encoder）
   代表：BERT
   特点：双向理解，擅长 NLU（理解）任务
   
2. Decoder-only（只用 Decoder）
   代表：GPT
   特点：单向生成，擅长 NLG（生成）任务
   
3. Encoder-Decoder（完整结构）
   代表：T5
   特点：输入→输出，擅长 Seq2Seq 任务
```

### 3.2.2 三种架构的区别

```
输入："我 喜欢 苹果"

Encoder-only (BERT)：
- 每个词可以看到所有词（双向）
- "苹果"可以看到"我"和"喜欢"
- 适合理解任务（分类、NER）

  我 ←→ 喜欢 ←→ 苹果
  （双向注意力，无掩码）


Decoder-only (GPT)：
- 每个词只能看到前面的词（单向）
- "苹果"可以看到"我"和"喜欢"
- "喜欢"只能看到"我"
- 适合生成任务（续写、对话）

  我 → 喜欢 → 苹果
  （单向注意力，有掩码）


Encoder-Decoder (T5)：
- Encoder：双向理解输入
- Decoder：单向生成输出
- 适合转换任务（翻译、摘要）

  [Encoder: 双向理解] → [Decoder: 单向生成]
```

### 3.2.3 为什么会有不同架构？

```
不同任务需要不同的"看"的方式：

理解任务（BERT 擅长）：
- 情感分类："这个产品真的很___" → 需要看完整句子
- 命名实体识别：需要上下文判断"苹果"是水果还是公司
- 需要双向理解

生成任务（GPT 擅长）：
- 文本续写："今天天气" → "很好"
- 对话生成：根据历史生成回复
- 只能看到已生成的内容，不能"偷看"未来

转换任务（T5 擅长）：
- 翻译："I love you" → "我爱你"
- 摘要：长文章 → 短摘要
- 需要先理解输入，再生成输出
```


## 3.3 Encoder-only：BERT

BERT（Bidirectional Encoder Representations from Transformers）是 2018 年 Google 发布的里程碑模型，开启了预训练语言模型时代。

### 3.3.1 BERT 的核心创新

```
BERT 之前的问题：

传统语言模型（如 GPT-1）只能单向看：
"我 喜欢 [?]" → 预测下一个词

问题：预测时只能看到左边的词，不能利用右边的信息

BERT 的解决方案：双向！

"我 [MASK] 苹果" → 预测被遮住的词

现在模型可以同时看到左边的"我"和右边的"苹果"
来预测中间的词应该是"喜欢"
```

### 3.3.2 BERT 的模型结构

```
BERT = 多层 Transformer Encoder 堆叠

输入：[CLS] 我 喜欢 苹果 [SEP]
        ↓
    Embedding（词向量 + 位置编码 + 段落编码）
        ↓
    ┌─────────────┐
    │ Encoder × N │  ← BERT-base: 12层
    │             │  ← BERT-large: 24层
    └─────────────┘
        ↓
    输出：每个位置的隐藏状态

特殊 Token：
- [CLS]：句子开头，用于分类任务
- [SEP]：句子分隔符
- [MASK]：遮蔽的词，用于 MLM 任务
```

### 3.3.3 预训练任务1：MLM（Masked Language Model）

MLM 是 BERT 最重要的创新，也叫"完形填空"。

```
MLM 的过程：

原始句子："我 喜欢 吃 苹果"

随机遮蔽 15% 的词：
"我 [MASK] 吃 苹果"

让模型预测被遮蔽的词：
模型输出：[MASK] → "喜欢"

为什么是 15%？
- 太少：模型学不到足够的信息
- 太多：上下文信息太少，难以预测
- 15% 是实验得出的最佳比例
```

**MLM 的数学形式**：

给定输入序列 $x = (x_1, x_2, ..., x_n)$，随机选择 15% 的位置集合 $M$，MLM 的损失函数为：

$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\backslash M}; \theta)$$

其中：
- $x_{\backslash M}$ 表示被遮蔽后的序列
- $\theta$ 是模型参数
- $P(x_i | x_{\backslash M})$ 是模型预测第 $i$ 个位置原始 token 的概率

**MLM 的细节处理**：

```
被选中的 15% 的词，不是全部变成 [MASK]：

- 80% 概率：替换成 [MASK]
  "我 喜欢 苹果" → "我 [MASK] 苹果"

- 10% 概率：替换成随机词
  "我 喜欢 苹果" → "我 香蕉 苹果"

- 10% 概率：保持不变
  "我 喜欢 苹果" → "我 喜欢 苹果"

为什么这样设计？
- 如果全是 [MASK]，模型只学会处理 [MASK]
- 微调时没有 [MASK]，会有不一致
- 随机替换让模型保持对所有词的关注
```

### 3.3.4 预训练任务2：NSP（Next Sentence Prediction）

NSP 用于学习句子之间的关系。

```
NSP 的过程：

输入两个句子，判断 B 是不是 A 的下一句

正例（IsNext）：
A: "我喜欢吃苹果"
B: "苹果很有营养"
标签：1（是连续的）

负例（NotNext）：
A: "我喜欢吃苹果"
B: "今天天气很好"
标签：0（不是连续的）

作用：
- 帮助模型理解句子之间的关系
- 对问答、推理等任务有帮助
```

**注意**：后来的研究（RoBERTa）发现 NSP 任务效果不明显，很多后续模型去掉了它。

### 3.3.5 BERT 的输入表示

```
BERT 的输入 = Token Embedding + Segment Embedding + Position Embedding

例子：判断两个句子的关系
输入：[CLS] 我 喜欢 苹果 [SEP] 苹果 很 好吃 [SEP]

Token Embedding：每个词的词向量
Segment Embedding：标记属于句子A还是句子B
Position Embedding：位置信息

      [CLS]  我   喜欢  苹果  [SEP]  苹果   很   好吃  [SEP]
Token:  E1   E2   E3    E4    E5    E6    E7    E8    E9
Segment: A    A    A     A     A     B     B     B     B
Position: 0    1    2     3     4     5     6     7     8

最终输入 = Token + Segment + Position（三者相加）
```

### 3.3.6 BERT 的微调

```
预训练完成后，针对不同任务微调：

1. 文本分类（情感分析）：
   输入：[CLS] 这个产品很好 [SEP]
   输出：取 [CLS] 的隐藏状态 → 分类器 → 正面/负面

2. 命名实体识别（NER）：
   输入：[CLS] 小明 在 北京 工作 [SEP]
   输出：每个词的隐藏状态 → 分类器 → 人名/地名/其他

3. 问答任务：
   输入：[CLS] 问题 [SEP] 文章 [SEP]
   输出：预测答案在文章中的起始和结束位置

微调的特点：
- 只需要少量标注数据（几百~几千条）
- 训练时间短（几小时）
- 效果很好
```

### 3.3.7 BERT 的规格

| 模型 | 层数 | 隐藏维度 | 注意力头数 | 参数量 |
|------|------|---------|-----------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |


## 3.4 BERT 的改进：RoBERTa 和 ALBERT

### 3.4.1 RoBERTa：更强的 BERT

RoBERTa（Robustly Optimized BERT Approach）是 Facebook 对 BERT 的优化版本。

```
RoBERTa 的改进：

1. 去掉 NSP 任务
   - 实验发现 NSP 对效果提升不明显
   - 只保留 MLM 任务

2. 动态遮蔽
   - BERT：预处理时就固定遮蔽位置
   - RoBERTa：每次训练时动态随机遮蔽
   - 让模型看到更多样的训练数据

3. 更大的数据和更长的训练
   - BERT：13GB 数据
   - RoBERTa：160GB 数据（10倍！）
   - 训练更多步数

4. 更大的 batch size
   - BERT：256
   - RoBERTa：8000
   - 大 batch 训练更稳定

结果：RoBERTa 在多个任务上超越 BERT
```

### 3.4.2 ALBERT：更小的 BERT

ALBERT（A Lite BERT）的目标是减少参数量。

```
ALBERT 的改进：

1. Embedding 分解
   - BERT：词向量维度 = 隐藏层维度（768）
   - ALBERT：词向量维度 = 128，再映射到 768
   - 大幅减少 Embedding 层参数

2. 跨层参数共享
   - BERT：每层 Encoder 有独立参数
   - ALBERT：所有层共享同一套参数
   - 参数量大幅减少

3. SOP 替代 NSP
   - NSP：判断两句是否连续
   - SOP（Sentence Order Prediction）：判断两句顺序是否正确
   - SOP 更难，效果更好

结果：
- ALBERT-xxlarge：参数量 235M（vs BERT-large 340M）
- 效果更好，但推理速度没有提升（因为计算量没变）
```


## 3.5 Encoder-Decoder：T5

T5（Text-to-Text Transfer Transformer）是 Google 提出的"大一统"模型。

### 3.5.1 T5 的核心思想：一切皆文本

```
T5 的革命性思想：把所有 NLP 任务统一成"文本到文本"的格式

传统方式：不同任务用不同的输出格式
- 分类任务：输出类别 ID（0, 1, 2...）
- 翻译任务：输出目标语言文本
- 问答任务：输出答案的位置

T5 方式：所有任务都是"输入文本 → 输出文本"
- 分类任务：输入文本 → 输出"positive"或"negative"
- 翻译任务：输入文本 → 输出翻译结果
- 问答任务：输入文本 → 输出答案文本
```

### 3.5.2 T5 的任务格式

```
T5 通过添加"任务前缀"来区分不同任务：

1. 翻译任务：
   输入："translate English to German: How are you?"
   输出："Wie geht es dir?"

2. 摘要任务：
   输入："summarize: [长文章内容]"
   输出："[摘要内容]"

3. 情感分类：
   输入："sentiment: This movie is great!"
   输出："positive"

4. 问答任务：
   输入："question: What is the capital of France? context: Paris is the capital of France."
   输出："Paris"

好处：
- 一个模型处理所有任务
- 不需要为每个任务设计特殊的输出层
- 模型可以学习任务之间的共性
```

### 3.5.3 T5 的模型结构

```
T5 = 完整的 Encoder-Decoder 结构

输入文本
    ↓
┌─────────────┐
│   Encoder   │  ← 双向注意力，理解输入
│   (N 层)    │
└─────────────┘
    ↓
┌─────────────┐
│   Decoder   │  ← 单向注意力 + Cross-Attention
│   (N 层)    │
└─────────────┘
    ↓
输出文本

与原始 Transformer 的区别：
- 使用相对位置编码
- LayerNorm 使用 RMSNorm
- 预训练任务是 Span Corruption（遮蔽连续片段）
```

### 3.5.4 T5 的预训练任务：Span Corruption

```
Span Corruption = 遮蔽连续的片段

原始句子：
"Thank you for inviting me to your party last week"

遮蔽后：
"Thank you <X> me to your party <Y> week"

目标输出：
"<X> for inviting <Y> last"

与 BERT 的 MLM 区别：
- BERT：遮蔽单个词
- T5：遮蔽连续的多个词（span）
- T5 需要生成被遮蔽的内容，而不只是预测
```

### 3.5.5 T5 的规格

| 模型 | 参数量 | 说明 |
|------|--------|------|
| T5-small | 60M | 小型 |
| T5-base | 220M | 基础 |
| T5-large | 770M | 大型 |
| T5-3B | 3B | 超大 |
| T5-11B | 11B | 巨型 |


## 3.6 Decoder-only：GPT 系列

GPT（Generative Pre-Training）是 OpenAI 的代表作，也是现代 LLM 的基础。

### 3.6.1 GPT 的核心思想

```
GPT 的思路非常简单：预测下一个词

训练数据：任何文本
训练任务：给定前面的词，预测下一个词

例子：
输入："今天天气"
目标：预测下一个词是"很"

输入："今天天气很"
目标：预测下一个词是"好"

这就是 CLM（Causal Language Model，因果语言模型）
也叫自回归语言模型（Autoregressive LM）
```

### 3.6.2 GPT vs BERT：两种不同的思路

```
BERT（完形填空）：
"我 [MASK] 苹果" → 预测中间的词
- 可以看到左右两边
- 适合理解任务
- 不擅长生成

GPT（续写）：
"我 喜欢" → 预测下一个词
- 只能看到左边
- 适合生成任务
- 更接近人类写作方式

为什么 GPT 最终胜出？
- 生成任务更通用（理解可以转化为生成）
- 更容易扩展（数据无限，模型可以无限大）
- 涌现能力在大模型上更明显
```

### 3.6.3 GPT 的模型结构

```
GPT = 多层 Transformer Decoder 堆叠（但没有 Cross-Attention）

输入："今天 天气 很"
        ↓
    Embedding + Position Embedding
        ↓
    ┌─────────────────────┐
    │ Masked Self-Attention│  ← 只能看到前面的词
    │        ↓            │
    │       FFN           │
    │        ↓            │
    │    × N 层           │
    └─────────────────────┘
        ↓
    预测下一个词："好"

注意：
- 使用 Masked Self-Attention（因果掩码）
- 每个位置只能看到自己和前面的位置
- 没有 Encoder，也没有 Cross-Attention
```

### 3.6.4 GPT 系列的演进

```
GPT-1 (2018)：
- 参数：117M
- 数据：5GB（BooksCorpus）
- 特点：首次提出预训练+微调范式
- 效果：不如同期的 BERT

GPT-2 (2019)：
- 参数：1.5B（是 GPT-1 的 10 倍）
- 数据：40GB（WebText）
- 特点：探索 zero-shot 能力
- 效果：在某些任务上不需要微调

GPT-3 (2020)：
- 参数：175B（是 GPT-2 的 100 倍！）
- 数据：570GB
- 特点：涌现能力！few-shot 学习
- 效果：开启 LLM 时代

ChatGPT (2022)：
- 基于 GPT-3.5
- 加入 RLHF（人类反馈强化学习）
- 特点：对话能力大幅提升
- 效果：引爆 AI 热潮
```

### 3.6.5 GPT 的预训练任务：CLM

```
CLM（Causal Language Model）= 预测下一个词

数学表示：
P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ...

例子：
P("今天天气很好") = P("今天") × P("天气"|"今天") × P("很"|"今天天气") × P("好"|"今天天气很")

训练过程：
输入：  [BOS] 今天 天气 很  好
目标：  今天  天气  很  好 [EOS]

每个位置预测下一个词，计算交叉熵损失
```

**CLM 的数学形式**：

给定序列 $x = (x_1, x_2, ..., x_n)$，CLM 的目标是最大化序列的联合概率：

$$P(x) = \prod_{i=1}^{n} P(x_i | x_1, x_2, ..., x_{i-1}; \theta)$$

对应的损失函数（负对数似然）：

$$\mathcal{L}_{CLM} = -\sum_{i=1}^{n} \log P(x_i | x_{<i}; \theta)$$

其中 $x_{<i} = (x_1, ..., x_{i-1})$ 表示位置 $i$ 之前的所有 token。

**Perplexity（困惑度）**：

困惑度是评估语言模型的常用指标：

$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_{<i})\right) = \exp(\mathcal{L}_{CLM})$$

- PPL 越低，模型越好
- PPL = 1 表示模型完美预测
- GPT-3 在测试集上的 PPL 约为 20-30

### 3.6.6 为什么 GPT 路线最终胜出？

```
2018-2020 年：BERT 系列占主导
- BERT 在各种 NLU 任务上效果最好
- GPT 被认为只适合生成任务

2020 年后：GPT 路线逆袭
原因：

1. Scaling Law（规模定律）
   - 模型越大，效果越好
   - GPT-3 证明了这一点

2. 涌现能力（Emergent Abilities）
   - 小模型没有的能力，大模型突然有了
   - 如：few-shot 学习、推理能力

3. 通用性
   - 生成任务可以覆盖理解任务
   - "判断情感" → "这句话的情感是___"
   - 一个模型解决所有问题

4. 数据效率
   - CLM 可以用任何文本训练
   - 互联网上有无限的文本数据
```


## 3.7 LLaMA：开源 LLM 的基石

LLaMA（Large Language Model Meta AI）是 Meta 发布的开源大模型，是目前大多数开源 LLM 的基础。

### 3.7.1 LLaMA 的重要性

```
为什么 LLaMA 如此重要？

1. 开源
   - GPT-3/4 是闭源的，只能通过 API 使用
   - LLaMA 开源了模型权重
   - 让研究者和开发者可以自由使用和修改

2. 效果好
   - LLaMA-13B 效果接近 GPT-3（175B）
   - 证明了小模型也能很强

3. 生态丰富
   - Alpaca、Vicuna、ChatGLM 等都基于 LLaMA
   - 形成了庞大的开源生态
```

### 3.7.2 LLaMA 的模型结构

```
LLaMA 基于 GPT 架构，但有一些改进：

1. Pre-Norm（前置归一化）
   - GPT：Attention → Add → Norm
   - LLaMA：Norm → Attention → Add
   - 训练更稳定

2. RMSNorm 替代 LayerNorm
   - 计算更简单，效果相当
   - RMSNorm(x) = x / RMS(x) × γ

3. SwiGLU 激活函数
   - 替代 ReLU/GELU
   - FFN 效果更好

4. RoPE 位置编码
   - 旋转位置编码
   - 更好的长度外推能力
```

### 3.7.3 RoPE（旋转位置编码）详解

```
RoPE 是 LLaMA 的重要创新，解决了位置编码的问题。

传统位置编码的问题：
- 绝对位置编码：位置 0, 1, 2, 3...
- 问题：训练时最长 2048，推理时遇到 4096 怎么办？

RoPE 的思想：
- 用旋转矩阵编码相对位置
- 位置信息融入到 Q 和 K 中
- 天然支持相对位置

RoPE 的优势：
- 更好的长度外推能力
- 可以处理比训练时更长的序列
- 现在几乎所有 LLM 都用 RoPE
```

**RoPE 的数学原理**（Su et al., 2021）：

RoPE 的核心思想：通过旋转矩阵将位置信息编码到 Q 和 K 中，使得 $q_m^T k_n$ 只依赖于相对位置 $m - n$。

对于位置 $m$ 的向量 $x$，RoPE 应用旋转变换：

$$f(x, m) = R_m x$$

其中旋转矩阵 $R_m$ 定义为（以 2D 为例）：

$$R_m = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

对于 $d$ 维向量，将其分成 $d/2$ 组，每组应用不同频率的旋转：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1$$

**关键性质**：

$$\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle$$

即：旋转后的内积只依赖于相对位置 $m - n$！

### 3.7.4 LLaMA 系列演进

```
LLaMA-1 (2023.02)：
- 规格：7B, 13B, 30B, 65B
- 数据：1T tokens
- 特点：首个高质量开源 LLM

LLaMA-2 (2023.07)：
- 规格：7B, 13B, 70B
- 数据：2T tokens
- 改进：
  - 上下文长度 2048 → 4096
  - GQA（分组查询注意力）
  - 更多训练数据

LLaMA-3 (2024.04)：
- 规格：8B, 70B, 400B（训练中）
- 数据：15T tokens
- 改进：
  - 上下文长度 8K
  - 更大的词表（128K）
  - 效果大幅提升
```

### 3.7.5 GQA（Grouped-Query Attention）

```
GQA 是 LLaMA-2 引入的优化，减少 KV Cache 的内存占用。

传统 MHA（Multi-Head Attention）：
- 每个头有独立的 Q, K, V
- 8 个头 = 8 组 Q + 8 组 K + 8 组 V
- KV Cache 很大

MQA（Multi-Query Attention）：
- 所有头共享同一组 K, V
- 8 个头 = 8 组 Q + 1 组 K + 1 组 V
- KV Cache 很小，但效果下降

GQA（Grouped-Query Attention）：
- 折中方案：几个头共享一组 K, V
- 8 个头 = 8 组 Q + 2 组 K + 2 组 V
- KV Cache 减少，效果基本不变

图示：
MHA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
      K1 K2 K3 K4 K5 K6 K7 K8
      V1 V2 V3 V4 V5 V6 V7 V8

GQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
      K1 K1 K2 K2 K3 K3 K4 K4  (4组K，每组被2个Q共享)
      V1 V1 V2 V2 V3 V3 V4 V4

MQA:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
      K1 K1 K1 K1 K1 K1 K1 K1  (1组K，被所有Q共享)
      V1 V1 V1 V1 V1 V1 V1 V1
```


## 3.8 从 PLM 到 LLM：规模的力量

### 3.8.1 什么是 LLM？

```
LLM = Large Language Model = 大语言模型

PLM 和 LLM 的区别：

PLM（预训练语言模型）：
- 参数量：百万~十亿级（BERT: 340M）
- 使用方式：预训练 + 微调
- 能力：特定任务表现好

LLM（大语言模型）：
- 参数量：百亿~万亿级（GPT-3: 175B）
- 使用方式：预训练 + 提示（Prompt）
- 能力：通用能力，涌现能力

分界线：一般认为 10B+ 参数的模型才算 LLM
```

### 3.8.2 Scaling Law（规模定律）

```
OpenAI 发现的重要规律：

模型性能 ∝ 模型大小 × 数据量 × 计算量

具体来说：
- 模型参数增加 10 倍，性能提升 X%
- 训练数据增加 10 倍，性能提升 Y%
- 计算量增加 10 倍，性能提升 Z%

这意味着：
- 只要有足够的数据和算力
- 模型越大，效果越好
- 没有明显的上限

这就是为什么大家都在"卷"模型大小：
GPT-3: 175B → GPT-4: ~1T（推测）
```

**Scaling Law 的数学形式**（Kaplan et al., 2020）：

OpenAI 发现，模型的测试损失 $L$ 与三个因素呈幂律关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

其中：
- $N$ = 模型参数量（非 Embedding 参数）
- $D$ = 训练数据量（token 数）
- $C$ = 计算量（FLOPs）
- $N_c, D_c, C_c$ 是常数

**Chinchilla Scaling Law**（Hoffmann et al., 2022）：

DeepMind 提出了更优的缩放策略：

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

关键结论：**模型参数和数据量应该同比例增长**

| 计算预算 | 最优参数量 | 最优数据量 |
|---------|-----------|-----------|
| GPT-3 级别 | 70B | 1.4T tokens |
| 10x 计算 | 220B | 4.4T tokens |

这解释了为什么 LLaMA-65B（1.4T tokens）效果接近 GPT-3-175B（300B tokens）。

### 3.8.3 涌现能力（Emergent Abilities）

```
涌现能力 = 小模型没有，大模型突然有的能力

例子：

1. Few-shot 学习
   - 小模型：给几个例子也学不会
   - 大模型：给几个例子就能学会新任务

2. 思维链（Chain-of-Thought）
   - 小模型：直接给答案，经常错
   - 大模型：可以一步步推理

3. 指令遵循
   - 小模型：不理解复杂指令
   - 大模型：可以理解并执行复杂指令

涌现的特点：
- 不是渐进提升，而是突然出现
- 像物理中的"相变"
- 这是 LLM 最神奇的地方
```

### 3.8.4 LLM 的三阶段训练

```
现代 LLM 的训练流程：

第一阶段：预训练（Pre-training）
- 数据：海量无标注文本（TB 级）
- 任务：预测下一个词（CLM）
- 目的：学习语言知识和世界知识
- 成本：最高（需要大量 GPU 和时间）

第二阶段：指令微调（Instruction Tuning / SFT）
- 数据：指令-回答对（万~百万条）
- 任务：学习遵循指令
- 目的：让模型学会"听话"
- 成本：中等

第三阶段：人类反馈强化学习（RLHF）
- 数据：人类偏好数据
- 任务：学习人类喜欢的回答方式
- 目的：让模型回答更有帮助、更安全
- 成本：较高（需要人工标注）

ChatGPT = GPT-3.5 + SFT + RLHF
```

### 3.8.5 SFT（指令微调）详解

SFT（Supervised Fine-Tuning）是让预训练模型学会"听话"的关键步骤。

```
为什么需要 SFT？

预训练后的模型：
- 只会"续写"文本
- 输入："中国的首都是" → 输出："北京。中国的首都是..."（继续写下去）
- 不会按照用户意图回答问题

SFT 后的模型：
- 学会理解指令并回答
- 输入："中国的首都是哪里？" → 输出："中国的首都是北京。"
- 知道什么时候该停止
```

**SFT 数据格式**：

```
SFT 数据 = 指令-回答对

格式1：简单问答
{
  "instruction": "中国的首都是哪里？",
  "output": "中国的首都是北京。"
}

格式2：带输入的任务
{
  "instruction": "将以下英文翻译成中文",
  "input": "Hello, how are you?",
  "output": "你好，你好吗？"
}

格式3：多轮对话
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "介绍一下北京"},
    {"role": "assistant", "content": "北京是中国的首都..."}
  ]
}
```

**SFT 的训练过程**：

```
SFT 本质上还是语言模型训练，但只计算回答部分的 Loss

输入：[指令] 中国的首都是哪里？ [回答] 北京。
Loss：  ----不计算----           --计算--

为什么只计算回答部分？
- 指令是用户输入，模型不需要"学会生成指令"
- 只需要学会"根据指令生成正确的回答"
```

**常见 SFT 数据集**：

| 数据集 | 规模 | 特点 |
|--------|------|------|
| Alpaca | 52K | Stanford 用 GPT-3.5 生成 |
| ShareGPT | 90K | 用户分享的 ChatGPT 对话 |
| BELLE | 1.5M | 中文指令数据 |
| Firefly | 1.1M | 中文多任务数据 |

### 3.8.6 RLHF（人类反馈强化学习）详解

RLHF 是让模型回答更符合人类偏好的关键技术。

```
为什么需要 RLHF？

SFT 后的模型问题：
- 可能生成有害内容
- 可能"一本正经地胡说八道"
- 回答风格可能不够友好
- 不知道什么是"好"的回答

RLHF 的目标：
- 让模型学会人类的偏好
- 生成更有帮助、更安全、更诚实的回答
```

**RLHF 的三个步骤**：

```
Step 1: 训练 Reward Model（奖励模型）

收集人类偏好数据：
问题："如何学习编程？"
回答A："多写代码，从简单项目开始..."（好）
回答B："编程很难，你可能学不会..."（差）
人类标注：A > B

训练奖励模型：
- 输入：问题 + 回答
- 输出：分数（越高越好）
- 目标：让 Score(A) > Score(B)


Step 2: 用 PPO 算法优化 LLM

PPO（Proximal Policy Optimization）是一种强化学习算法

训练循环：
1. LLM 生成回答
2. Reward Model 打分
3. 用 PPO 更新 LLM，让它生成更高分的回答
4. 重复

关键约束：
- 不能让 LLM 变化太大（KL 散度约束）
- 防止模型"作弊"（只生成高分但无意义的内容）


Step 3: 迭代优化

不断收集新的人类反馈，重复上述过程
```

**Reward Model 的数学形式**：

给定偏好数据 $(x, y_w, y_l)$，其中 $y_w$ 是人类偏好的回答，$y_l$ 是较差的回答。

Reward Model 的损失函数（Bradley-Terry 模型）：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

其中：
- $r_\phi(x, y)$ 是 Reward Model 对 $(x, y)$ 的打分
- $\sigma$ 是 sigmoid 函数

**PPO 优化目标**：

$$\mathcal{L}_{PPO} = \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot KL(\pi_\theta || \pi_{ref}) \right]$$

其中：
- $\pi_\theta$ 是当前策略（LLM）
- $\pi_{ref}$ 是参考策略（SFT 后的模型）
- $\beta$ 是 KL 惩罚系数，防止模型偏离太远

**RLHF 的图示**：

```
                    ┌─────────────┐
                    │  人类标注者  │
                    └──────┬──────┘
                           │ 标注偏好
                           ↓
┌─────────┐    生成回答   ┌─────────────┐
│   LLM   │ ───────────→ │ Reward Model│
└────┬────┘              └──────┬──────┘
     │                          │ 打分
     │    ←─────────────────────┘
     │         PPO 更新
     ↓
┌─────────┐
│ 更好的LLM│
└─────────┘
```

### 3.8.7 DPO：RLHF 的简化替代

DPO（Direct Preference Optimization）是 2023 年提出的新方法，可以跳过 Reward Model。

```
RLHF 的问题：
- 需要训练单独的 Reward Model
- PPO 算法复杂，训练不稳定
- 需要大量计算资源

DPO 的思路：
- 直接用偏好数据优化 LLM
- 不需要 Reward Model
- 不需要强化学习
- 训练更简单、更稳定

DPO 的做法：
- 输入：问题 + 好回答 + 差回答
- 目标：让 P(好回答) > P(差回答)
- 直接用交叉熵损失训练

对比：
RLHF: 偏好数据 → Reward Model → PPO → 更好的 LLM
DPO:  偏好数据 → 直接优化 → 更好的 LLM
```

**DPO 的数学推导**（Rafailov et al., 2023）：

DPO 的核心洞察：最优的 Reward Model 可以用策略本身表示：

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

将此代入 Bradley-Terry 模型，得到 DPO 损失：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

简化形式：

$$\mathcal{L}_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta (r_\theta(x, y_w) - r_\theta(x, y_l)) \right) \right]$$

其中隐式奖励 $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$

**DPO 的优势**：

| 对比项 | RLHF | DPO |
|--------|------|-----|
| 训练复杂度 | 高（需要 RM + PPO） | 低（直接优化） |
| 稳定性 | 较差 | 较好 |
| 计算资源 | 多 | 少 |
| 效果 | 好 | 接近 RLHF |
| 超参数 | 多（PPO 相关） | 少（主要是 β） |

目前很多开源模型（如 Zephyr、Neural Chat）都使用 DPO 进行对齐。

**论文引用**：
- RLHF: Ouyang et al., "Training language models to follow instructions with human feedback", 2022
- DPO: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023


## 3.9 Tokenizer（分词器）详解

Tokenizer 是 LLM 的"眼睛"，决定了模型如何"看"文本。

### 3.9.1 为什么需要 Tokenizer？

```
模型不能直接处理文本，需要转换成数字

文本 → Tokenizer → Token IDs → 模型

例子：
"Hello world" → ["Hello", "world"] → [15496, 995]
```

### 3.9.2 三种主流 Tokenizer

**1. BPE（Byte Pair Encoding）**

```
BPE 的核心思想：从字符开始，不断合并最频繁的字符对

训练过程：
初始词表：所有单个字符 ['a', 'b', 'c', ..., 'z', ' ', ...]

Step 1: 统计最频繁的字符对
  "l" + "o" 出现最多 → 合并成 "lo"
  
Step 2: 继续合并
  "lo" + "w" → "low"
  
Step 3: 重复直到词表达到目标大小

结果：
- 常见词：整词（"the", "and"）
- 罕见词：拆成子词（"unhappiness" → "un" + "happiness"）

优点：
- 平衡词表大小和覆盖率
- 能处理未见过的词
- GPT 系列使用
```

### 3.9.3 Tokenizer 对比

| Tokenizer | 使用模型 | 特点 |
|-----------|---------|------|
| BPE | GPT 系列 | 基于频率合并 |
| WordPiece | BERT | 基于概率合并，## 前缀 |
| SentencePiece | LLaMA, T5 | 语言无关，▁ 表示空格 |

### 3.9.4 词表大小的影响

```
词表大小是一个 trade-off：

词表太小（如 10K）：
- 优点：Embedding 层参数少
- 缺点：一个词要拆成很多 token，序列变长

词表太大（如 200K）：
- 优点：大多数词是整词，序列短
- 缺点：Embedding 层参数多，罕见词学不好

常见词表大小：
- GPT-2: 50K
- LLaMA: 32K
- LLaMA-3: 128K（支持更多语言）
- GPT-4: 100K+
```


## 3.10 KV Cache：推理加速的关键

KV Cache 是 LLM 推理优化的核心技术。

### 3.10.1 为什么需要 KV Cache？

```
LLM 生成文本是自回归的：

生成 "今天天气很好" 的过程：
Step 1: 输入 [BOS] → 输出 "今天"
Step 2: 输入 [BOS] 今天 → 输出 "天气"
Step 3: 输入 [BOS] 今天 天气 → 输出 "很"
Step 4: 输入 [BOS] 今天 天气 很 → 输出 "好"

问题：每一步都要重新计算之前所有 token 的 K 和 V！

Step 4 时：
- 要计算 [BOS] 的 K, V（第4次计算！）
- 要计算 "今天" 的 K, V（第3次计算！）
- 要计算 "天气" 的 K, V（第2次计算！）
- 要计算 "很" 的 K, V（第1次计算）

大量重复计算！
```

### 3.10.2 KV Cache 的原理

```
KV Cache 的思路：缓存已计算的 K 和 V

Step 1: 计算 [BOS] 的 K, V → 存入 Cache
Step 2: 从 Cache 取 [BOS] 的 K, V，计算 "今天" 的 K, V → 存入 Cache
Step 3: 从 Cache 取之前的 K, V，计算 "天气" 的 K, V → 存入 Cache
Step 4: 从 Cache 取之前的 K, V，计算 "很" 的 K, V → 存入 Cache

每个 token 的 K, V 只计算一次！

速度提升：
- 没有 KV Cache：O(n²) 计算量
- 有 KV Cache：O(n) 计算量
```

### 3.10.3 KV Cache 的内存占用

```
KV Cache 的代价：需要大量显存

计算公式：
KV Cache 大小 = 2 × batch_size × num_layers × seq_len × hidden_dim × 2bytes

例子（LLaMA-7B，序列长度 2048）：
= 2 × 1 × 32 × 2048 × 4096 × 2 bytes
= 1GB

问题：
- 序列越长，KV Cache 越大
- batch_size 越大，KV Cache 越大
- 这就是为什么长文本推理需要大显存

优化方法：
- GQA：减少 K, V 的数量
- 量化：用更少的 bit 存储
- PagedAttention：动态分配显存（vLLM）
```

### 3.10.4 KV Cache 图示

```
没有 KV Cache：

Step 1: [BOS] → 计算 K₁,V₁ → 输出 "今天"
Step 2: [BOS] 今天 → 计算 K₁,V₁,K₂,V₂ → 输出 "天气"
Step 3: [BOS] 今天 天气 → 计算 K₁,V₁,K₂,V₂,K₃,V₃ → 输出 "很"
                          ↑ 重复计算！


有 KV Cache：

Step 1: [BOS] → 计算 K₁,V₁ → Cache=[K₁,V₁] → 输出 "今天"
Step 2: 今天 → 计算 K₂,V₂ → Cache=[K₁,V₁,K₂,V₂] → 输出 "天气"
Step 3: 天气 → 计算 K₃,V₃ → Cache=[K₁,V₁,K₂,V₂,K₃,V₃] → 输出 "很"
        ↑ 只计算新 token！
```


## 3.11 中文大模型

### 3.11.1 主流中文 LLM

```
国内主要的开源/闭源 LLM：

开源模型：
┌─────────────┬──────────┬─────────────────────┐
│    模型      │   机构   │        特点          │
├─────────────┼──────────┼─────────────────────┤
│ ChatGLM     │   智谱   │ 首个中文开源对话模型   │
│ Qwen        │  阿里云  │ 效果好，多模态支持     │
│ Baichuan    │  百川    │ 中英双语，开源友好     │
│ InternLM    │ 上海AI Lab│ 书生浦语，长文本支持  │
│ Yi          │ 零一万物 │ 李开复团队，效果强     │
│ DeepSeek    │   幻方   │ 代码能力强，MoE架构   │
└─────────────┴──────────┴─────────────────────┘

闭源模型：
- 文心一言（百度）
- 通义千问（阿里）
- 星火大模型（讯飞）
- 混元（腾讯）
- 豆包（字节）
```

### 3.11.2 ChatGLM 系列

```
ChatGLM 是国内最早的开源中文对话模型

ChatGLM-6B (2023.03)：
- 参数：6B
- 特点：首个可在消费级显卡运行的中文模型
- 架构：GLM（General Language Model）

ChatGLM2-6B (2023.06)：
- 改进：更长上下文（32K）、更快推理
- 使用 Multi-Query Attention

ChatGLM3-6B (2023.10)：
- 改进：支持工具调用、代码执行
- 更好的对话能力

GLM-4 (2024)：
- 参数：更大
- 能力：接近 GPT-4
```

### 3.11.3 Qwen（通义千问）系列

```
Qwen 是阿里云开源的大模型，目前效果最好的中文开源模型之一

Qwen-7B/14B/72B (2023)：
- 多种规格可选
- 中英双语能力强

Qwen1.5 (2024.02)：
- 改进：更好的指令遵循
- 支持 32K 上下文

Qwen2 (2024.06)：
- 规格：0.5B ~ 72B
- 支持 128K 上下文
- 多语言支持（29种语言）

Qwen2.5 (2024.09)：
- 最新版本
- 代码、数学能力大幅提升
- 开源最强中文模型之一
```

### 3.11.4 中文 LLM 的挑战

```
中文 LLM 面临的特殊挑战：

1. 分词问题
   - 中文没有空格分隔
   - 需要专门的中文 Tokenizer
   - 词表要包含足够的中文字符

2. 训练数据
   - 高质量中文数据相对较少
   - 英文互联网数据远多于中文
   - 需要专门收集中文语料

3. 评测标准
   - 英文有 MMLU、HellaSwag 等标准
   - 中文评测标准还在发展中
   - C-Eval、CMMLU 等中文评测集

4. 文化适配
   - 中国特有的知识（历史、地理、文化）
   - 需要专门的中文知识注入
```


## 3.12 本章小结

### 3.12.1 三种架构对比

| 架构 | 代表模型 | 注意力方向 | 预训练任务 | 擅长任务 |
|------|---------|-----------|-----------|---------|
| Encoder-only | BERT | 双向 | MLM, NSP | 理解（分类、NER） |
| Decoder-only | GPT | 单向 | CLM | 生成（对话、续写） |
| Encoder-Decoder | T5 | 双向+单向 | Span Corruption | 转换（翻译、摘要） |

### 3.12.2 预训练任务对比

| 任务 | 全称 | 做法 | 使用模型 |
|------|------|------|---------|
| MLM | Masked Language Model | 遮蔽词，预测被遮蔽的词 | BERT |
| NSP | Next Sentence Prediction | 判断两句是否连续 | BERT |
| SOP | Sentence Order Prediction | 判断两句顺序是否正确 | ALBERT |
| CLM | Causal Language Model | 预测下一个词 | GPT |

### 3.12.3 重要模型时间线

```
2017.06  Transformer 发布
    ↓
2018.06  GPT-1 发布（OpenAI）
2018.10  BERT 发布（Google）← 预训练时代开始
    ↓
2019.02  GPT-2 发布
2019.07  RoBERTa 发布
2019.10  T5 发布
2019.10  ALBERT 发布
    ↓
2020.05  GPT-3 发布 ← LLM 时代开始
    ↓
2022.11  ChatGPT 发布 ← AI 热潮爆发
    ↓
2023.02  LLaMA 发布 ← 开源 LLM 时代
2023.03  GPT-4 发布
2023.07  LLaMA-2 发布
    ↓
2024.04  LLaMA-3 发布
```

### 3.12.4 核心概念速查

| 概念 | 解释 |
|------|------|
| 预训练 | 在海量无标注数据上学习通用知识 |
| 微调 | 在少量标注数据上适配具体任务 |
| MLM | 完形填空，预测被遮蔽的词 |
| CLM | 续写，预测下一个词 |
| Scaling Law | 模型越大、数据越多，效果越好 |
| 涌现能力 | 大模型突然具有的新能力 |
| Few-shot | 给几个例子就能学会新任务 |
| SFT | 指令微调，让模型学会遵循指令 |
| RLHF | 用人类反馈来优化模型 |
| DPO | RLHF 的简化替代，直接优化偏好 |
| Tokenizer | 分词器，将文本转换为 token |
| BPE | 一种分词算法，基于频率合并 |
| KV Cache | 缓存 K/V 向量，加速推理 |
| GQA | 分组查询注意力，减少 KV Cache |

### 3.12.5 下一章预告

第4章将学习**大语言模型（LLM）**：
- LLM 的能力和特点
- 训练流程详解（预训练、SFT、RLHF）
- Prompt Engineering
- LLM 的应用和局限

---

## 思考题

1. 为什么 BERT 用双向注意力，而 GPT 用单向注意力？各有什么优缺点？

2. MLM 和 CLM 哪个更适合预训练？为什么最终 CLM（GPT 路线）胜出？

3. 什么是涌现能力？为什么小模型没有而大模型有？

4. LLaMA 相比 GPT 有哪些技术改进？

5. 为什么说 ChatGPT 是 GPT-3.5 + SFT + RLHF？每个阶段的作用是什么？

6. RLHF 和 DPO 有什么区别？为什么 DPO 更简单？

7. 为什么需要 KV Cache？它如何加速推理？代价是什么？

8. BPE、WordPiece、SentencePiece 三种 Tokenizer 有什么区别？

---

## 推荐阅读

1. **BERT 论文**：《BERT: Pre-training of Deep Bidirectional Transformers》
2. **GPT-3 论文**：《Language Models are Few-Shot Learners》
3. **LLaMA 论文**：《LLaMA: Open and Efficient Foundation Language Models》
4. **Scaling Law 论文**：《Scaling Laws for Neural Language Models》

---
学习日期：2025年12月
