# Bi-encoder 与 Cross-encoder 详解

## 一、问题背景

### 1.1 文本匹配任务

给定两段文本 $(A, B)$，判断它们的关系：
- **语义相似度**：两段文本是否表达相似的意思
- **问答匹配**：问题和答案是否匹配
- **自然语言推理**：前提是否蕴含假设

### 1.2 两种建模方式

**表示学习（Representation Learning）**：
- 将文本编码为向量
- 通过向量运算计算相似度
- 代表：Bi-encoder

**交互学习（Interaction Learning）**：
- 让两段文本充分交互
- 直接输出匹配分数
- 代表：Cross-encoder

---

## 二、Bi-encoder（双塔模型）

### 2.1 架构

```
Text A: "什么是机器学习"          Text B: "机器学习是AI的分支"
         ↓                                ↓
    [Encoder]                        [Encoder]
    (共享参数)                        (共享参数)
         ↓                                ↓
    向量 a ∈ ℝᵈ                      向量 b ∈ ℝᵈ
         ↓_____________↓_____________↓
                  相似度计算
              sim(a, b) = a·b / (||a||·||b||)
```

### 2.2 数学形式

设 Encoder 为 $f_\theta$，则：

$$\mathbf{a} = f_\theta(A)$$
$$\mathbf{b} = f_\theta(B)$$
$$\text{score}(A, B) = \text{sim}(\mathbf{a}, \mathbf{b})$$

常用相似度函数：
- 余弦相似度：$\frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$
- 点积：$\mathbf{a} \cdot \mathbf{b}$
- 欧氏距离：$-||\mathbf{a} - \mathbf{b}||$

### 2.3 Encoder 的实现

#### 基于 BERT

```python
import torch
from transformers import BertModel, BertTokenizer

class BiEncoder(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        
    def encode(self, input_ids, attention_mask):
        """
        编码单个文本
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 方式1：使用 [CLS] token
        # cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 方式2：Mean Pooling（通常更好）
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        sum_embeddings = torch.sum(
            token_embeddings * attention_mask_expanded, dim=1
        )
        sum_mask = attention_mask_expanded.sum(dim=1)
        mean_embeddings = sum_embeddings / sum_mask
        
        return mean_embeddings
    
    def forward(self, query_ids, query_mask, doc_ids, doc_mask):
        """
        计算 query 和 doc 的相似度
        """
        query_emb = self.encode(query_ids, query_mask)
        doc_emb = self.encode(doc_ids, doc_mask)
        
        # 归一化
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        doc_emb = torch.nn.functional.normalize(doc_emb, p=2, dim=1)
        
        # 余弦相似度
        scores = torch.matmul(query_emb, doc_emb.T)
        
        return scores
```

#### Pooling 策略对比

| 策略 | 公式 | 特点 |
|------|------|------|
| [CLS] | $\mathbf{h}_{[CLS]}$ | 简单，但 [CLS] 可能不是最优表示 |
| Mean | $\frac{1}{n}\sum_i \mathbf{h}_i$ | 考虑所有 token，通常效果更好 |
| Max | $\max_i \mathbf{h}_i$ | 捕捉最显著的特征 |
| Weighted Mean | $\sum_i w_i \mathbf{h}_i$ | 可学习的权重 |

### 2.4 训练目标

#### 对比学习损失（InfoNCE）

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+)/\tau)}{\sum_{d \in \mathcal{D}} \exp(\text{sim}(\mathbf{q}, \mathbf{d})/\tau)}$$

其中：
- $\mathbf{q}$：query 向量
- $\mathbf{d}^+$：正样本文档向量
- $\mathcal{D}$：所有候选文档（正样本 + 负样本）
- $\tau$：温度参数

#### In-batch Negatives

利用同一 batch 内的其他样本作为负样本：

```python
def compute_loss(query_embs, doc_embs, temperature=0.05):
    """
    query_embs: [batch_size, dim]
    doc_embs: [batch_size, dim]  # 每个 query 对应一个正样本 doc
    """
    # 计算所有 query-doc 对的相似度
    # scores[i][j] = sim(query_i, doc_j)
    scores = torch.matmul(query_embs, doc_embs.T) / temperature
    
    # 对角线是正样本
    labels = torch.arange(scores.size(0), device=scores.device)
    
    # 交叉熵损失
    loss = torch.nn.functional.cross_entropy(scores, labels)
    
    return loss
```

#### Hard Negative Mining

简单负样本太容易区分，需要挖掘难负样本：

```python
def mine_hard_negatives(query, positive_doc, candidate_docs, model, top_k=10):
    """
    从候选文档中挖掘 hard negatives
    """
    query_emb = model.encode(query)
    
    # 计算与所有候选的相似度
    scores = []
    for doc in candidate_docs:
        if doc == positive_doc:
            continue
        doc_emb = model.encode(doc)
        score = cosine_similarity(query_emb, doc_emb)
        scores.append((doc, score))
    
    # 选择相似度最高的非正样本作为 hard negatives
    scores.sort(key=lambda x: -x[1])
    hard_negatives = [doc for doc, _ in scores[:top_k]]
    
    return hard_negatives
```

### 2.5 优缺点

**优点**：
- 文档向量可以**离线预计算**并存储
- 查询时只需计算 query 向量，然后做向量检索
- 支持**大规模检索**（百万、千万级）
- 检索速度快（毫秒级）

**缺点**：
- query 和 doc **独立编码**，无法捕捉细粒度交互
- 对于需要深度理解的任务，精度不如 Cross-encoder

---

## 三、Cross-encoder（交叉编码器）

### 3.1 架构

```
Text A: "什么是机器学习"    Text B: "机器学习是AI的分支"
              ↓                        ↓
         [SEP] 拼接 [SEP]
              ↓
"[CLS] 什么是机器学习 [SEP] 机器学习是AI的分支 [SEP]"
              ↓
         [Encoder]
              ↓
      [CLS] 向量 → Linear → 相关性分数
```

### 3.2 数学形式

$$\text{score}(A, B) = \sigma(W \cdot f_\theta([CLS; A; SEP; B; SEP]) + b)$$

其中：
- $f_\theta$：Transformer encoder
- $W, b$：分类头参数
- $\sigma$：sigmoid 函数（二分类）或 softmax（多分类）

### 3.3 实现

```python
import torch
from transformers import BertModel, BertTokenizer

class CrossEncoder(torch.nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=1):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(
            self.encoder.config.hidden_size, 
            num_labels
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        输入是拼接后的 [CLS] A [SEP] B [SEP]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用 [CLS] token 的表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict(self, text_a, text_b, tokenizer, device):
        """
        预测两个文本的相关性分数
        """
        # 拼接并编码
        inputs = tokenizer(
            text_a, text_b,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            logits = self.forward(**inputs)
            scores = torch.sigmoid(logits).squeeze()
        
        return scores.cpu().numpy()
```

### 3.4 训练目标

#### 二分类（相关/不相关）

```python
def compute_loss(logits, labels):
    """
    logits: [batch_size, 1]
    labels: [batch_size] (0 或 1)
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(), 
        labels.float()
    )
    return loss
```

#### 回归（相似度分数）

```python
def compute_loss(logits, labels):
    """
    logits: [batch_size, 1]
    labels: [batch_size] (0-1 之间的相似度)
    """
    predictions = torch.sigmoid(logits.squeeze())
    loss = torch.nn.functional.mse_loss(predictions, labels)
    return loss
```

#### Pairwise Ranking

```python
def compute_pairwise_loss(pos_scores, neg_scores, margin=1.0):
    """
    正样本分数应该比负样本高出 margin
    """
    loss = torch.nn.functional.relu(margin - pos_scores + neg_scores)
    return loss.mean()
```

### 3.5 为什么 Cross-encoder 更准确？

#### 注意力机制的交互

在 Cross-encoder 中，Text A 和 Text B 的 token 可以相互 attend：

```
Self-Attention 矩阵：

              [CLS] 什 么 是 机 器 学 习 [SEP] 机 器 学 习 是 AI 的 分 支 [SEP]
[CLS]           ✓   ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓    ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓
什              ✓   ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓    ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓
...
机(Text A)      ✓   ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓    ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓   ✓
                                              ↑
                                    可以 attend 到 Text B 的 "机器学习"
```

这种交互让模型能够：
1. 对齐两段文本中的相同/相似概念
2. 理解上下文依赖的语义
3. 捕捉细粒度的匹配信号

#### Bi-encoder 的局限

Bi-encoder 中，Text A 和 Text B 独立编码：

```
Text A 的 Self-Attention：只能看到 Text A 内部
Text B 的 Self-Attention：只能看到 Text B 内部
```

无法捕捉跨文本的交互。

#### 示例对比

```
Query: "苹果公司市值"
Doc1: "苹果公司市值突破3万亿美元"  ← 高度相关
Doc2: "苹果富含维生素，市值..."    ← 不太相关

Bi-encoder 的问题：
- "苹果" 和 "市值" 在两个文档中都出现
- 独立编码时，无法区分 "苹果公司" vs "苹果(水果)"
- 可能给 Doc2 较高的分数

Cross-encoder 的优势：
- 可以看到 Query 中的 "公司" 和 Doc1 中的 "公司" 对应
- 可以理解 Doc2 中的 "苹果" 和 "市值" 是不相关的两个话题
- 能正确判断 Doc1 更相关
```

### 3.6 优缺点

**优点**：
- 充分的文本交互，**精度高**
- 适合需要深度理解的任务

**缺点**：
- 无法预计算文档表示
- 每个 query-doc 对都需要**实时计算**
- 不适合大规模检索（太慢）

---

## 四、复杂度对比

### 4.1 时间复杂度

设：
- $n$：文档数量
- $L$：文本平均长度
- $d$：隐藏层维度
- Transformer 的复杂度：$O(L^2 \cdot d)$

| 操作 | Bi-encoder | Cross-encoder |
|------|------------|---------------|
| 索引 n 个文档 | $O(n \cdot L^2 \cdot d)$ | 不支持预索引 |
| 查询 1 个 query | $O(L^2 \cdot d + n \cdot d)$ | $O(n \cdot (2L)^2 \cdot d)$ |

**具体数字**：
- 10000 个文档，每个 100 token
- Bi-encoder 查询：~10ms（向量检索）
- Cross-encoder 查询：~100s（10000 次前向传播）

### 4.2 空间复杂度

| 存储 | Bi-encoder | Cross-encoder |
|------|------------|---------------|
| 模型参数 | $O(|θ|)$ | $O(|θ|)$ |
| 文档索引 | $O(n \cdot d)$ | 无 |

---

## 五、实际应用：两阶段检索

### 5.1 架构

```
Query
  ↓
[Bi-encoder] ← 向量数据库（百万文档）
  ↓
Top-K 候选（如 100 个）
  ↓
[Cross-encoder]
  ↓
Top-N 结果（如 10 个）
  ↓
LLM 生成
```

### 5.2 实现

```python
class TwoStageRetriever:
    def __init__(self, bi_encoder, cross_encoder, vector_db):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.vector_db = vector_db
    
    def retrieve(self, query, top_k=100, top_n=10):
        """
        两阶段检索
        """
        # 阶段1：Bi-encoder 召回
        query_embedding = self.bi_encoder.encode(query)
        candidates = self.vector_db.search(query_embedding, top_k)
        
        # 阶段2：Cross-encoder 重排
        scores = []
        for doc_id, doc_text in candidates:
            score = self.cross_encoder.predict(query, doc_text)
            scores.append((doc_id, doc_text, score))
        
        # 按分数排序
        scores.sort(key=lambda x: -x[2])
        
        return scores[:top_n]
```

### 5.3 参数选择

| 参数 | 建议值 | 考虑因素 |
|------|--------|----------|
| top_k（召回数） | 50-200 | 召回率 vs 重排成本 |
| top_n（最终结果） | 5-20 | 下游任务需求 |

**权衡**：
- top_k 太小：可能漏掉相关文档
- top_k 太大：Cross-encoder 计算成本高

---

## 六、模型选择与训练

### 6.1 预训练模型

| 模型 | 类型 | 特点 |
|------|------|------|
| Sentence-BERT | Bi-encoder | 经典，效果稳定 |
| BGE | Bi-encoder | 中文效果好 |
| E5 | Bi-encoder | 多语言，MTEB 领先 |
| cross-encoder/ms-marco | Cross-encoder | 专门为重排训练 |

### 6.2 训练数据

**Bi-encoder**：
- 大规模弱监督数据（搜索日志、点击数据）
- 高质量标注数据（问答对、NLI）

**Cross-encoder**：
- 需要更高质量的标注数据
- 可以用 Bi-encoder 的 hard negatives

### 6.3 蒸馏

用 Cross-encoder 指导 Bi-encoder 训练：

```python
def distillation_loss(bi_scores, cross_scores, temperature=1.0):
    """
    让 Bi-encoder 的分数分布接近 Cross-encoder
    """
    bi_probs = torch.softmax(bi_scores / temperature, dim=-1)
    cross_probs = torch.softmax(cross_scores / temperature, dim=-1)
    
    loss = torch.nn.functional.kl_div(
        bi_probs.log(), 
        cross_probs, 
        reduction='batchmean'
    )
    
    return loss
```

---

## 七、进阶话题

### 7.1 Late Interaction（ColBERT）

介于 Bi-encoder 和 Cross-encoder 之间：

```
Query: [q1, q2, q3]  →  Encoder  →  [v_q1, v_q2, v_q3]
Doc:   [d1, d2, d3, d4]  →  Encoder  →  [v_d1, v_d2, v_d3, v_d4]

MaxSim:
score = Σ_i max_j (v_qi · v_dj)
```

- 保留 token 级别的向量
- 通过 MaxSim 实现轻量级交互
- 比 Bi-encoder 准，比 Cross-encoder 快

### 7.2 Poly-encoder

```
Query → Encoder → m 个 context codes [c1, ..., cm]
Doc → Encoder → doc embedding d

Attention:
weights = softmax(c · d)
query_emb = Σ weights_i * c_i

score = query_emb · d
```

- 使用少量 context codes 表示 query
- 比 Bi-encoder 更有表达力
- 比 Cross-encoder 更快

### 7.3 选择建议

| 场景 | 推荐方案 |
|------|----------|
| 大规模检索（>100万） | Bi-encoder |
| 高精度重排 | Cross-encoder |
| 平衡速度和精度 | ColBERT / Poly-encoder |
| 实时性要求高 | Bi-encoder + 轻量 Cross-encoder |
