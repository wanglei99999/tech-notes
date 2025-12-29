# Chunking 分片策略详解

## 一、为什么需要分片

### 1.1 LLM 上下文长度限制

| 模型 | 上下文长度 |
|------|------------|
| GPT-3.5 | 4K / 16K tokens |
| GPT-4 | 8K / 32K / 128K tokens |
| Claude | 100K / 200K tokens |

即使是 128K 的上下文，也无法放入一本书（通常 50万+ tokens）。

### 1.2 检索粒度的权衡

**粒度太大**：
- 包含太多无关信息（噪声）
- Embedding 质量下降（长文本难以压缩到固定维度）
- 检索精度降低

**粒度太小**：
- 上下文不完整
- 需要检索更多 chunks
- 可能切断重要信息

### 1.3 Embedding 模型的限制

大多数 Embedding 模型有最大长度限制：

| 模型 | 最大长度 |
|------|----------|
| OpenAI text-embedding-3 | 8191 tokens |
| BGE-large | 512 tokens |
| E5-large | 512 tokens |

超过限制的文本会被截断，信息丢失。

---

## 二、分片的基本概念

### 2.1 Chunk 的定义

一个 Chunk 是文档的一个片段，包含：
- **文本内容**：实际的文字
- **元数据**：来源文档、位置、标题等
- **向量表示**：Embedding

### 2.2 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| chunk_size | 每个 chunk 的大小 | 200-2000 tokens |
| chunk_overlap | 相邻 chunk 的重叠部分 | 10-20% of chunk_size |
| separator | 分割符 | \n\n, \n, 空格 |

### 2.3 分片的目标

1. **语义完整性**：每个 chunk 应该是语义完整的单元
2. **大小适中**：不太大也不太小
3. **边界合理**：在自然边界处切分（段落、句子）

---

## 三、常见分片策略

### 3.1 固定长度分片

最简单的方法：按固定字符/token 数切分。

```python
def fixed_length_chunk(text, chunk_size=500, overlap=50):
    """
    固定长度分片
    
    Args:
        text: 输入文本
        chunk_size: 每个 chunk 的字符数
        overlap: 重叠字符数
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 下一个 chunk 从 overlap 位置开始
    
    return chunks

# 示例
text = "这是一段很长的文本..." * 100
chunks = fixed_length_chunk(text, chunk_size=500, overlap=50)
```

**优点**：
- 简单，易于实现
- chunk 大小均匀

**缺点**：
- 可能从句子/词语中间切断
- 不考虑语义边界

### 3.2 基于分隔符的分片

按自然分隔符（段落、句子）切分。

```python
def separator_based_chunk(text, separators=["\n\n", "\n", ". ", " "], 
                          chunk_size=500, overlap=50):
    """
    基于分隔符的分片
    
    优先使用高级分隔符（段落），如果 chunk 太大再用低级分隔符（句子）
    """
    def split_text(text, separator):
        if separator:
            return text.split(separator)
        else:
            return list(text)  # 按字符分割
    
    def merge_chunks(splits, separator, chunk_size, overlap):
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            if current_length + split_length > chunk_size and current_chunk:
                # 当前 chunk 已满，保存并开始新 chunk
                chunks.append(separator.join(current_chunk))
                
                # 保留 overlap 部分
                overlap_chunks = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= overlap:
                        overlap_chunks.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunks
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_length
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    # 递归分割
    chunks = [text]
    for separator in separators:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                splits = split_text(chunk, separator)
                merged = merge_chunks(splits, separator, chunk_size, overlap)
                new_chunks.extend(merged)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
    
    return chunks
```

**优点**：
- 尊重自然边界
- 语义更完整

**缺点**：
- chunk 大小不均匀
- 实现较复杂

### 3.3 递归字符分割（LangChain 默认）

LangChain 的 `RecursiveCharacterTextSplitter`：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)
```

**工作原理**：
1. 首先尝试按 `\n\n`（段落）分割
2. 如果某个片段仍然太大，按 `\n`（行）分割
3. 如果还太大，按空格分割
4. 最后按字符分割

### 3.4 语义分片

基于语义相似度在话题转换处切分。

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def semantic_chunk(text, model, threshold=0.5, min_chunk_size=100):
    """
    语义分片：在语义变化处切分
    
    Args:
        text: 输入文本
        model: Embedding 模型
        threshold: 相似度阈值，低于此值则切分
        min_chunk_size: 最小 chunk 大小
    """
    # 1. 分句
    sentences = text.split('. ')
    
    # 2. 计算每个句子的 embedding
    embeddings = model.encode(sentences)
    
    # 3. 计算相邻句子的相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(sim)
    
    # 4. 在相似度低的地方切分
    chunks = []
    current_chunk = [sentences[0]]
    
    for i, sim in enumerate(similarities):
        if sim < threshold and len('. '.join(current_chunk)) >= min_chunk_size:
            # 相似度低，切分
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

# 使用
model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = semantic_chunk(text, model, threshold=0.5)
```

**优点**：
- 语义最完整
- 在话题转换处自然切分

**缺点**：
- 计算成本高（需要计算所有句子的 embedding）
- 依赖 embedding 模型质量

### 3.5 基于文档结构的分片

利用文档的结构信息（标题、章节）。

```python
import re

def structure_based_chunk(markdown_text, max_chunk_size=1000):
    """
    基于 Markdown 结构分片
    """
    # 按标题分割
    pattern = r'^(#{1,6})\s+(.+)$'
    
    chunks = []
    current_chunk = []
    current_headers = []
    
    for line in markdown_text.split('\n'):
        match = re.match(pattern, line)
        
        if match:
            level = len(match.group(1))
            title = match.group(2)
            
            # 遇到新标题，保存当前 chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > 0:
                    chunks.append({
                        'text': chunk_text,
                        'headers': current_headers.copy()
                    })
            
            # 更新标题层级
            current_headers = current_headers[:level-1] + [title]
            current_chunk = [line]
        else:
            current_chunk.append(line)
            
            # 如果当前 chunk 太大，强制切分
            if len('\n'.join(current_chunk)) > max_chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'headers': current_headers.copy()
                })
                current_chunk = []
    
    # 保存最后一个 chunk
    if current_chunk:
        chunks.append({
            'text': '\n'.join(current_chunk),
            'headers': current_headers.copy()
        })
    
    return chunks
```

**优点**：
- 保留文档结构信息
- 可以利用标题作为元数据

**缺点**：
- 依赖文档格式
- 不适用于无结构文本

---

## 四、Chunk Size 的选择

### 4.1 影响因素

| 因素 | 小 chunk | 大 chunk |
|------|----------|----------|
| 检索精度 | 高（更精确匹配） | 低（可能包含噪声） |
| 上下文完整性 | 低（信息可能不完整） | 高（上下文丰富） |
| 检索数量 | 需要更多 chunks | 需要更少 chunks |
| Embedding 质量 | 高（短文本更容易编码） | 低（长文本压缩损失） |

### 4.2 经验值

| 场景 | 建议 chunk_size | 理由 |
|------|-----------------|------|
| 问答系统 | 200-500 tokens | 精确匹配问题 |
| 文档摘要 | 500-1000 tokens | 需要更多上下文 |
| 代码检索 | 按函数/类分割 | 保持代码完整性 |
| 法律文档 | 按条款分割 | 保持条款完整性 |

### 4.3 实验方法

```python
def evaluate_chunk_size(documents, queries, ground_truth, chunk_sizes):
    """
    评估不同 chunk size 的效果
    """
    results = {}
    
    for chunk_size in chunk_sizes:
        # 分片
        chunks = chunk_documents(documents, chunk_size)
        
        # 建立索引
        index = build_index(chunks)
        
        # 检索评估
        recall_at_k = evaluate_retrieval(index, queries, ground_truth, k=10)
        
        results[chunk_size] = recall_at_k
    
    return results

# 测试不同 chunk size
chunk_sizes = [100, 200, 500, 1000, 2000]
results = evaluate_chunk_size(docs, queries, labels, chunk_sizes)

# 选择最佳 chunk size
best_size = max(results, key=results.get)
```

---

## 五、Overlap 的作用

### 5.1 为什么需要 Overlap

```
无 Overlap：
  Chunk 1: [句子1, 句子2, 句子3]
  Chunk 2: [句子4, 句子5, 句子6]
  
  问题：如果答案跨越句子3和句子4，两个 chunk 都不完整

有 Overlap：
  Chunk 1: [句子1, 句子2, 句子3, 句子4]
  Chunk 2: [句子3, 句子4, 句子5, 句子6]
  
  句子3和句子4在两个 chunk 中都有，提高召回率
```

### 5.2 Overlap 的大小

**经验值**：Overlap = 10-20% of chunk_size

| chunk_size | 建议 overlap |
|------------|--------------|
| 200 | 20-40 |
| 500 | 50-100 |
| 1000 | 100-200 |

### 5.3 Overlap 的权衡

**Overlap 太小**：
- 边界信息可能丢失
- 召回率下降

**Overlap 太大**：
- 存储冗余
- 检索到重复内容
- 索引变大

---

## 六、高级分片技术

### 6.1 父子 Chunk（Parent-Child Chunking）

```
父 Chunk（大）：用于提供上下文
    ├── 子 Chunk 1（小）：用于检索
    ├── 子 Chunk 2（小）：用于检索
    └── 子 Chunk 3（小）：用于检索
```

**工作流程**：
1. 用小 chunk 做检索（精确匹配）
2. 检索到后，返回对应的父 chunk（完整上下文）

```python
def parent_child_chunk(text, parent_size=2000, child_size=400):
    """
    父子分片
    """
    # 先分成大的父 chunks
    parent_chunks = fixed_length_chunk(text, parent_size, overlap=0)
    
    result = []
    for i, parent in enumerate(parent_chunks):
        # 每个父 chunk 再分成小的子 chunks
        child_chunks = fixed_length_chunk(parent, child_size, overlap=50)
        
        for j, child in enumerate(child_chunks):
            result.append({
                'child_text': child,
                'parent_text': parent,
                'parent_id': i,
                'child_id': j
            })
    
    return result

# 检索时
def retrieve_with_parent(query, index, top_k=5):
    # 用子 chunk 检索
    child_results = index.search(query, top_k)
    
    # 返回父 chunk
    parent_texts = [r['parent_text'] for r in child_results]
    
    # 去重
    unique_parents = list(dict.fromkeys(parent_texts))
    
    return unique_parents
```

### 6.2 滑动窗口 + 句子边界

```python
import nltk

def sliding_window_sentence_chunk(text, window_size=5, stride=2):
    """
    滑动窗口分片，但在句子边界处切分
    
    Args:
        window_size: 每个 chunk 包含的句子数
        stride: 滑动步长（句子数）
    """
    # 分句
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    for i in range(0, len(sentences), stride):
        chunk_sentences = sentences[i:i + window_size]
        if chunk_sentences:
            chunks.append(' '.join(chunk_sentences))
    
    return chunks
```

### 6.3 基于 Token 的精确分片

使用 tokenizer 确保不超过模型限制：

```python
from transformers import AutoTokenizer

def token_based_chunk(text, tokenizer, max_tokens=512, overlap_tokens=50):
    """
    基于 token 的精确分片
    """
    # 编码整个文本
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        
        # 解码回文本
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        start = end - overlap_tokens
    
    return chunks

# 使用
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
chunks = token_based_chunk(text, tokenizer, max_tokens=512)
```

### 6.4 多粒度索引

同时建立多个粒度的索引：

```python
class MultiGranularityIndex:
    def __init__(self, text):
        self.sentence_chunks = sentence_chunk(text)
        self.paragraph_chunks = paragraph_chunk(text)
        self.document_chunks = [text]
        
        self.sentence_index = build_index(self.sentence_chunks)
        self.paragraph_index = build_index(self.paragraph_chunks)
    
    def search(self, query, granularity='paragraph'):
        if granularity == 'sentence':
            return self.sentence_index.search(query)
        elif granularity == 'paragraph':
            return self.paragraph_index.search(query)
        else:
            return self.document_chunks
```

---

## 七、特殊文档类型的分片

### 7.1 代码文件

```python
import ast

def chunk_python_code(code):
    """
    按函数/类分割 Python 代码
    """
    tree = ast.parse(code)
    chunks = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # 获取函数/类的源代码
            start_line = node.lineno - 1
            end_line = node.end_lineno
            
            lines = code.split('\n')
            chunk = '\n'.join(lines[start_line:end_line])
            
            chunks.append({
                'text': chunk,
                'type': type(node).__name__,
                'name': node.name
            })
    
    return chunks
```

### 7.2 表格数据

```python
def chunk_table(table_text, rows_per_chunk=10):
    """
    分割表格，保留表头
    """
    lines = table_text.strip().split('\n')
    header = lines[0]  # 假设第一行是表头
    
    chunks = []
    for i in range(1, len(lines), rows_per_chunk):
        chunk_lines = [header] + lines[i:i + rows_per_chunk]
        chunks.append('\n'.join(chunk_lines))
    
    return chunks
```

### 7.3 对话数据

```python
def chunk_conversation(messages, turns_per_chunk=5):
    """
    按对话轮次分割
    """
    chunks = []
    
    for i in range(0, len(messages), turns_per_chunk):
        chunk_messages = messages[i:i + turns_per_chunk]
        chunk_text = '\n'.join([
            f"{m['role']}: {m['content']}" 
            for m in chunk_messages
        ])
        chunks.append(chunk_text)
    
    return chunks
```

---

## 八、分片质量评估

### 8.1 评估指标

1. **语义完整性**：chunk 是否包含完整的语义单元
2. **边界合理性**：是否在自然边界处切分
3. **检索效果**：使用该分片策略的检索 Recall@K

### 8.2 评估方法

```python
def evaluate_chunking_strategy(documents, queries, labels, chunking_fn):
    """
    评估分片策略
    """
    # 分片
    all_chunks = []
    chunk_to_doc = {}
    for doc_id, doc in enumerate(documents):
        chunks = chunking_fn(doc)
        for chunk in chunks:
            chunk_to_doc[len(all_chunks)] = doc_id
            all_chunks.append(chunk)
    
    # 建立索引
    index = build_index(all_chunks)
    
    # 评估检索效果
    hits = 0
    for query, relevant_docs in zip(queries, labels):
        results = index.search(query, top_k=10)
        retrieved_docs = set(chunk_to_doc[r] for r in results)
        if any(d in retrieved_docs for d in relevant_docs):
            hits += 1
    
    recall = hits / len(queries)
    
    # 评估 chunk 大小分布
    sizes = [len(c) for c in all_chunks]
    avg_size = np.mean(sizes)
    std_size = np.std(sizes)
    
    return {
        'recall@10': recall,
        'avg_chunk_size': avg_size,
        'std_chunk_size': std_size,
        'num_chunks': len(all_chunks)
    }
```

---

## 九、实践建议

### 9.1 选择策略的决策树

```
文档类型？
├── 结构化（Markdown/HTML）→ 基于结构分片
├── 代码 → 按函数/类分片
├── 对话 → 按轮次分片
└── 纯文本
    ├── 需要高精度 → 语义分片
    └── 需要高效率 → 递归字符分片
```

### 9.2 参数调优流程

1. 从默认参数开始（chunk_size=500, overlap=50）
2. 在验证集上评估检索效果
3. 尝试不同的 chunk_size（200, 500, 1000, 2000）
4. 选择 Recall@K 最高的配置
5. 微调 overlap

### 9.3 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 检索不到相关内容 | chunk 太大，噪声多 | 减小 chunk_size |
| 检索到的内容不完整 | chunk 太小 | 增大 chunk_size 或使用父子分片 |
| 边界处信息丢失 | overlap 太小 | 增大 overlap |
| 索引太大 | chunk 太多 | 增大 chunk_size |
