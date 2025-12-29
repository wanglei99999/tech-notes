# Query 改写与扩展详解

## 一、为什么需要 Query 改写

### 1.1 用户 Query 的问题

用户输入的查询往往存在以下问题：

| 问题类型 | 示例 | 影响 |
|----------|------|------|
| 太短 | "LLM" | 信息不足，检索结果不精确 |
| 口语化 | "咋用 Python 爬虫" | 与文档风格不匹配 |
| 有歧义 | "苹果" | 水果？公司？ |
| 拼写错误 | "mechine learning" | 无法匹配正确词汇 |
| 复杂问题 | "比较 A 和 B 的优缺点" | 需要拆解 |

### 1.2 Query-Document 不匹配

**词汇鸿沟（Lexical Gap）**：
```
Query: "如何提高代码运行速度"
Document: "性能优化技巧：减少时间复杂度..."

问题：Query 用 "运行速度"，Document 用 "性能优化"
BM25 无法匹配，向量检索可能也不够精确
```

**风格差异**：
```
Query（口语）: "Python 咋读文件"
Document（书面）: "Python 文件 I/O 操作指南"
```

### 1.3 Query 改写的目标

1. **扩展语义**：添加同义词、相关词
2. **消除歧义**：明确查询意图
3. **风格转换**：口语 → 书面语
4. **问题分解**：复杂问题 → 多个简单问题

---

## 二、Query 扩展（Query Expansion）

### 2.1 同义词扩展

使用同义词词典或词向量找到相似词：

```python
from gensim.models import KeyedVectors

def synonym_expansion(query, word_vectors, top_k=3):
    """
    基于词向量的同义词扩展
    """
    words = query.split()
    expanded_words = []
    
    for word in words:
        expanded_words.append(word)
        
        if word in word_vectors:
            # 找到最相似的词
            similar = word_vectors.most_similar(word, topn=top_k)
            for sim_word, score in similar:
                if score > 0.7:  # 相似度阈值
                    expanded_words.append(sim_word)
    
    return ' '.join(expanded_words)

# 示例
# Query: "机器学习"
# 扩展后: "机器学习 深度学习 人工智能 ML"
```

### 2.2 基于 LLM 的扩展

```python
def llm_query_expansion(query, llm):
    """
    使用 LLM 生成相关词
    """
    prompt = f"""给定查询："{query}"

请生成 5 个与该查询语义相关的词或短语，用于扩展搜索。
只输出词语，用逗号分隔。"""

    response = llm.generate(prompt)
    expanded_terms = response.strip().split(',')
    
    return query + ' ' + ' '.join(expanded_terms)

# 示例
# Query: "Python 性能优化"
# LLM 输出: "代码加速, 运行效率, 时间复杂度, 内存优化, profiling"
# 扩展后: "Python 性能优化 代码加速 运行效率 时间复杂度 内存优化 profiling"
```

### 2.3 伪相关反馈（Pseudo Relevance Feedback）

1. 先用原始 Query 检索
2. 假设 top-k 结果是相关的
3. 从这些结果中提取关键词扩展 Query

```python
def pseudo_relevance_feedback(query, retriever, top_k=3, expand_terms=5):
    """
    伪相关反馈
    """
    # 1. 初始检索
    initial_results = retriever.search(query, top_k=top_k)
    
    # 2. 从结果中提取关键词
    all_text = ' '.join([doc['text'] for doc in initial_results])
    keywords = extract_keywords(all_text, top_n=expand_terms)
    
    # 3. 扩展 Query
    expanded_query = query + ' ' + ' '.join(keywords)
    
    return expanded_query

def extract_keywords(text, top_n=5):
    """
    使用 TF-IDF 提取关键词
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=top_n)
    vectorizer.fit([text])
    
    return vectorizer.get_feature_names_out().tolist()
```

---

## 三、HyDE（Hypothetical Document Embeddings）

### 3.1 核心思想

让 LLM 先生成一个"假想答案"，用假想答案的 Embedding 去检索。

**直觉**：假想答案的风格更接近文档，比短 Query 的 Embedding 质量更高。

### 3.2 工作流程

```
Query: "量子计算的优势是什么"
        ↓
      [LLM 生成假想答案]
        ↓
假想答案: "量子计算相比经典计算具有指数级加速优势，
         特别是在因数分解、优化问题、量子模拟等领域。
         量子比特的叠加态和纠缠特性使其能够并行处理
         大量计算任务..."
        ↓
    [Embedding]
        ↓
    向量检索
        ↓
    真实文档
```

### 3.3 实现

```python
class HyDE:
    def __init__(self, llm, embedding_model, retriever):
        self.llm = llm
        self.embedding_model = embedding_model
        self.retriever = retriever
    
    def generate_hypothetical_document(self, query):
        """
        生成假想文档
        """
        prompt = f"""请回答以下问题，提供详细的解释：

问题：{query}

回答："""
        
        hypothetical_doc = self.llm.generate(prompt, max_tokens=256)
        return hypothetical_doc
    
    def search(self, query, top_k=5):
        """
        HyDE 检索
        """
        # 1. 生成假想文档
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # 2. 编码假想文档
        hyde_embedding = self.embedding_model.encode(hypothetical_doc)
        
        # 3. 用假想文档的 embedding 检索
        results = self.retriever.search_by_vector(hyde_embedding, top_k=top_k)
        
        return results
    
    def search_with_fusion(self, query, top_k=5, alpha=0.5):
        """
        融合原始 Query 和 HyDE 的结果
        """
        # 原始 Query 检索
        query_embedding = self.embedding_model.encode(query)
        query_results = self.retriever.search_by_vector(query_embedding, top_k=top_k)
        
        # HyDE 检索
        hyde_results = self.search(query, top_k=top_k)
        
        # 融合结果（RRF）
        fused_results = reciprocal_rank_fusion([query_results, hyde_results])
        
        return fused_results[:top_k]
```

### 3.4 HyDE 的优缺点

**优点**：
- 假想文档风格接近真实文档
- 对短 Query 效果提升明显
- 不需要额外训练

**缺点**：
- 需要额外的 LLM 调用（延迟增加）
- LLM 可能生成错误信息
- 对事实性问题可能引入偏差

### 3.5 改进：多假想文档

生成多个假想文档，取平均或投票：

```python
def multi_hyde(query, llm, embedding_model, retriever, n_hypotheses=3):
    """
    生成多个假想文档
    """
    hypothetical_docs = []
    for i in range(n_hypotheses):
        # 使用不同的 prompt 或 temperature
        doc = generate_hypothetical_document(query, llm, temperature=0.7 + i*0.1)
        hypothetical_docs.append(doc)
    
    # 方法1：平均 embedding
    embeddings = [embedding_model.encode(doc) for doc in hypothetical_docs]
    avg_embedding = np.mean(embeddings, axis=0)
    
    # 方法2：分别检索后融合
    all_results = []
    for doc in hypothetical_docs:
        emb = embedding_model.encode(doc)
        results = retriever.search_by_vector(emb, top_k=10)
        all_results.append(results)
    
    fused = reciprocal_rank_fusion(all_results)
    
    return fused
```

---

## 四、Query Decomposition（问题分解）

### 4.1 适用场景

复杂问题需要拆解为多个子问题：

```
复杂问题: "比较 Python 和 Java 在机器学习领域的优劣"

子问题:
1. "Python 在机器学习领域的优势"
2. "Python 在机器学习领域的劣势"
3. "Java 在机器学习领域的优势"
4. "Java 在机器学习领域的劣势"
```

### 4.2 实现

```python
class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, query):
        """
        将复杂问题分解为子问题
        """
        prompt = f"""将以下复杂问题分解为多个简单的子问题，每个子问题应该可以独立回答。

复杂问题：{query}

请输出子问题列表，每行一个子问题："""

        response = self.llm.generate(prompt)
        sub_queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        return sub_queries
    
    def search_and_merge(self, query, retriever, top_k=5):
        """
        分解查询，分别检索，合并结果
        """
        # 1. 分解问题
        sub_queries = self.decompose(query)
        
        # 2. 分别检索
        all_results = []
        for sub_query in sub_queries:
            results = retriever.search(sub_query, top_k=top_k)
            all_results.append(results)
        
        # 3. 合并结果
        merged = reciprocal_rank_fusion(all_results)
        
        return merged[:top_k], sub_queries

# 使用
decomposer = QueryDecomposer(llm)
results, sub_queries = decomposer.search_and_merge(
    "比较 React 和 Vue 的性能和学习曲线",
    retriever
)
```

### 4.3 递归分解

对于更复杂的问题，可以递归分解：

```python
def recursive_decompose(query, llm, max_depth=2, current_depth=0):
    """
    递归分解问题
    """
    if current_depth >= max_depth:
        return [query]
    
    # 判断是否需要分解
    if is_simple_query(query, llm):
        return [query]
    
    # 分解
    sub_queries = decompose(query, llm)
    
    # 递归处理子问题
    all_sub_queries = []
    for sub_q in sub_queries:
        all_sub_queries.extend(
            recursive_decompose(sub_q, llm, max_depth, current_depth + 1)
        )
    
    return all_sub_queries

def is_simple_query(query, llm):
    """
    判断问题是否足够简单
    """
    prompt = f"""判断以下问题是否是一个简单的、可以直接回答的问题。
如果是，回答"是"；如果问题复杂需要分解，回答"否"。

问题：{query}

回答："""
    
    response = llm.generate(prompt).strip()
    return response == "是"
```

---

## 五、Step-back Prompting

### 5.1 核心思想

先问一个更抽象/更基础的问题，获取背景知识，再回答具体问题。

```
具体问题: "如果地球温度升高2度，北极熊会怎样"

Step-back 问题: "气候变化对北极生态系统的影响是什么"

流程:
1. 检索 step-back 问题的答案（背景知识）
2. 结合背景知识回答原始问题
```

### 5.2 实现

```python
class StepBackPrompting:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def generate_stepback_query(self, query):
        """
        生成 step-back 问题
        """
        prompt = f"""给定一个具体问题，生成一个更抽象、更基础的问题，
这个问题的答案可以作为回答原始问题的背景知识。

具体问题：{query}

抽象问题："""

        stepback_query = self.llm.generate(prompt).strip()
        return stepback_query
    
    def search(self, query, top_k=5):
        """
        Step-back 检索
        """
        # 1. 生成 step-back 问题
        stepback_query = self.generate_stepback_query(query)
        
        # 2. 检索 step-back 问题
        stepback_results = self.retriever.search(stepback_query, top_k=top_k)
        
        # 3. 检索原始问题
        original_results = self.retriever.search(query, top_k=top_k)
        
        # 4. 合并结果
        combined = stepback_results + original_results
        
        # 去重
        seen = set()
        unique_results = []
        for r in combined:
            if r['id'] not in seen:
                seen.add(r['id'])
                unique_results.append(r)
        
        return unique_results[:top_k]
    
    def answer(self, query, top_k=5):
        """
        使用 step-back 策略回答问题
        """
        # 获取检索结果
        results = self.search(query, top_k)
        context = '\n'.join([r['text'] for r in results])
        
        # 生成答案
        prompt = f"""基于以下背景知识回答问题。

背景知识：
{context}

问题：{query}

回答："""

        answer = self.llm.generate(prompt)
        return answer
```

### 5.3 示例

```
原始问题: "为什么 Transformer 比 RNN 更适合处理长序列"

Step-back 问题: "RNN 和 Transformer 的核心区别是什么"

检索 step-back 问题得到:
- RNN 的顺序处理特性
- Transformer 的自注意力机制
- 并行计算能力对比

结合背景知识回答原始问题:
"Transformer 比 RNN 更适合处理长序列，主要因为：
1. 自注意力机制可以直接建模任意位置之间的依赖，不受距离限制
2. 并行计算能力强，不需要顺序处理
3. 没有梯度消失问题..."
```

---

## 六、Query Rewriting（查询重写）

### 6.1 口语转书面语

```python
def colloquial_to_formal(query, llm):
    """
    口语化查询转书面语
    """
    prompt = f"""将以下口语化的查询改写为正式的书面语查询，保持原意不变。

口语查询：{query}

书面查询："""

    formal_query = llm.generate(prompt).strip()
    return formal_query

# 示例
# 输入: "Python 咋读文件啊"
# 输出: "Python 如何读取文件"
```

### 6.2 消除歧义

```python
def disambiguate_query(query, context, llm):
    """
    根据上下文消除查询歧义
    """
    prompt = f"""根据对话上下文，将模糊的查询改写为明确的查询。

对话上下文：
{context}

模糊查询：{query}

明确查询："""

    clear_query = llm.generate(prompt).strip()
    return clear_query

# 示例
# 上下文: "我在研究苹果公司的产品"
# 模糊查询: "苹果的市值是多少"
# 明确查询: "苹果公司(Apple Inc.)的市值是多少"
```

### 6.3 多轮对话中的 Query 重写

```python
class ConversationalQueryRewriter:
    def __init__(self, llm):
        self.llm = llm
        self.history = []
    
    def rewrite(self, query):
        """
        结合对话历史重写查询
        """
        if not self.history:
            return query
        
        history_text = '\n'.join([
            f"用户: {h['user']}\n助手: {h['assistant']}"
            for h in self.history[-3:]  # 只用最近3轮
        ])
        
        prompt = f"""根据对话历史，将用户的最新查询改写为独立的、完整的查询。

对话历史：
{history_text}

用户最新查询：{query}

改写后的查询（应该是独立的，不依赖上下文）："""

        rewritten = self.llm.generate(prompt).strip()
        return rewritten
    
    def add_turn(self, user_query, assistant_response):
        self.history.append({
            'user': user_query,
            'assistant': assistant_response
        })

# 示例
# 历史: 
#   用户: "介绍一下 Transformer"
#   助手: "Transformer 是一种基于自注意力机制的神经网络架构..."
# 当前查询: "它有什么优点"
# 重写后: "Transformer 架构有什么优点"
```

---

## 七、混合策略

### 7.1 Query 处理流水线

```python
class QueryProcessor:
    def __init__(self, llm, embedding_model, retriever):
        self.llm = llm
        self.embedding_model = embedding_model
        self.retriever = retriever
    
    def process(self, query, strategy='auto'):
        """
        根据策略处理查询
        """
        if strategy == 'auto':
            strategy = self.detect_strategy(query)
        
        if strategy == 'expansion':
            return self.expand_query(query)
        elif strategy == 'hyde':
            return self.hyde(query)
        elif strategy == 'decomposition':
            return self.decompose(query)
        elif strategy == 'stepback':
            return self.stepback(query)
        else:
            return query
    
    def detect_strategy(self, query):
        """
        自动检测应该使用的策略
        """
        # 短查询 → 扩展
        if len(query.split()) < 3:
            return 'expansion'
        
        # 复杂问题 → 分解
        if any(word in query for word in ['比较', '区别', '优缺点', '和', '与']):
            return 'decomposition'
        
        # 需要背景知识 → step-back
        if any(word in query for word in ['为什么', '原因', '影响']):
            return 'stepback'
        
        # 默认使用 HyDE
        return 'hyde'
    
    def multi_strategy_search(self, query, top_k=10):
        """
        使用多种策略检索，融合结果
        """
        all_results = []
        
        # 原始查询
        results = self.retriever.search(query, top_k=top_k)
        all_results.append(results)
        
        # 扩展查询
        expanded = self.expand_query(query)
        results = self.retriever.search(expanded, top_k=top_k)
        all_results.append(results)
        
        # HyDE
        hyde_results = self.hyde(query)
        all_results.append(hyde_results)
        
        # 融合
        fused = reciprocal_rank_fusion(all_results)
        
        return fused[:top_k]
```

### 7.2 自适应策略选择

```python
def adaptive_query_processing(query, llm, retriever):
    """
    让 LLM 决定使用哪种策略
    """
    prompt = f"""分析以下查询，决定最佳的处理策略。

查询：{query}

可选策略：
1. direct - 直接检索，适合简单明确的查询
2. expansion - 同义词扩展，适合短查询
3. hyde - 生成假想答案，适合需要详细解释的问题
4. decomposition - 问题分解，适合复杂的多方面问题
5. stepback - 先问基础问题，适合需要背景知识的问题

请选择最佳策略（只输出策略名称）："""

    strategy = llm.generate(prompt).strip().lower()
    
    # 根据策略处理
    if strategy == 'direct':
        return retriever.search(query)
    elif strategy == 'expansion':
        expanded = expand_query(query, llm)
        return retriever.search(expanded)
    # ... 其他策略
```

---

## 八、评估与优化

### 8.1 评估指标

```python
def evaluate_query_processing(original_queries, processed_queries, 
                               retriever, ground_truth, k=10):
    """
    评估 Query 处理的效果
    """
    original_recall = []
    processed_recall = []
    
    for orig_q, proc_q, relevant_docs in zip(
        original_queries, processed_queries, ground_truth
    ):
        # 原始查询的召回率
        orig_results = retriever.search(orig_q, top_k=k)
        orig_hits = len(set(orig_results) & set(relevant_docs))
        original_recall.append(orig_hits / len(relevant_docs))
        
        # 处理后查询的召回率
        proc_results = retriever.search(proc_q, top_k=k)
        proc_hits = len(set(proc_results) & set(relevant_docs))
        processed_recall.append(proc_hits / len(relevant_docs))
    
    return {
        'original_recall@k': np.mean(original_recall),
        'processed_recall@k': np.mean(processed_recall),
        'improvement': np.mean(processed_recall) - np.mean(original_recall)
    }
```

### 8.2 A/B 测试

```python
def ab_test_query_processing(queries, retriever, strategy_a, strategy_b, 
                              ground_truth, n_samples=100):
    """
    A/B 测试两种 Query 处理策略
    """
    results_a = []
    results_b = []
    
    for query, relevant in zip(queries[:n_samples], ground_truth[:n_samples]):
        # 策略 A
        processed_a = strategy_a(query)
        hits_a = retriever.search(processed_a, top_k=10)
        recall_a = len(set(hits_a) & set(relevant)) / len(relevant)
        results_a.append(recall_a)
        
        # 策略 B
        processed_b = strategy_b(query)
        hits_b = retriever.search(processed_b, top_k=10)
        recall_b = len(set(hits_b) & set(relevant)) / len(relevant)
        results_b.append(recall_b)
    
    # 统计显著性检验
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results_a, results_b)
    
    return {
        'mean_recall_a': np.mean(results_a),
        'mean_recall_b': np.mean(results_b),
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

---

## 九、实践建议

### 9.1 策略选择指南

| 场景 | 推荐策略 |
|------|----------|
| 短查询（<3词） | Query 扩展 |
| 口语化查询 | Query 重写 |
| 复杂比较问题 | Query 分解 |
| 需要背景知识 | Step-back |
| 一般问题 | HyDE |
| 多轮对话 | 对话重写 |

### 9.2 延迟与效果权衡

| 策略 | 额外延迟 | 效果提升 |
|------|----------|----------|
| 同义词扩展 | 低 | 中 |
| LLM 扩展 | 中 | 中-高 |
| HyDE | 高 | 高 |
| Query 分解 | 高 | 高（复杂问题） |
| Step-back | 高 | 高（需背景知识） |

### 9.3 常见问题

| 问题 | 解决方案 |
|------|----------|
| LLM 生成错误信息 | 使用多个假想文档取平均 |
| 延迟太高 | 缓存常见查询的处理结果 |
| 扩展词不相关 | 提高相似度阈值，使用领域词典 |
| 分解过度 | 限制分解深度，判断问题复杂度 |
