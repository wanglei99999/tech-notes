# ANN（近似最近邻）算法详解

## 一、问题定义

### 1.1 最近邻搜索问题

给定：
- 数据集 $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n\}$，其中 $\mathbf{x}_i \in \mathbb{R}^d$
- 查询向量 $\mathbf{q} \in \mathbb{R}^d$
- 距离函数 $d(\cdot, \cdot)$

目标：找到 $\mathbf{x}^* = \arg\min_{\mathbf{x} \in \mathcal{D}} d(\mathbf{q}, \mathbf{x})$

### 1.2 暴力搜索的复杂度

```python
def brute_force_search(query, database):
    min_dist = float('inf')
    nearest = None
    for x in database:  # O(n)
        dist = distance(query, x)  # O(d)
        if dist < min_dist:
            min_dist = dist
            nearest = x
    return nearest
```

时间复杂度：$O(n \cdot d)$

当 $n = 10^8$（1亿向量），$d = 768$，单次查询需要约 768 亿次浮点运算，不可接受。

### 1.3 ANN 的思想

用**少量精度损失**换取**数量级的速度提升**。

定义 c-近似最近邻：

$$d(\mathbf{q}, \mathbf{x}_{ANN}) \leq c \cdot d(\mathbf{q}, \mathbf{x}^*)$$

其中 $c \geq 1$ 是近似比，$c=1$ 表示精确最近邻。

---

## 二、HNSW（Hierarchical Navigable Small World）

### 2.1 理论基础：小世界网络

**六度分隔理论**：任意两个人之间平均只需要 6 个中间人就能建立联系。

**小世界网络特性**：
1. 高聚类系数：朋友的朋友很可能也是朋友
2. 短平均路径长度：任意两点之间的最短路径很短

### 2.2 NSW（Navigable Small World）

#### 图构建

每个向量是一个节点，通过以下方式建立边：

```python
def build_nsw(vectors, M):
    """
    M: 每个节点的最大邻居数
    """
    graph = {i: [] for i in range(len(vectors))}
    
    for i, v in enumerate(vectors):
        # 找到当前图中最近的 M 个节点
        neighbors = greedy_search(graph, v, M)
        
        # 建立双向边
        for neighbor in neighbors:
            if len(graph[i]) < M:
                graph[i].append(neighbor)
            if len(graph[neighbor]) < M:
                graph[neighbor].append(i)
    
    return graph
```

#### 贪心搜索

```python
def greedy_search(graph, query, k, entry_point):
    """
    从 entry_point 开始，贪心地向 query 靠近
    """
    visited = set()
    candidates = [entry_point]  # 优先队列，按距离排序
    result = []
    
    while candidates:
        current = candidates.pop(0)  # 取最近的候选
        
        if current in visited:
            continue
        visited.add(current)
        
        # 检查是否比当前结果更好
        if len(result) < k or dist(query, current) < dist(query, result[-1]):
            result.append(current)
            result.sort(key=lambda x: dist(query, x))
            result = result[:k]
        
        # 扩展邻居
        for neighbor in graph[current]:
            if neighbor not in visited:
                candidates.append(neighbor)
        candidates.sort(key=lambda x: dist(query, x))
    
    return result
```

#### NSW 的问题

- 搜索可能陷入局部最优
- 对于大规模数据，搜索路径可能很长

### 2.3 HNSW：分层结构

#### 核心思想

构建多层图，高层稀疏用于快速定位，底层稠密用于精确搜索。

```
Layer 2:  A -------- D -------- G        (最稀疏)
          |          |          |
Layer 1:  A --- B -- D --- E -- G        (中等密度)
          |    |     |     |    |
Layer 0:  A-B-C-D-E-F-G-H-I-J-K-L        (最稠密，包含所有节点)
```

#### 层级分配

每个节点被分配到的最高层级 $l$ 服从指数分布：

$$l = \lfloor -\ln(\text{uniform}(0,1)) \cdot m_L \rfloor$$

其中 $m_L = 1/\ln(M)$ 是归一化因子，$M$ 是每层的最大邻居数。

这意味着：
- 大多数节点只在 Layer 0
- 少数节点在 Layer 1
- 极少数节点在更高层

#### 搜索算法

```python
def hnsw_search(query, k, ef):
    """
    ef: 搜索时的候选集大小（ef >= k）
    """
    # 从最高层的入口点开始
    entry_point = get_entry_point()
    current_layer = get_max_layer()
    
    # 在高层快速定位
    while current_layer > 0:
        entry_point = greedy_search_layer(
            query, entry_point, 1, current_layer
        )
        current_layer -= 1
    
    # 在 Layer 0 精确搜索
    result = greedy_search_layer(
        query, entry_point, ef, layer=0
    )
    
    return result[:k]
```

#### 插入算法

```python
def hnsw_insert(new_vector, M, ef_construction):
    """
    M: 每层最大邻居数
    ef_construction: 构建时的候选集大小
    """
    # 1. 确定新节点的最高层级
    max_layer = floor(-ln(random()) * m_L)
    
    # 2. 从顶层开始，找到每层的入口点
    entry_point = get_entry_point()
    current_layer = get_max_layer()
    
    # 3. 在高于 max_layer 的层，只做贪心搜索找入口
    while current_layer > max_layer:
        entry_point = greedy_search_layer(
            new_vector, entry_point, 1, current_layer
        )
        current_layer -= 1
    
    # 4. 在 max_layer 及以下的层，建立连接
    while current_layer >= 0:
        neighbors = greedy_search_layer(
            new_vector, entry_point, ef_construction, current_layer
        )
        
        # 选择最近的 M 个作为邻居
        selected = select_neighbors(new_vector, neighbors, M)
        
        # 建立双向连接
        for neighbor in selected:
            add_edge(new_vector, neighbor, current_layer)
            
            # 如果邻居的连接数超过 M，需要裁剪
            if len(get_neighbors(neighbor, current_layer)) > M:
                prune_connections(neighbor, M, current_layer)
        
        entry_point = neighbors[0]
        current_layer -= 1
```

#### 邻居选择策略

**简单策略**：选择最近的 M 个

**启发式策略**（效果更好）：

```python
def select_neighbors_heuristic(query, candidates, M):
    """
    不只选最近的，还要保证多样性
    """
    result = []
    candidates = sorted(candidates, key=lambda x: dist(query, x))
    
    for candidate in candidates:
        if len(result) >= M:
            break
        
        # 检查 candidate 是否比已选的任何节点更近
        good = True
        for selected in result:
            if dist(candidate, selected) < dist(query, candidate):
                good = False
                break
        
        if good:
            result.append(candidate)
    
    return result
```

这个启发式确保选择的邻居在不同方向上，提高图的导航性。

### 2.4 HNSW 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|------------|------------|
| 构建 | $O(n \log n)$ | $O(n \cdot M)$ |
| 搜索 | $O(\log n)$ | - |
| 插入 | $O(\log n)$ | - |

### 2.5 HNSW 参数调优

| 参数 | 含义 | 建议值 | 影响 |
|------|------|--------|------|
| M | 每层最大邻居数 | 16-64 | 越大召回率越高，内存越大 |
| ef_construction | 构建时候选集大小 | 100-500 | 越大构建越慢，质量越高 |
| ef_search | 搜索时候选集大小 | 50-200 | 越大搜索越慢，召回率越高 |

---

## 三、IVF（Inverted File Index）

### 3.1 核心思想

将向量空间划分为多个区域（Voronoi cells），查询时只搜索最近的几个区域。

### 3.2 训练阶段：K-Means 聚类

#### K-Means 算法

目标：将 n 个向量划分为 k 个簇，最小化簇内距离之和。

$$\min_{\mathcal{C}} \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2$$

其中 $\boldsymbol{\mu}_i$ 是第 i 个簇的中心。

```python
def kmeans(vectors, k, max_iter=100):
    # 1. 随机初始化 k 个中心
    centroids = random_sample(vectors, k)
    
    for _ in range(max_iter):
        # 2. 分配：每个向量分配到最近的中心
        assignments = []
        for v in vectors:
            nearest = argmin([dist(v, c) for c in centroids])
            assignments.append(nearest)
        
        # 3. 更新：重新计算每个簇的中心
        new_centroids = []
        for i in range(k):
            cluster_vectors = [v for v, a in zip(vectors, assignments) if a == i]
            new_centroids.append(mean(cluster_vectors))
        
        # 4. 检查收敛
        if centroids == new_centroids:
            break
        centroids = new_centroids
    
    return centroids, assignments
```

### 3.3 索引结构

```
聚类中心（Centroids）: [c1, c2, ..., ck]

倒排列表（Inverted Lists）:
  c1 → [v1, v5, v9, ...]    # 属于簇1的所有向量
  c2 → [v2, v3, v7, ...]    # 属于簇2的所有向量
  ...
  ck → [v4, v8, v10, ...]   # 属于簇k的所有向量
```

### 3.4 搜索算法

```python
def ivf_search(query, k, nprobe):
    """
    nprobe: 搜索的簇数量
    """
    # 1. 找到最近的 nprobe 个聚类中心
    distances = [dist(query, c) for c in centroids]
    nearest_clusters = argsort(distances)[:nprobe]
    
    # 2. 在这些簇内搜索
    candidates = []
    for cluster_id in nearest_clusters:
        for vector in inverted_lists[cluster_id]:
            candidates.append((vector, dist(query, vector)))
    
    # 3. 返回最近的 k 个
    candidates.sort(key=lambda x: x[1])
    return candidates[:k]
```

### 3.5 参数选择

**nlist（簇数量）**：

经验公式：
- $n < 10^6$：$nlist = 4 \sqrt{n}$ 到 $16 \sqrt{n}$
- $n \geq 10^6$：$nlist = 65536$ 到 $262144$

**nprobe（搜索簇数）**：

- 越大召回率越高，但速度越慢
- 通常 $nprobe = 1\%$ 到 $10\%$ 的 $nlist$

### 3.6 IVF 的局限性

1. **边界问题**：查询点可能靠近簇边界，真正的最近邻在相邻簇中
2. **聚类质量**：K-Means 对初始化敏感，可能收敛到局部最优
3. **高维诅咒**：高维空间中，聚类效果下降

---

## 四、PQ（Product Quantization）

### 4.1 向量量化基础

**标量量化**：将连续值映射到离散值

$$q(x) = \arg\min_{c \in \mathcal{C}} |x - c|$$

其中 $\mathcal{C}$ 是码本（codebook）。

**向量量化**：将向量映射到码本中最近的码字

$$q(\mathbf{x}) = \arg\min_{\mathbf{c} \in \mathcal{C}} ||\mathbf{x} - \mathbf{c}||$$

问题：如果向量是 d 维，码本大小为 k，存储码本需要 $O(k \cdot d)$，k 不能太大。

### 4.2 乘积量化的思想

将 d 维向量分成 m 个子向量，每个子向量独立量化。

$$\mathbf{x} = [\mathbf{x}^1, \mathbf{x}^2, ..., \mathbf{x}^m]$$

其中 $\mathbf{x}^i \in \mathbb{R}^{d/m}$。

每个子空间有自己的码本 $\mathcal{C}^i$，大小为 $k^*$（通常 256）。

**总码本大小**：$m \times k^* \times (d/m) = k^* \times d$

**可表示的码字数**：$(k^*)^m$

例如：$m=8, k^*=256$，可表示 $256^8 \approx 1.8 \times 10^{19}$ 个不同的码字！

### 4.3 训练过程

```python
def train_pq(vectors, m, k_star):
    """
    vectors: 训练向量集
    m: 子空间数量
    k_star: 每个子空间的码本大小
    """
    d = vectors.shape[1]
    sub_dim = d // m
    codebooks = []
    
    for i in range(m):
        # 提取第 i 个子向量
        sub_vectors = vectors[:, i*sub_dim : (i+1)*sub_dim]
        
        # 对子向量做 K-Means
        centroids, _ = kmeans(sub_vectors, k_star)
        codebooks.append(centroids)
    
    return codebooks
```

### 4.4 编码过程

```python
def encode_pq(vector, codebooks):
    """
    将向量编码为 m 个码字 ID
    """
    m = len(codebooks)
    d = len(vector)
    sub_dim = d // m
    codes = []
    
    for i in range(m):
        sub_vector = vector[i*sub_dim : (i+1)*sub_dim]
        
        # 找到最近的码字
        distances = [dist(sub_vector, c) for c in codebooks[i]]
        nearest_code = argmin(distances)
        codes.append(nearest_code)
    
    return codes  # m 个整数，每个 0-255
```

**压缩比**：

原始向量：$d \times 4$ 字节（float32）
编码后：$m$ 字节（每个码字用 1 字节表示）

例如：$d=768, m=8$，压缩比 = $768 \times 4 / 8 = 384$ 倍

### 4.5 距离计算

#### 对称距离（SDC）

$$d_{SDC}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{m} ||q^i(\mathbf{x}^i) - q^i(\mathbf{y}^i)||^2$$

两个向量都用量化后的码字计算距离。

#### 非对称距离（ADC）

$$d_{ADC}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{m} ||\mathbf{x}^i - q^i(\mathbf{y}^i)||^2$$

查询向量 $\mathbf{x}$ 不量化，数据库向量 $\mathbf{y}$ 量化。

ADC 更精确，因为查询向量保持原始精度。

#### 查表加速

预计算查询向量到所有码字的距离：

```python
def build_distance_table(query, codebooks):
    """
    预计算距离表
    """
    m = len(codebooks)
    k_star = len(codebooks[0])
    sub_dim = len(query) // m
    
    # distance_table[i][j] = query 的第 i 个子向量到第 i 个码本的第 j 个码字的距离
    distance_table = np.zeros((m, k_star))
    
    for i in range(m):
        sub_query = query[i*sub_dim : (i+1)*sub_dim]
        for j in range(k_star):
            distance_table[i][j] = dist(sub_query, codebooks[i][j])
    
    return distance_table

def compute_distance_with_table(codes, distance_table):
    """
    使用距离表计算距离
    """
    total_dist = 0
    for i, code in enumerate(codes):
        total_dist += distance_table[i][code]
    return total_dist
```

复杂度：
- 构建距离表：$O(m \times k^* \times d/m) = O(k^* \times d)$
- 计算单个距离：$O(m)$（只需要 m 次查表）

### 4.6 PQ 的量化误差

量化误差：

$$E = \mathbb{E}[||\mathbf{x} - q(\mathbf{x})||^2]$$

影响因素：
1. **子空间数 m**：m 越大，每个子空间维度越低，量化越粗糙
2. **码本大小 $k^*$**：越大误差越小，但存储增加
3. **数据分布**：数据越聚集，量化效果越好

### 4.7 OPQ（Optimized Product Quantization）

**问题**：PQ 假设子空间独立，但实际数据可能有相关性。

**解决**：学习一个旋转矩阵 R，使旋转后的数据更适合 PQ。

$$\min_{R, \mathcal{C}} \sum_{\mathbf{x}} ||\mathbf{x} - R^\top q(R\mathbf{x})||^2$$

其中 R 是正交矩阵。

---

## 五、组合索引

### 5.1 IVF-PQ

结合 IVF 的粗筛和 PQ 的压缩：

```
1. 粗量化：IVF 将向量分配到簇
2. 残差计算：r = x - centroid
3. 细量化：PQ 编码残差
```

**为什么编码残差？**

残差的方差比原始向量小，PQ 量化误差更小。

```python
def ivf_pq_add(vector, centroids, codebooks):
    # 1. 找到最近的聚类中心
    cluster_id = argmin([dist(vector, c) for c in centroids])
    
    # 2. 计算残差
    residual = vector - centroids[cluster_id]
    
    # 3. PQ 编码残差
    codes = encode_pq(residual, codebooks)
    
    # 4. 存储
    inverted_lists[cluster_id].append((vector_id, codes))

def ivf_pq_search(query, k, nprobe, centroids, codebooks):
    # 1. 找到最近的 nprobe 个簇
    cluster_distances = [dist(query, c) for c in centroids]
    nearest_clusters = argsort(cluster_distances)[:nprobe]
    
    candidates = []
    for cluster_id in nearest_clusters:
        # 2. 计算查询相对于该簇中心的残差
        query_residual = query - centroids[cluster_id]
        
        # 3. 构建距离表
        distance_table = build_distance_table(query_residual, codebooks)
        
        # 4. 遍历该簇内的所有向量
        for vector_id, codes in inverted_lists[cluster_id]:
            dist = compute_distance_with_table(codes, distance_table)
            candidates.append((vector_id, dist))
    
    # 5. 返回最近的 k 个
    candidates.sort(key=lambda x: x[1])
    return candidates[:k]
```

### 5.2 HNSW + PQ

用 HNSW 做图索引，用 PQ 压缩存储：

- 图结构存储节点 ID 和边
- 向量用 PQ 编码存储
- 搜索时用 PQ 距离做近似计算

---

## 六、性能对比

### 6.1 理论复杂度

| 算法 | 构建时间 | 搜索时间 | 空间 |
|------|----------|----------|------|
| 暴力搜索 | $O(1)$ | $O(nd)$ | $O(nd)$ |
| HNSW | $O(n\log n)$ | $O(\log n)$ | $O(nM)$ |
| IVF | $O(nk)$ | $O(n/k \cdot nprobe)$ | $O(nd + kd)$ |
| PQ | $O(nmk^*)$ | $O(nm)$ | $O(nm)$ |
| IVF-PQ | $O(nk + nmk^*)$ | $O(m \cdot nprobe \cdot n/k)$ | $O(nm + kd)$ |

### 6.2 实际选型建议

| 数据规模 | 内存充足 | 内存受限 |
|----------|----------|----------|
| < 100万 | HNSW | IVF-PQ |
| 100万-1000万 | HNSW | IVF-PQ |
| > 1000万 | HNSW + 分片 | IVF-PQ |

### 6.3 召回率 vs 速度权衡

```
召回率 (Recall@10)
    ^
1.0 |     * HNSW (ef=200)
    |   *   HNSW (ef=100)
0.9 | *     HNSW (ef=50)
    |         * IVF (nprobe=64)
0.8 |       *   IVF (nprobe=32)
    |     *     IVF (nprobe=16)
0.7 |   *       IVF-PQ (nprobe=64)
    | *         IVF-PQ (nprobe=32)
    +---------------------------------> QPS (queries/sec)
      100   1000  10000  100000
```

---

## 七、实践建议

### 7.1 FAISS 使用示例

```python
import faiss
import numpy as np

# 数据准备
d = 768  # 向量维度
n = 1000000  # 数据量
vectors = np.random.random((n, d)).astype('float32')
query = np.random.random((1, d)).astype('float32')

# 1. 暴力搜索（基准）
index_flat = faiss.IndexFlatL2(d)
index_flat.add(vectors)
D, I = index_flat.search(query, k=10)

# 2. HNSW
index_hnsw = faiss.IndexHNSWFlat(d, M=32)
index_hnsw.hnsw.efConstruction = 200
index_hnsw.add(vectors)
index_hnsw.hnsw.efSearch = 100
D, I = index_hnsw.search(query, k=10)

# 3. IVF
nlist = 1024
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
index_ivf.train(vectors)
index_ivf.add(vectors)
index_ivf.nprobe = 32
D, I = index_ivf.search(query, k=10)

# 4. IVF-PQ
m = 8  # 子空间数
index_ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, 8)
index_ivfpq.train(vectors)
index_ivfpq.add(vectors)
index_ivfpq.nprobe = 32
D, I = index_ivfpq.search(query, k=10)
```

### 7.2 参数调优流程

1. **确定召回率目标**：如 95%
2. **选择索引类型**：根据数据规模和内存限制
3. **网格搜索参数**：在验证集上测试不同参数组合
4. **权衡速度和精度**：选择满足召回率目标的最快配置
