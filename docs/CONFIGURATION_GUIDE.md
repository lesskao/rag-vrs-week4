# 配置指南

本文档详细说明了 RAG 系统的配置选项和最佳实践。

## 📋 目录

1. [环境配置](#环境配置)
2. [向量数据库配置](#向量数据库配置)
3. [模型配置](#模型配置)
4. [检索参数调优](#检索参数调优)
5. [生产环境部署](#生产环境部署)

## 环境配置

### 基础环境变量

在 `.env` 文件中配置以下变量：

```env
# ========== DeepSeek API ==========
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# ========== Embedding 模型 ==========
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDING_DIMENSION=768

# ========== 检索参数 ==========
TOP_K=20
FINAL_TOP_K=5

# ========== Milvus 配置 ==========
MILVUS_HOST=localhost
MILVUS_PORT=19530

# ========== Qdrant 配置 ==========
QDRANT_HOST=localhost
QDRANT_PORT=6333

# ========== Chroma 配置 ==========
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### Embedding 模型选择

| 模型 | 维度 | 语言 | 性能 | 适用场景 |
|------|------|------|------|---------|
| paraphrase-multilingual-mpnet-base-v2 | 768 | 多语言 | 中等 | 通用场景 |
| text-embedding-ada-002 (OpenAI) | 1536 | 多语言 | 高 | 高质量需求 |
| m3e-base | 768 | 中文优化 | 中等 | 中文为主 |
| bge-large-zh | 1024 | 中文 | 高 | 中文高质量 |

**建议**:
- 开发测试: `paraphrase-multilingual-mpnet-base-v2`
- 生产环境（中文）: `bge-large-zh`
- 生产环境（多语言）: `text-embedding-ada-002`

## 向量数据库配置

### Chroma 配置

**无需额外服务，适合快速开始**

```python
from vector_store_chroma import ChromaVectorStore

vector_store = ChromaVectorStore(
    collection_name="my_collection"
)
```

**配置建议**:
- `persist_directory`: 设置持久化目录
- 定期备份数据目录
- 不要在生产环境使用 Chroma（性能限制）

### Qdrant 配置

**Docker 部署**:

Linux/Mac:
```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Windows CMD:
```cmd
docker run -d --name qdrant -p 6333:6333 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

Windows PowerShell:
```powershell
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

**Python 配置**:
```python
from vector_store_qdrant import QdrantVectorStore

vector_store = QdrantVectorStore(
    collection_name="my_collection"
)
```

**生产配置**:
```yaml
# qdrant_config.yaml
storage:
  storage_path: /qdrant/storage
  
service:
  grpc_port: 6334
  http_port: 6333
  
collection:
  optimizer:
    indexing_threshold: 10000
    
  wal:
    wal_capacity_mb: 32
```

**调优建议**:
- 启用 HNSW 索引（默认启用）
- 调整 `indexing_threshold` 控制索引更新频率
- 增加 `wal_capacity_mb` 提升写入性能

### Milvus 配置

**Docker 部署**:

Linux/Mac:
```bash
# Standalone 模式
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
```

Windows CMD/PowerShell:
```cmd
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 -v milvus_data:/var/lib/milvus milvusdb/milvus:latest
```

**Python 配置**:
```python
from vector_store_milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    collection_name="my_collection"
)

# 自定义索引参数
vector_store.create_collection(
    dimension=768,
    index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)
```

**索引类型选择**:

| 索引类型 | 性能 | 内存 | 准确率 | 适用场景 |
|---------|------|------|--------|---------|
| FLAT | 慢 | 低 | 100% | < 10万条 |
| IVF_FLAT | 中 | 中 | 95%+ | 10万-100万 |
| IVF_SQ8 | 快 | 低 | 90%+ | 内存受限 |
| HNSW | 最快 | 高 | 99%+ | > 100万，追求性能 |

**生产调优**:
```yaml
# milvus.yaml
dataNode:
  flush:
    insertBufSize: 16777216  # 16MB

indexNode:
  scheduler:
    buildParallel: 1  # 并行构建索引

queryNode:
  cache:
    enabled: true
    memoryLimit: 2147483648  # 2GB
```

**建议**:
- 小数据集（< 10万）: `FLAT`
- 中等数据集（10万-100万）: `IVF_FLAT`
- 大数据集（> 100万）: `HNSW`

## 检索参数调优

### 基础参数

```python
request = QueryRequest(
    query="用户查询",
    top_k=20,              # 初检返回数量
    final_top_k=5,         # 最终返回数量
    enable_hybrid=True,    # 启用混合检索
    enable_rerank=True     # 启用重排序
)
```

### 混合检索权重调优

```python
# 在 hybrid_search.py 中调整
hybrid_results = HybridSearchEngine.reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    k=60,              # RRF 参数，越大越平滑
    vector_weight=0.6  # 向量检索权重（0-1）
)
```

**权重选择建议**:
- `vector_weight=0.7`: 语义理解为主（问答、语义搜索）
- `vector_weight=0.5`: 平衡（通用场景）
- `vector_weight=0.3`: 关键词匹配为主（代码搜索、专业术语）

### 分块参数

```python
from chunking_strategy import ChunkingStrategy

# 简单分块
chunks = ChunkingStrategy.simple_chunk(
    text=long_text,
    chunk_size=512,      # 块大小（字符数）
    chunk_overlap=50     # 重叠大小
)

# 父子分块
rag_engine = AdvancedRAGEngine(
    vector_store,
    use_parent_child=True
)
```

**分块大小建议**:

| 内容类型 | chunk_size | chunk_overlap | 说明 |
|---------|-----------|---------------|------|
| 短文本（问答对） | 256 | 20 | 避免过度分割 |
| 中等文档（文章） | 512 | 50 | 平衡精度和上下文 |
| 长文档（书籍） | 1024 | 100 | 保证足够上下文 |
| 代码 | 2048 | 200 | 保持代码完整性 |

### Multi-Query 参数

```python
# 在 deepseek_client.py 中
queries = client.generate_multi_queries(
    query="用户查询",
    num_queries=3  # 生成查询数量
)
```

**建议**:
- `num_queries=2-3`: 一般场景
- `num_queries=4-5`: 歧义性强的查询
- 注意: 数量越多，API 调用成本越高

### 重排序参数

```python
results = rag_engine.rerank(
    query="用户查询",
    results=initial_results,
    top_k=5  # 最终保留数量
)
```

**建议**:
- 初检 `top_k=20`，重排序后取 `top_k=5`
- 重排序比例: 1:4 到 1:5 之间最佳
- 对延迟敏感的场景可以关闭重排序

## 性能优化

### 批量索引优化

```python
# 大批量索引时
rag_engine.index_documents(
    documents,
    batch_size=100,      # 批次大小
    show_progress=True   # 显示进度
)
```

**建议**:
- Chroma: `batch_size=50-100`
- Qdrant: `batch_size=100-200`
- Milvus: `batch_size=200-500`

### 缓存策略

```python
# 实现查询缓存（示例）
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query: str, top_k: int):
    return rag_engine.search(query, top_k)
```

### 并发处理

```python
# 使用线程池处理多个查询
from concurrent.futures import ThreadPoolExecutor

def process_queries(queries):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(rag_engine.search, queries))
    return results
```

## 生产环境部署

### 1. 向量数据库选择

**小规模（< 10万文档）**:
- 推荐: Chroma 或 Qdrant
- 单机部署即可

**中等规模（10万-100万）**:
- 推荐: Qdrant
- 考虑主从复制

**大规模（> 100万）**:
- 推荐: Milvus
- 必须使用分布式部署

### 2. 高可用配置

**Qdrant 集群**:
```yaml
# 主节点
cluster:
  enabled: true
  p2p:
    port: 6335
  
# 从节点
consensus:
  tick_period_ms: 100
```

**Milvus 集群**:
- 使用 Kubernetes 部署
- 配置多个 QueryNode 和 DataNode
- 使用外部 etcd 和 MinIO

### 3. 监控和告警

**关键指标**:
- 检索延迟（P50, P95, P99）
- 索引吞吐量
- 数据库连接数
- 内存使用率

**监控工具**:
- Prometheus + Grafana
- 向量数据库自带的 metrics 接口

### 4. 备份策略

**Chroma**:
```bash
# 定期备份数据目录
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db/
```

**Qdrant**:
```bash
# 使用 snapshot API
curl -X POST "http://localhost:6333/collections/my_collection/snapshots"
```

**Milvus**:
```bash
# 备份 metadata 和 binlogs
milvus-backup create --collection my_collection
```

## 常见问题

### Q1: 检索速度慢怎么办？

**A**: 
1. 检查索引类型是否合适
2. 增加向量数据库的 cache 配置
3. 考虑减少 `top_k` 或关闭重排序
4. 使用批量查询

### Q2: 检索结果不相关？

**A**:
1. 启用混合检索和重排序
2. 调整分块大小
3. 尝试不同的 Embedding 模型
4. 使用 Multi-Query 或 HyDE

### Q3: 内存占用过高？

**A**:
1. 使用量化索引（如 IVF_SQ8）
2. 调小 cache 配置
3. 限制并发查询数
4. 考虑使用磁盘索引

### Q4: API 成本过高？

**A**:
1. 减少 Multi-Query 的查询数量
2. 只在必要时使用重排序
3. 关闭 HyDE（成本较高）
4. 使用本地 Reranking 模型替代 API

## 总结

根据您的场景选择合适的配置：

**开发测试**: Chroma + 基础 Embedding + 关闭高级功能
**小规模生产**: Qdrant + 混合检索 + 选择性重排序
**大规模生产**: Milvus + 全部优化技术 + 集群部署

记住：**没有最好的配置，只有最适合的配置**。根据实际需求和资源进行权衡。
