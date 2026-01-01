# 项目结构说明

本文档详细说明了项目的文件组织结构和各个模块的职责。

## 📁 目录结构

```
rag-vrs-week4/
│
├── src/                              # 源代码目录
│   ├── __init__.py
│   │
│   ├── core/                         # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py                 # 全局配置管理
│   │   └── models.py                 # Pydantic 数据模型定义
│   │
│   ├── vectorstores/                 # 向量数据库模块
│   │   ├── __init__.py
│   │   ├── vector_store_base.py      # 抽象接口
│   │   ├── vector_store_milvus.py    # Milvus 实现
│   │   ├── vector_store_qdrant.py    # Qdrant 实现
│   │   └── vector_store_chroma.py    # Chroma 实现
│   │
│   ├── retrievers/                   # 检索模块
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py         # BM25 关键词检索
│   │   ├── hybrid_search.py          # 混合检索（RRF 融合）
│   │   └── chunking_strategy.py      # 文档分块策略
│   │
│   ├── llm/                          # 大语言模型模块
│   │   ├── __init__.py
│   │   ├── deepseek_client.py        # DeepSeek API 客户端
│   │   └── embedding_manager.py      # Embedding 模型管理
│   │
│   ├── utils/                        # 工具模块
│   │   └── __init__.py
│   │
│   └── rag_engine.py                 # RAG 核心引擎（主入口）
│
├── examples/                         # 示例代码
│   ├── quick_start.py                # 快速启动脚本
│   └── example_usage.py              # 使用示例
│
├── tests/                            # 测试代码
│   └── benchmark.py                  # 性能评测脚本
│
├── scripts/                          # 辅助脚本
│   ├── start_milvus.bat              # 启动 Milvus (docker-compose)
│   ├── stop_milvus.bat               # 停止 Milvus 服务
│   ├── check_milvus_status.bat       # 检查服务状态
│   ├── test_chroma_simple.py         # Chroma 快速测试
│   ├── MILVUS_GUIDE.md               # Milvus 完整指南
│   └── README.md                     # 脚本说明
│
├── docs/                             # 文档目录
│   ├── CONFIGURATION_GUIDE.md        # 配置指南
│   └── PROJECT_STRUCTURE.md          # 本文件
│
├── config/                           # 配置文件目录
│   └── .env.example                  # 环境变量示例
│
├── .gitignore                        # Git 忽略文件
├── requirements.txt                  # Python 依赖
└── README.md                         # 项目说明
```

## 🔍 模块详细说明

### 1. src/core/ - 核心模块

#### config.py
**职责**: 管理全局配置，使用 Pydantic Settings 从环境变量读取配置

**主要内容**:
- DeepSeek API 配置
- 向量数据库连接配置
- Embedding 模型配置
- 检索参数配置

**使用方式**:
```python
from src.core.config import settings
api_key = settings.deepseek_api_key
```

#### models.py
**职责**: 定义所有数据模型，确保类型安全和数据验证

**主要模型**:
- `Document`: 文档数据模型
- `SearchResult`: 检索结果模型
- `QueryRequest`: 查询请求模型
- `ChunkStrategy`: 分块策略配置

### 2. src/vectorstores/ - 向量数据库模块

#### vector_store_base.py
**职责**: 定义向量数据库的抽象接口（ABC）

**核心方法**:
```python
class RAGVectorStore(ABC):
    @abstractmethod
    def batch_upsert(documents) -> bool
    @abstractmethod
    def search(query_embedding, top_k, filters) -> List[SearchResult]
    @abstractmethod
    def create_collection(dimension) -> bool
```

#### vector_store_milvus.py
Milvus 向量数据库实现 - 高性能，适合大规模生产环境

#### vector_store_qdrant.py
Qdrant 向量数据库实现 - 现代化，丰富的过滤功能

#### vector_store_chroma.py
Chroma 向量数据库实现 - 轻量级，开发测试首选

### 3. src/retrievers/ - 检索模块

#### bm25_retriever.py
**职责**: BM25 关键词检索器

**特点**:
- 支持中英文混合分词
- 提供与向量检索相同的接口

#### hybrid_search.py
**职责**: 实现混合检索和结果融合

**核心算法**: RRF (Reciprocal Rank Fusion)

#### chunking_strategy.py
**职责**: 文档分块策略实现

**支持的策略**:
- 简单分块：固定大小 + 重叠
- 父子分块：索引小块，返回大块

### 4. src/llm/ - 大语言模型模块

#### deepseek_client.py
**职责**: DeepSeek API 的封装客户端

**核心功能**:
- Multi-Query: 查询改写
- HyDE: 生成假设性文档
- Reranking: 文档重排序
- Answer Generation: 基于上下文生成答案

#### embedding_manager.py
**职责**: 管理 Embedding 模型，将文本转换为向量

### 5. src/rag_engine.py - RAG 核心引擎

**职责**: 整合所有功能的主引擎

**核心功能**:
1. 文档索引：支持批量索引和父子分块
2. 高级检索：混合检索、Multi-Query、HyDE
3. 结果优化：重排序、去重、合并
4. 答案生成：基于检索上下文生成答案

## 🎯 导入规范

### 包内导入（推荐）

使用相对导入或完整路径：

```python
# 在 src/rag_engine.py 中
from src.vectorstores import RAGVectorStore
from src.core.models import Document
from src.llm import DeepSeekClient
```

### 外部使用

```python
# 在 examples/ 或 tests/ 中
import sys
sys.path.insert(0, '..')

from src.vectorstores import ChromaVectorStore
from src.rag_engine import AdvancedRAGEngine
from src.core.models import Document
```

## 🔗 模块依赖关系

```
┌─────────────────┐
│   src/core/     │
│  config, models │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────────────┐    ┌──────────────────┐
│ src/vectorstores │    │   src/llm/       │
│ (向量数据库)      │    │ (LLM & Embedding)│
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         │      ┌────────────────┴────┐
         │      │                     │
         └──────►  src/retrievers/    │
                │  (检索策略)          │
                └────────┬────────────┘
                         │
                    ┌────▼────┐
                    │rag_     │
                    │engine.py│
                    └─────────┘
```

## 📚 使用示例

### 基本导入

```python
# 导入向量数据库
from src.vectorstores import ChromaVectorStore, MilvusVectorStore, QdrantVectorStore

# 导入数据模型
from src.core.models import Document, QueryRequest

# 导入主引擎
from src.rag_engine import AdvancedRAGEngine

# 导入配置
from src.core.config import settings
```

### 完整使用流程

```python
# 1. 选择向量数据库
from src.vectorstores import ChromaVectorStore
vector_store = ChromaVectorStore("my_collection")
vector_store.create_collection(dimension=768)

# 2. 创建 RAG 引擎
from src.rag_engine import AdvancedRAGEngine
rag_engine = AdvancedRAGEngine(vector_store)

# 3. 索引文档
from src.core.models import Document
documents = [
    Document(id="doc1", content="...", metadata={})
]
rag_engine.index_documents(documents)

# 4. 查询
from src.core.models import QueryRequest
request = QueryRequest(query="问题", top_k=5)
response = rag_engine.query(request)
```

## 🔧 扩展指南

### 添加新的向量数据库

1. 在 `src/vectorstores/` 创建新文件
2. 继承 `RAGVectorStore` 抽象类
3. 实现所有抽象方法
4. 在 `src/vectorstores/__init__.py` 中导出

### 添加新的检索算法

1. 在 `src/retrievers/` 创建新文件
2. 实现检索逻辑
3. 在 `src/rag_engine.py` 中集成

### 添加新的分块策略

在 `src/retrievers/chunking_strategy.py` 中添加新方法

## 💡 设计原则

1. **模块化**: 每个目录负责单一功能领域
2. **可扩展**: 通过抽象接口实现多态
3. **低耦合**: 模块间依赖清晰，易于替换
4. **易测试**: 示例和测试代码独立于源码

## 📝 最佳实践

### 1. 目录职责清晰
- `src/`: 只包含可复用的源代码
- `examples/`: 使用示例，面向用户
- `tests/`: 测试和评测代码
- `docs/`: 文档资料

### 2. 导入路径统一
始终使用 `from src.xxx import yyy` 的形式

### 3. 配置集中管理
所有配置通过 `src/core/config.py` 统一管理

### 4. 接口先行
先定义抽象接口，再实现具体类

## 🎓 与原结构对比

### 原结构问题
- 所有文件在根目录，难以管理
- 缺少层次结构
- 模块职责不清晰

### 新结构优势
- ✅ 清晰的目录层次
- ✅ 模块职责明确
- ✅ 符合 Python 包规范
- ✅ 易于扩展和维护
- ✅ 适合团队协作

## 相关文档

- [README.md](../README.md) - 项目概述和快速开始
- [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) - 详细配置指南

---

**项目结构设计遵循"高内聚、低耦合"原则，便于长期维护和扩展。**
