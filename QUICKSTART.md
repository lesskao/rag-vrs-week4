# 快速开始指南

欢迎使用高级 RAG 系统！本指南将帮助你快速上手，从安装到运行第一个示例。

## 🚀 快速开始（5 分钟）

### 步骤 1: 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 uv（更快）
uv venv
uv pip install -r requirements.txt
```

### 步骤 2: 启动 Milvus 向量数据库

**Windows 用户（推荐）**：
```cmd
# 双击运行或在 CMD 中执行
scripts\start_milvus.bat
```

**所有平台（通用方法）**：
```bash
cd config
docker-compose up -d
```

等待约 30 秒让服务完全就绪。

### 步骤 3: 运行快速启动脚本

```bash
python examples/quick_start.py
```

选择 **[1] 基础版**，无需 API 配置即可体验核心功能！

### 步骤 4: 查看输出

你将看到系统：
1. ✅ 连接到 Milvus 向量数据库
2. ✅ 索引 5 条技术文档
3. ✅ 执行混合检索（向量 + BM25）
4. ✅ 显示相关度排序的结果

## 📝 第一个代码示例

创建文件 `my_first_rag.py`：

```python
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.vectorstores import MilvusVectorStore
from src.rag_engine import AdvancedRAGEngine
from src.core.models import Document

# 1. 初始化 Milvus 数据库
vector_store = MilvusVectorStore("my_first_collection")
vector_store.create_collection(dimension=768)

# 2. 创建 RAG 引擎
rag = AdvancedRAGEngine(vector_store, use_parent_child=False)

# 3. 索引文档
docs = [
    Document(
        id="1",
        content="Python 是一种易学易用的编程语言，广泛应用于数据科学和机器学习",
        metadata={"topic": "编程"}
    ),
    Document(
        id="2",
        content="深度学习使用多层神经网络处理数据，在图像识别和自然语言处理领域表现出色",
        metadata={"topic": "AI"}
    ),
]
rag.index_documents(docs)

# 4. 执行检索（混合检索：向量 + BM25）
results = rag.search("什么是 Python？", top_k=2, enable_hybrid=True)
for r in results:
    print(f"相似度: {r.score:.4f}")
    print(f"话题: {r.document.metadata.get('topic')}")
    print(f"内容: {r.document.content}\n")

# 5. 清理
vector_store.drop_collection()
```

运行：
```bash
# 确保 Milvus 已启动
python my_first_rag.py
```

## 🎯 进阶使用

### 方案 1: 体验高级功能（需要 DeepSeek API）

#### 1. 配置 API Key

**Windows 用户**：
```cmd
# 使用记事本编辑
notepad .env
```

**Linux/Mac 用户**：
```bash
# 创建配置文件
cp .env.template .env

# 编辑配置文件
nano .env  # 或使用你喜欢的编辑器
```

在 `.env` 文件中修改：
```bash
DEEPSEEK_API_KEY=sk-your-actual-key-here
```

#### 2. 运行高级示例

```bash
python examples/quick_start.py
```

选择 **[2] 高级版**，体验完整功能：
- 🔄 **Multi-Query**: 查询扩展，提高召回率
- 🎯 **HyDE**: 假设性文档生成，对齐语义空间
- 🏆 **Reranking**: 使用 DeepSeek 智能重排序
- 💬 **Answer Generation**: 自动生成高质量答案

### 方案 2: 测试简化脚本（无需 API）

运行专门的测试脚本：

```bash
# 测试 Milvus（推荐）
python scripts/test_milvus_simple.py

# 测试 Chroma（轻量级）
python scripts/test_chroma_simple.py
```

## 🔄 切换向量数据库

### 当前默认：Milvus（已配置）

系统已配置使用 Milvus 作为默认向量数据库。

**管理 Milvus**：
```cmd
# 启动
scripts\start_milvus.bat

# 停止
scripts\stop_milvus.bat

# 查看状态
cd config
docker-compose ps
```

### 可选：切换到 Chroma（轻量级）

**优点**：无需 Docker，开箱即用

**修改代码**：
```python
from src.vectorstores import ChromaVectorStore

# 替换 MilvusVectorStore
vector_store = ChromaVectorStore("my_collection")
vector_store.create_collection(dimension=768)
```

### 可选：切换到 Qdrant（生产环境）

**启动 Qdrant**：
```bash
# Windows CMD
docker run -d --name qdrant -p 6333:6333 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# Linux/Mac
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

**修改代码**：
```python
from src.vectorstores import QdrantVectorStore

vector_store = QdrantVectorStore("my_collection")
vector_store.create_collection(dimension=768)
```

### 数据库选择建议

| 数据库 | 适用场景 | 优势 |
|--------|---------|------|
| **Milvus** | 生产环境、大规模数据 | 高性能、可扩展、企业级 |
| **Chroma** | 开发测试、小型项目 | 轻量级、零配置、快速启动 |
| **Qdrant** | 生产环境、复杂过滤 | 现代化、丰富的过滤功能 |

## 📊 性能评测

想知道不同数据库和检索策略的性能差异？运行评测脚本：

```bash
python tests/benchmark.py
```

评测内容：
- ✅ 对比 Milvus、Qdrant、Chroma 的性能
- ✅ 测试不同检索策略（向量、混合、重排序）
- ✅ 生成详细的性能报告

## 📚 更多资源

### 文档
- 📖 [完整文档](README.md) - 项目概览和特性介绍
- ⚙️ [配置指南](docs/CONFIGURATION_GUIDE.md) - 详细配置说明
- 🏗️ [项目结构](docs/PROJECT_STRUCTURE.md) - 代码架构说明

### 脚本和工具
- 🔧 [脚本说明](scripts/README.md) - 管理脚本使用指南
- 📘 [Milvus 指南](scripts/MILVUS_GUIDE.md) - Milvus 完整使用文档

### 示例代码
- 💡 [快速启动](examples/quick_start.py) - 交互式演示
- 📝 [完整示例](examples/example_usage.py) - 所有功能演示
- 🧪 [简单测试](scripts/test_milvus_simple.py) - 快速验证

## ❓ 常见问题

### Q: Milvus 启动失败怎么办？

**A**: 按以下步骤排查：

1. **检查 Docker 是否运行**：
   ```cmd
   docker ps
   ```

2. **查看服务状态**：
   ```cmd
   cd config
   docker-compose ps
   ```

3. **查看日志**：
   ```cmd
   docker-compose logs milvus-standalone
   ```

4. **重启服务**：
   ```cmd
   docker-compose down
   docker-compose up -d
   ```

详细故障排查请参考 [Milvus 指南](scripts/MILVUS_GUIDE.md)。

### Q: ModuleNotFoundError: No module named 'src'

**A**: 这是路径问题，在脚本开头添加：

```python
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

或者确保在项目根目录运行脚本。

### Q: 需要 GPU 吗？

**A**: 不需要！Embedding 模型会自动使用 CPU，速度也很快（首次加载模型会稍慢）。

### Q: 支持中文吗？

**A**: 完全支持！默认使用的 `paraphrase-multilingual-mpnet-base-v2` 是多语言模型，对中文支持良好。

### Q: 不想用 Milvus，可以用其他数据库吗？

**A**: 当然可以！

- **Chroma**（最简单）：无需 Docker，修改代码即可
  ```python
  from src.vectorstores import ChromaVectorStore
  vector_store = ChromaVectorStore("my_collection")
  ```

- **Qdrant**（生产级）：启动 Docker 容器后修改代码
  ```python
  from src.vectorstores import QdrantVectorStore
  vector_store = QdrantVectorStore("my_collection")
  ```

### Q: 如何在生产环境使用？

**A**: 参考以下文档：
- [配置指南](docs/CONFIGURATION_GUIDE.md) - 生产环境配置
- [Milvus 指南](scripts/MILVUS_GUIDE.md) - 生产部署建议

### Q: DeepSeek API 是必需的吗？

**A**: 不是！

- **基础功能**（无需 API）：向量检索、混合检索、元数据过滤
- **高级功能**（需要 API）：Multi-Query、HyDE、Reranking、答案生成

选择 **[1] 基础版** 即可无需配置直接使用。

## 🎉 完成！

恭喜！你已经成功运行了第一个 RAG 应用！

### 🚀 接下来可以做什么？

#### 1. 索引你自己的文档
```python
docs = [
    Document(
        id=f"doc_{i}",
        content=your_text,
        metadata={"source": "my_data", "category": "tech"}
    )
    for i, your_text in enumerate(your_documents)
]
rag.index_documents(docs)
```

#### 2. 调整检索参数
```python
results = rag.search(
    query="你的问题",
    top_k=20,              # 初检数量
    enable_hybrid=True,    # 启用混合检索
    enable_rerank=True,    # 启用重排序（需要 API）
    final_top_k=5          # 最终返回数量
)
```

#### 3. 启用高级功能
- 配置 `.env` 文件中的 `DEEPSEEK_API_KEY`
- 运行 `python examples/quick_start.py` 选择 [2] 高级版
- 体验 Multi-Query、HyDE、Reranking 等功能

#### 4. 性能评测
```bash
python tests/benchmark.py
```

#### 5. 查看更多示例
```bash
python examples/example_usage.py
```

### 📞 需要帮助？

- 📖 查看 [完整文档](README.md)
- 🔧 参考 [配置指南](docs/CONFIGURATION_GUIDE.md)
- 💡 浏览 [示例代码](examples/)
- 🐛 遇到问题？查看 [常见问题](#-常见问题)

---

祝你使用愉快！🚀

如有问题或建议，欢迎提 Issue！
