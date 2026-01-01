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