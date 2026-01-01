"""
RAG 系统使用示例
"""
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vectorstores import ChromaVectorStore
from src.rag_engine import AdvancedRAGEngine
from src.core.models import Document, QueryRequest


def example_basic_usage():
    """基础使用示例"""
    print("\n" + "="*60)
    print("示例 1: 基础 RAG 使用")
    print("="*60)
    
    # 1. 初始化向量数据库（使用 Chroma，无需额外服务）
    vector_store = ChromaVectorStore("demo_collection")
    vector_store.create_collection(dimension=768)
    
    # 2. 创建 RAG 引擎
    rag_engine = AdvancedRAGEngine(vector_store)
    
    # 3. 准备文档
    documents = [
        Document(
            id="doc1",
            content="深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。常见的框架包括 TensorFlow 和 PyTorch。",
            metadata={"category": "tech", "topic": "深度学习"}
        ),
        Document(
            id="doc2",
            content="Python 是一种高级编程语言，广泛应用于数据科学、机器学习和 Web 开发。它有丰富的库生态系统。",
            metadata={"category": "tech", "topic": "编程语言"}
        ),
        Document(
            id="doc3",
            content="向量数据库是专门用于存储和检索向量数据的数据库。常见的向量数据库包括 Milvus、Qdrant 和 Chroma。",
            metadata={"category": "tech", "topic": "数据库"}
        )
    ]
    
    # 4. 索引文档
    rag_engine.index_documents(documents)
    
    # 5. 查询
    request = QueryRequest(
        query="什么是深度学习？",
        top_k=2,
        enable_hybrid=True,
        enable_rerank=False
    )
    
    response = rag_engine.query(request, return_answer=False)
    
    print(f"\n查询结果:")
    for i, result in enumerate(response["results"]):
        print(f"\n结果 {i+1}:")
        print(f"  ID: {result.document.id}")
        print(f"  分数: {result.score:.4f}")
        print(f"  内容: {result.document.content}")
    
    # 清理
    vector_store.drop_collection()


def example_advanced_features():
    """高级功能示例"""
    print("\n" + "="*60)
    print("示例 2: 高级功能 - Multi-Query + Reranking")
    print("="*60)
    
    # 1. 初始化
    vector_store = ChromaVectorStore("advanced_demo")
    vector_store.create_collection(dimension=768)
    rag_engine = AdvancedRAGEngine(vector_store)
    
    # 2. 准备更多文档
    documents = [
        Document(
            id=f"doc{i}",
            content=content,
            metadata={"category": "tech", "index": i}
        )
        for i, content in enumerate([
            "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习。",
            "自然语言处理（NLP）专注于让计算机理解和生成人类语言。",
            "计算机视觉使计算机能够从图像中获取理解，应用包括人脸识别和自动驾驶。",
            "强化学习是机器学习的一种，通过与环境交互来学习最优策略。",
            "深度学习使用多层神经网络，在图像识别和语音识别领域取得突破。",
            "大数据技术用于处理超大规模数据集，包括 Hadoop 和 Spark。",
            "云计算通过互联网提供计算资源，主要提供商包括 AWS 和 Azure。"
        ])
    ]
    
    rag_engine.index_documents(documents)
    
    # 3. 使用高级查询
    request = QueryRequest(
        query="AI 和机器学习有什么关系？",
        top_k=10,
        enable_hybrid=True,
        enable_rerank=True  # 启用重排序
    )
    
    response = rag_engine.query(request, return_answer=True)
    
    print(f"\n检索结果 (Top 3):")
    for i, result in enumerate(response["results"][:3]):
        print(f"\n{i+1}. [分数: {result.score:.4f}]")
        print(f"   {result.document.content}")
    
    print(f"\n生成的答案:")
    print(response["answer"])
    
    # 清理
    vector_store.drop_collection()


def example_parent_child_chunking():
    """父子分块示例"""
    print("\n" + "="*60)
    print("示例 3: 父子分块策略")
    print("="*60)
    
    # 1. 初始化（启用父子分块）
    vector_store = ChromaVectorStore("parent_child_demo")
    vector_store.create_collection(dimension=768)
    rag_engine = AdvancedRAGEngine(vector_store, use_parent_child=True)
    
    # 2. 准备长文档
    long_document = """
    人工智能（Artificial Intelligence, AI）是计算机科学的一个重要分支，旨在创建能够执行通常需要人类智能的任务的系统。
    AI 的历史可以追溯到 20 世纪 50 年代，当时艾伦·图灵提出了著名的"图灵测试"。
    
    机器学习是 AI 的一个核心子领域，它使计算机能够从数据中学习而无需明确编程。
    机器学习算法可以分为三大类：监督学习、无监督学习和强化学习。
    监督学习使用标记的训练数据来学习输入和输出之间的映射关系。
    
    深度学习是机器学习的一个分支，它使用多层神经网络（也称为深度神经网络）来学习数据的层次化表示。
    深度学习在计算机视觉、语音识别和自然语言处理等领域取得了突破性进展。
    卷积神经网络（CNN）特别适合图像处理任务，而循环神经网络（RNN）和 Transformer 架构则在序列数据处理中表现出色。
    
    自然语言处理（NLP）是 AI 和语言学的交叉领域，专注于让计算机理解、解释和生成人类语言。
    近年来，大型语言模型如 GPT、BERT 等的出现，极大地推动了 NLP 领域的发展。
    这些模型通过在海量文本数据上进行预训练，能够理解复杂的语言模式和语义关系。
    """
    
    documents = [
        Document(
            id="long_doc1",
            content=long_document,
            metadata={"category": "tech", "type": "article"}
        )
    ]
    
    # 3. 索引（会自动进行父子分块）
    rag_engine.index_documents(documents)
    
    # 4. 查询（检索到子块，但返回父块）
    request = QueryRequest(
        query="什么是深度学习？",
        top_k=3,
        enable_hybrid=False,
        enable_rerank=False
    )
    
    response = rag_engine.query(request, return_answer=False)
    
    print(f"\n检索结果（返回父块以提供更多上下文）:")
    for i, result in enumerate(response["results"][:2]):
        print(f"\n结果 {i+1}:")
        print(f"  分数: {result.score:.4f}")
        print(f"  内容: {result.document.content[:200]}...")
        print(f"  是否替换为父块: {result.document.metadata.get('replaced_with_parent', False)}")
    
    # 清理
    vector_store.drop_collection()


def main():
    """运行所有示例"""
    print("\n🚀 RAG 系统使用示例")
    
    try:
        # 示例 1: 基础使用
        example_basic_usage()
        
        # 示例 2: 高级功能（需要 DeepSeek API）
        # example_advanced_features()
        
        # 示例 3: 父子分块
        # example_parent_child_chunking()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n请确保:")
        print("  1. 已安装所有依赖: pip install -r requirements.txt")
        print("  2. 已配置 .env 文件（如果使用 DeepSeek API）")


if __name__ == "__main__":
    main()
