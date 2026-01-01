"""
快速启动脚本 - 无需配置即可体验基础功能
"""
import os
import sys


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "pydantic",
        "sentence_transformers",
        "pymilvus",
        "rank_bm25"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ 缺少以下依赖包:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    
    return True


def demo_basic_rag():
    """演示基础 RAG 功能（无需 DeepSeek API）"""
    print("\n" + "="*60)
    print("🚀 RAG 系统快速体验 - 基础版（Milvus）")
    print("="*60)
    
    import sys
    import os
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.vectorstores import MilvusVectorStore
    from src.rag_engine import AdvancedRAGEngine
    from src.core.models import Document
    
    # 1. 初始化
    print("\n[1/4] 初始化 Milvus 向量数据库...")
    print("   连接地址: localhost:19530")
    vector_store = MilvusVectorStore(collection_name="quick_start_demo")
    vector_store.create_collection(dimension=768)
    
    # 2. 创建 RAG 引擎（不使用父子分块，简化流程）
    print("[2/4] 创建 RAG 引擎...")
    rag_engine = AdvancedRAGEngine(vector_store, use_parent_child=False)
    
    # 3. 准备示例文档
    print("[3/4] 准备示例文档...")
    documents = [
        Document(
            id="doc1",
            content="深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。常见的深度学习框架包括 TensorFlow、PyTorch 和 Keras。",
            metadata={"category": "tech", "topic": "深度学习"}
        ),
        Document(
            id="doc2",
            content="Python 是一种高级编程语言，以其简洁的语法和强大的功能而闻名。Python 广泛应用于数据科学、机器学习、Web 开发和自动化脚本等领域。它有丰富的第三方库生态系统，如 NumPy、Pandas 和 Scikit-learn。",
            metadata={"category": "tech", "topic": "编程语言"}
        ),
        Document(
            id="doc3",
            content="向量数据库是专门用于存储和检索高维向量数据的数据库系统。它们使用特殊的索引结构（如 HNSW、IVF）来实现高效的相似度搜索。常见的向量数据库包括 Milvus、Qdrant 和 Chroma，广泛应用于推荐系统和语义搜索。",
            metadata={"category": "tech", "topic": "数据库"}
        ),
        Document(
            id="doc4",
            content="RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。它首先从知识库中检索相关文档，然后将这些文档作为上下文传递给大语言模型，从而生成更准确、更有依据的答案。RAG 可以有效减少大模型的幻觉问题。",
            metadata={"category": "tech", "topic": "RAG"}
        ),
        Document(
            id="doc5",
            content="自然语言处理（NLP）是人工智能和语言学的交叉领域，专注于让计算机理解、解释和生成人类语言。NLP 技术包括分词、词性标注、命名实体识别、情感分析、机器翻译和问答系统等。BERT、GPT 等预训练模型极大地推动了 NLP 的发展。",
            metadata={"category": "tech", "topic": "NLP"}
        ),
    ]
    
    print(f"✓ 准备了 {len(documents)} 条示例文档")
    
    # 4. 索引文档
    print("[4/4] 索引文档到向量数据库...")
    rag_engine.index_documents(documents, show_progress=False)
    
    # 5. 执行检索测试
    print("\n" + "="*60)
    print("📝 检索测试")
    print("="*60)
    
    test_queries = [
        "什么是深度学习？",
        "如何使用 Python？",
        "向量数据库有哪些？",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n【查询 {i}】: {query}")
        print("-" * 60)
        
        # 执行检索（不启用需要 API 的功能）
        results = rag_engine.search(
            query=query,
            top_k=3,
            enable_hybrid=True,      # 混合检索（BM25 + 向量）
            enable_multi_query=False, # 关闭（需要 DeepSeek API）
            enable_hyde=False         # 关闭（需要 DeepSeek API）
        )
        
        print(f"\n找到 {len(results)} 条相关结果:\n")
        for j, result in enumerate(results[:2], 1):
            print(f"  [{j}] 相似度: {result.score:.4f}")
            print(f"      话题: {result.document.metadata.get('topic', 'N/A')}")
            print(f"      内容: {result.document.content[:100]}...")
            print()
    
    # 6. 清理
    print("\n" + "="*60)
    print("🧹 清理测试数据...")
    vector_store.drop_collection()
    print("✓ 清理完成")
    
    print("\n" + "="*60)
    print("🎉 快速体验完成！Milvus 向量数据库运行正常")
    print("="*60)
    print("\n💡 下一步:")
    print("  1. 查看 README.md 了解完整功能")
    print("  2. 配置 .env 文件以启用 DeepSeek API 功能")
    print("  3. 运行 python examples/example_usage.py 查看更多示例")
    print("  4. 运行 python tests/benchmark.py 进行性能评测")
    print()


def demo_with_api():
    """演示完整功能（需要 DeepSeek API）"""
    print("\n" + "="*60)
    print("🚀 RAG 系统完整体验 - 高级版")
    print("="*60)
    
    # 检查 API 配置
    try:
        import sys
        import os
        # 获取项目根目录的绝对路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from src.core.config import settings
        if not settings.deepseek_api_key or settings.deepseek_api_key == "your_deepseek_api_key_here":
            print("\n❌ 未配置 DeepSeek API Key")
            print("请在 .env 文件中设置 DEEPSEEK_API_KEY")
            return
    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        return
    
    from src.vectorstores import MilvusVectorStore
    from src.rag_engine import AdvancedRAGEngine
    from src.core.models import Document, QueryRequest
    
    # 初始化
    print("\n正在初始化 Milvus...")
    vector_store = MilvusVectorStore(collection_name="advanced_demo")
    vector_store.create_collection(dimension=768)
    rag_engine = AdvancedRAGEngine(vector_store)
    
    # 准备文档
    documents = [
        Document(
            id=f"doc{i}",
            content=content,
            metadata={"category": "tech", "index": i}
        )
        for i, content in enumerate([
            "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。",
            "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习而无需明确编程。",
            "深度学习使用多层神经网络，在图像识别、语音识别和自然语言处理领域取得突破。",
            "自然语言处理（NLP）专注于让计算机理解和生成人类语言，应用包括机器翻译和问答系统。",
            "RAG（检索增强生成）结合了检索和生成技术，可以有效减少大模型的幻觉问题。",
        ])
    ]
    
    print("正在索引文档...")
    rag_engine.index_documents(documents, show_progress=False)
    
    # 测试查询
    print("\n" + "="*60)
    print("📝 高级检索测试（Multi-Query + Reranking）")
    print("="*60)
    
    query = "人工智能和机器学习有什么关系？"
    print(f"\n查询: {query}\n")
    
    request = QueryRequest(
        query=query,
        top_k=10,
        enable_hybrid=True,
        enable_rerank=True
    )
    
    print("正在检索并生成答案...\n")
    response = rag_engine.query(request, return_answer=True)
    
    print("\n【检索结果】")
    for i, result in enumerate(response["results"][:3], 1):
        print(f"{i}. [分数: {result.score:.4f}] {result.document.content[:80]}...")
    
    print("\n【生成的答案】")
    print(response["answer"])
    
    # 清理
    vector_store.drop_collection()
    print("\n✓ 完成")


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║         🤖 高级 RAG 系统 - 快速启动                        ║
║                                                           ║
║    一个功能强大的 RAG 系统，支持多种向量数据库              ║
║    和先进的检索优化技术                                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    print("\n请选择体验模式:")
    print("  [1] 基础版 - 无需 API 配置，快速体验核心功能")
    print("  [2] 高级版 - 需要 DeepSeek API，体验完整功能")
    print("  [Q] 退出")
    
    choice = input("\n请输入选择 (1/2/Q): ").strip().lower()
    
    if choice == "1":
        demo_basic_rag()
    elif choice == "2":
        demo_with_api()
    elif choice in ["q", "quit", "exit"]:
        print("再见！👋")
    else:
        print("无效的选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
