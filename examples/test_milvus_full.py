"""
Milvus 完整功能测试
包含所有高级功能：混合检索、Multi-Query、重排序等
"""
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("\n" + "="*70)
print("🚀 Milvus RAG 系统完整功能测试")
print("="*70)

try:
    from src.vectorstores import MilvusVectorStore
    from src.rag_engine import AdvancedRAGEngine
    from src.core.models import Document, QueryRequest
    
    print("\n✓ 模块导入成功")
    
    # 1. 初始化
    print("\n" + "="*70)
    print("📦 [1/4] 初始化 Milvus 向量数据库")
    print("="*70)
    
    vector_store = MilvusVectorStore("rag_demo_collection")
    vector_store.create_collection(dimension=768)
    print("✓ Milvus 集合创建成功")
    
    # 2. 准备测试数据
    print("\n" + "="*70)
    print("📝 [2/4] 准备测试文档")
    print("="*70)
    
    documents = [
        Document(
            id="doc1",
            content="人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。AI 技术包括机器学习、深度学习、自然语言处理和计算机视觉。",
            metadata={"category": "tech", "topic": "人工智能", "difficulty": "beginner"}
        ),
        Document(
            id="doc2",
            content="机器学习是人工智能的一个核心子领域，它使计算机能够从数据中学习而无需明确编程。常见的机器学习算法包括决策树、随机森林、支持向量机和神经网络。",
            metadata={"category": "tech", "topic": "机器学习", "difficulty": "intermediate"}
        ),
        Document(
            id="doc3",
            content="深度学习是机器学习的一个分支，使用多层神经网络（也称为深度神经网络）来学习数据的层次化表示。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。",
            metadata={"category": "tech", "topic": "深度学习", "difficulty": "advanced"}
        ),
        Document(
            id="doc4",
            content="自然语言处理（NLP）是人工智能和语言学的交叉领域，专注于让计算机理解、解释和生成人类语言。NLP 技术包括分词、词性标注、命名实体识别、情感分析和机器翻译。",
            metadata={"category": "tech", "topic": "NLP", "difficulty": "intermediate"}
        ),
        Document(
            id="doc5",
            content="向量数据库是专门用于存储和检索高维向量数据的数据库系统。它们使用特殊的索引结构（如 HNSW、IVF）来实现高效的相似度搜索，广泛应用于推荐系统和语义搜索。",
            metadata={"category": "tech", "topic": "向量数据库", "difficulty": "intermediate"}
        ),
        Document(
            id="doc6",
            content="RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。它首先从知识库中检索相关文档，然后将这些文档作为上下文传递给大语言模型，从而生成更准确、更有依据的答案。",
            metadata={"category": "tech", "topic": "RAG", "difficulty": "advanced"}
        ),
        Document(
            id="doc7",
            content="Python 是一种高级编程语言，以其简洁的语法和强大的功能而闻名。Python 广泛应用于数据科学、机器学习、Web 开发和自动化脚本等领域。",
            metadata={"category": "tech", "topic": "编程语言", "difficulty": "beginner"}
        ),
        Document(
            id="doc8",
            content="Transformer 是一种基于自注意力机制的深度学习架构，最初用于自然语言处理任务。BERT、GPT 等现代大语言模型都基于 Transformer 架构。",
            metadata={"category": "tech", "topic": "深度学习", "difficulty": "advanced"}
        ),
    ]
    
    print(f"准备了 {len(documents)} 条测试文档")
    
    # 3. 索引文档
    print("\n" + "="*70)
    print("🔄 [3/4] 索引文档到 Milvus")
    print("="*70)
    
    rag_engine = AdvancedRAGEngine(vector_store)
    rag_engine.index_documents(documents, show_progress=True)
    
    print("✓ 文档索引完成")
    
    # 4. 测试各种检索功能
    print("\n" + "="*70)
    print("🔍 [4/4] 测试检索功能")
    print("="*70)
    
    # 测试 1: 基础向量检索
    print("\n" + "-"*70)
    print("【测试 1: 基础向量检索】")
    print("-"*70)
    query = "什么是深度学习？"
    print(f"查询: {query}\n")
    
    results = rag_engine.search(
        query=query,
        top_k=3,
        enable_hybrid=False,
        enable_multi_query=False
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result.score:.4f}]")
        print(f"   内容: {result.document.content[:100]}...")
        print(f"   主题: {result.document.metadata.get('topic')}")
        print()
    
    # 测试 2: 混合检索
    print("\n" + "-"*70)
    print("【测试 2: 混合检索（向量 + BM25）】")
    print("-"*70)
    query = "向量数据库 检索"
    print(f"查询: {query}\n")
    
    results = rag_engine.search(
        query=query,
        top_k=3,
        enable_hybrid=True,
        enable_multi_query=False
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result.score:.4f}]")
        print(f"   内容: {result.document.content[:100]}...")
        print()
    
    # 测试 3: 元数据过滤
    print("\n" + "-"*70)
    print("【测试 3: 元数据过滤】")
    print("-"*70)
    query = "技术"
    filters = {"category": "tech"}
    print(f"查询: {query}")
    print(f"过滤条件: category == 'tech'\n")
    
    results = rag_engine.search(
        query=query,
        top_k=3,
        filters=filters,
        enable_hybrid=False
    )
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [分数: {result.score:.4f}]")
        print(f"   内容: {result.document.content[:80]}...")
        print(f"   难度: {result.document.metadata.get('difficulty')}")
        print()
    
    # 测试 4: 完整 RAG 查询（如果配置了 DeepSeek API）
    print("\n" + "-"*70)
    print("【测试 4: 完整 RAG 查询】")
    print("-"*70)
    
    try:
        from src.core.config import settings
        if settings.deepseek_api_key and settings.deepseek_api_key != "your_deepseek_api_key_here":
            query = "RAG 技术是如何工作的？"
            print(f"查询: {query}\n")
            
            request = QueryRequest(
                query=query,
                top_k=10,
                enable_hybrid=True,
                enable_rerank=False  # 不启用重排序以节省时间
            )
            
            response = rag_engine.query(request, return_answer=False)
            
            print(f"检索到 {len(response['results'])} 条相关文档\n")
            print("Top 3 结果:")
            for i, result in enumerate(response['results'][:3], 1):
                print(f"{i}. [分数: {result.score:.4f}]")
                print(f"   {result.document.content[:100]}...")
                print()
        else:
            print("ℹ 未配置 DeepSeek API，跳过高级功能测试")
            print("  配置方法: 在 .env 文件中设置 DEEPSEEK_API_KEY")
    except:
        print("ℹ 跳过 DeepSeek API 测试")
    
    # 统计信息
    print("\n" + "="*70)
    print("📊 集合统计信息")
    print("="*70)
    stats = vector_store.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 完成
    print("\n" + "="*70)
    print("✅ ✅ ✅  所有测试完成！Milvus 工作正常！ ✅ ✅ ✅")
    print("="*70)
    
    # 清理选项
    print("\n是否删除测试集合？")
    cleanup = input("输入 'y' 删除，或直接回车保留: ").strip().lower()
    
    if cleanup == 'y':
        vector_store.drop_collection()
        print("✓ 测试集合已删除")
    else:
        print(f"ℹ 测试集合已保留: rag_demo_collection")
        print(f"  包含 {len(documents)} 条文档")
    
    print("\n🎉 测试完成！\n")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 故障排查:")
    print("  1. 确认 Milvus 正在运行: cd config && docker-compose ps")
    print("  2. 查看 Milvus 日志: cd config && docker-compose logs standalone")
    print("  3. 重启服务: cd scripts && ./start_milvus.bat")
