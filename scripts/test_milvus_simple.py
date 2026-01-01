"""
Milvus 向量数据库测试脚本
确保 Milvus 已通过 docker-compose 启动
"""
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 60)
print("Milvus 向量数据库测试")
print("=" * 60)
print()

try:
    # 1. 测试连接
    print("[1/5] 测试 Milvus 连接...")
    from pymilvus import connections
    
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )
    print("✓ Milvus 连接成功！")
    
    # 2. 导入模块
    print("\n[2/5] 导入 RAG 模块...")
    from src.vectorstores import MilvusVectorStore
    from src.rag_engine import AdvancedRAGEngine
    from src.core.models import Document
    print("✓ 模块导入成功")
    
    # 3. 创建数据库
    print("\n[3/5] 创建 Milvus 集合...")
    vector_store = MilvusVectorStore("test_collection")
    vector_store.create_collection(dimension=768)
    print("✓ 集合创建成功")
    
    # 4. 创建 RAG 引擎并索引
    print("\n[4/5] 索引测试文档...")
    rag = AdvancedRAGEngine(vector_store)
    
    docs = [
        Document(
            id="doc1",
            content="Python 是一种高级编程语言，广泛用于数据科学和机器学习",
            metadata={"category": "tech", "source": "test"}
        ),
        Document(
            id="doc2",
            content="深度学习使用多层神经网络来学习数据的表示",
            metadata={"category": "ai", "source": "test"}
        ),
        Document(
            id="doc3",
            content="Milvus 是一个开源的向量数据库，专为 AI 应用设计",
            metadata={"category": "database", "source": "test"}
        ),
        Document(
            id="doc4",
            content="RAG 结合了检索和生成技术，提高了 AI 回答的准确性",
            metadata={"category": "ai", "source": "test"}
        ),
        Document(
            id="doc5",
            content="向量数据库使用相似度搜索来找到语义相关的文档",
            metadata={"category": "database", "source": "test"}
        ),
    ]
    
    rag.index_documents(docs, show_progress=False)
    print(f"✓ 索引完成，共 {len(docs)} 条文档")
    
    # 5. 测试查询
    print("\n[5/5] 测试查询功能...")
    
    # 测试 1: 基础向量检索
    print("\n【测试 1: 基础向量检索】")
    query1 = "什么是深度学习？"
    print(f"查询: {query1}")
    results1 = rag.search(query1, top_k=2, enable_hybrid=False)
    
    print("检索结果:")
    for i, result in enumerate(results1, 1):
        print(f"  {i}. [分数: {result.score:.4f}] {result.document.content}")
    
    # 测试 2: 混合检索
    print("\n【测试 2: 混合检索（向量 + BM25）】")
    query2 = "向量数据库"
    print(f"查询: {query2}")
    results2 = rag.search(query2, top_k=2, enable_hybrid=True)
    
    print("检索结果:")
    for i, result in enumerate(results2, 1):
        print(f"  {i}. [分数: {result.score:.4f}] {result.document.content}")
    
    # 测试 3: 元数据过滤
    print("\n【测试 3: 元数据过滤】")
    query3 = "技术"
    filters = {"category": "ai"}
    print(f"查询: {query3}")
    print(f"过滤: category == 'ai'")
    results3 = rag.search(query3, top_k=3, filters=filters, enable_hybrid=False)
    
    print("检索结果:")
    for i, result in enumerate(results3, 1):
        print(f"  {i}. [分数: {result.score:.4f}] {result.document.content}")
        print(f"      类别: {result.document.metadata.get('category')}")
    
    # 6. 获取统计信息
    print("\n" + "=" * 60)
    print("集合统计信息:")
    stats = vector_store.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 7. 清理
    print("\n" + "=" * 60)
    print("✓✓✓ 所有测试通过！Milvus 运行正常！✓✓✓")
    print("=" * 60)
    
    # 询问是否清理数据
    print("\n清理测试数据？")
    cleanup = input("输入 'y' 删除测试集合，或直接回车保留: ").strip().lower()
    
    if cleanup == 'y':
        vector_store.drop_collection()
        print("✓ 测试集合已删除")
    else:
        print("ℹ 测试集合已保留，集合名: test_collection")
    
    # 断开连接
    connections.disconnect()
    
    print("\n💡 下一步:")
    print("   1. 运行完整示例: cd examples && python quick_start.py")
    print("   2. 运行性能评测: cd tests && python benchmark.py")
    print("   3. 修改代码使用 MilvusVectorStore 开始开发")
    print()
    
except ModuleNotFoundError as e:
    print(f"\n✗ 模块导入错误: {e}")
    print("\n请确保:")
    print("  1. 已安装依赖: pip install -r requirements.txt")
    print("  2. 在项目根目录或 scripts 目录运行此脚本")
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    print("\n请检查:")
    print("  1. Milvus 服务是否正在运行")
    print("     查看状态: cd config && docker-compose ps")
    print("  2. 端口 19530 是否可访问")
    print("  3. 查看 Milvus 日志: cd config && docker-compose logs standalone")
    
    import traceback
    traceback.print_exc()
