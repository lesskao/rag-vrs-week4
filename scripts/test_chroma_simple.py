"""
Chroma 快速测试 - 无需 Docker！
"""
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 60)
print("Chroma 向量数据库测试")
print("=" * 60)
print()

try:
    # 1. 导入
    print("[1/4] 导入模块...")
    from src.vectorstores import ChromaVectorStore
    from src.rag_engine import AdvancedRAGEngine
    from src.core.models import Document
    print("✓ 导入成功")
    
    # 2. 创建数据库
    print("\n[2/4] 创建向量数据库...")
    vector_store = ChromaVectorStore("test_collection")
    vector_store.create_collection(dimension=768)
    print("✓ 数据库创建成功（无需 Docker！）")
    
    # 3. 创建 RAG 引擎并索引
    print("\n[3/4] 索引测试文档...")
    rag = AdvancedRAGEngine(vector_store)
    
    docs = [
        Document(
            id="1",
            content="Python 是一种高级编程语言，广泛用于数据科学和机器学习",
            metadata={"category": "tech"}
        ),
        Document(
            id="2",
            content="深度学习使用多层神经网络来学习数据的表示",
            metadata={"category": "ai"}
        ),
        Document(
            id="3",
            content="RAG 结合了检索和生成技术，提高了 AI 回答的准确性",
            metadata={"category": "ai"}
        ),
    ]
    
    rag.index_documents(docs, show_progress=False)
    print("✓ 索引完成，共 3 条文档")
    
    # 4. 测试查询
    print("\n[4/4] 测试查询...")
    results = rag.search("什么是深度学习？", top_k=2, enable_hybrid=True)
    
    print("\n查询结果:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  相似度: {result.score:.4f}")
        print(f"  内容: {result.document.content}")
        print(f"  分类: {result.document.metadata.get('category', 'N/A')}")
    
    # 5. 清理
    print("\n" + "=" * 60)
    print("✓✓✓ 测试完成！Chroma 运行正常！✓✓✓")
    print("=" * 60)
    
    vector_store.drop_collection()
    
    print("\n💡 下一步:")
    print("   1. 运行: cd examples && python quick_start.py")
    print("   2. 在你的代码中使用 ChromaVectorStore")
    print("   3. 享受零配置的便利！")
    print()
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    print("\n请确保:")
    print("  1. 已安装依赖: pip install -r requirements.txt")
    print("  2. 在项目根目录运行此脚本")
    import traceback
    traceback.print_exc()
