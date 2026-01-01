"""
性能评测脚本
"""
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
from typing import List, Dict
from src.vectorstores import MilvusVectorStore, QdrantVectorStore, ChromaVectorStore
from src.rag_engine import AdvancedRAGEngine
from src.core.models import Document, QueryRequest
from src.core.config import settings
import json


class RAGBenchmark:
    """RAG 系统性能评测"""
    
    def __init__(self, num_documents: int = 1000):
        self.num_documents = num_documents
        self.test_documents = self._generate_test_documents()
        self.test_queries = [
            "什么是深度学习？",
            "如何使用 Python 进行数据分析？",
            "人工智能在医疗领域的应用有哪些？",
            "机器学习和深度学习的区别是什么？",
            "自然语言处理的主要技术有哪些？"
        ]
    
    def _generate_test_documents(self) -> List[Document]:
        """生成测试文档"""
        print(f"\n生成 {self.num_documents} 条测试文档...")
        
        # 示例文档模板
        tech_topics = [
            ("深度学习", "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。常见的深度学习框架包括 TensorFlow、PyTorch 和 Keras。"),
            ("数据分析", "数据分析是指通过统计和计算方法来检查、清理、转换和建模数据的过程。Python 是数据分析的热门语言，常用的库包括 Pandas、NumPy 和 Matplotlib。数据分析可以帮助企业做出更好的决策。"),
            ("人工智能", "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。AI 技术包括机器学习、深度学习、自然语言处理和计算机视觉。AI 在医疗、金融、教育等多个领域都有广泛应用。"),
            ("机器学习", "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习而无需明确编程。机器学习算法可以分为监督学习、无监督学习和强化学习。常见的应用包括推荐系统、欺诈检测和预测分析。"),
            ("自然语言处理", "自然语言处理（NLP）是人工智能和语言学的交叉领域，专注于让计算机理解、解释和生成人类语言。NLP 技术包括文本分类、情感分析、机器翻译和问答系统。BERT 和 GPT 是流行的 NLP 模型。"),
            ("计算机视觉", "计算机视觉是使计算机能够从图像或视频中获取高级理解的领域。它包括图像识别、目标检测、语义分割等任务。卷积神经网络（CNN）是计算机视觉中最常用的架构。应用包括自动驾驶和人脸识别。"),
            ("云计算", "云计算是通过互联网提供计算资源和服务的模型。主要的云服务提供商包括 AWS、Azure 和 Google Cloud。云计算提供了可扩展性、灵活性和成本效益，是现代应用开发的基础设施。"),
            ("大数据", "大数据指的是传统数据处理软件无法在合理时间内处理的超大规模数据集。大数据技术包括 Hadoop、Spark 和 NoSQL 数据库。大数据分析可以揭示隐藏的模式和趋势，为业务提供洞察。"),
            ("区块链", "区块链是一种分布式账本技术，可以安全地记录交易。每个区块包含一组交易，并通过加密哈希链接到前一个区块。区块链技术被应用于加密货币、供应链管理和智能合约等领域。"),
            ("物联网", "物联网（IoT）是指通过互联网连接的物理设备网络。这些设备可以收集和交换数据，实现智能家居、工业自动化和智慧城市等应用。IoT 设备通常配备传感器和执行器。")
        ]
        
        documents = []
        for i in range(self.num_documents):
            topic, content = tech_topics[i % len(tech_topics)]
            
            # 添加一些变化
            doc_content = f"{content} 这是第 {i+1} 条文档的补充信息。"
            
            doc = Document(
                id=f"doc_{i:04d}",
                content=doc_content,
                metadata={
                    "category": "tech",
                    "topic": topic,
                    "index": i,
                    "source": "benchmark"
                }
            )
            documents.append(doc)
        
        print(f"✓ 成功生成 {len(documents)} 条文档")
        return documents
    
    def benchmark_vector_store(
        self,
        store_name: str,
        vector_store
    ) -> Dict:
        """
        评测单个向量数据库
        
        Args:
            store_name: 数据库名称
            vector_store: 向量数据库实例
        
        Returns:
            评测结果字典
        """
        print(f"\n{'='*60}")
        print(f"评测: {store_name}")
        print(f"{'='*60}")
        
        results = {
            "name": store_name,
            "index_time": 0,
            "search_time": 0,
            "avg_search_time": 0,
            "errors": []
        }
        
        try:
            # 1. 创建集合
            vector_store.create_collection(dimension=settings.embedding_dimension)
            
            # 2. 索引文档
            print(f"\n索引 {self.num_documents} 条文档...")
            rag_engine = AdvancedRAGEngine(vector_store, use_parent_child=False)
            
            start_time = time.time()
            rag_engine.index_documents(self.test_documents, show_progress=True)
            index_time = time.time() - start_time
            results["index_time"] = index_time
            
            print(f"✓ 索引完成，耗时: {index_time:.2f} 秒")
            
            # 3. 检索测试
            print(f"\n执行 {len(self.test_queries)} 次检索...")
            search_times = []
            
            for i, query in enumerate(self.test_queries):
                start_time = time.time()
                search_results = rag_engine.search(
                    query=query,
                    top_k=10,
                    enable_hybrid=True
                )
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                print(f"  查询 {i+1}: '{query[:30]}...' - {search_time:.3f}秒 - {len(search_results)} 个结果")
            
            total_search_time = sum(search_times)
            avg_search_time = total_search_time / len(search_times)
            
            results["search_time"] = total_search_time
            results["avg_search_time"] = avg_search_time
            
            print(f"\n✓ 平均检索时间: {avg_search_time:.3f} 秒")
            
        except Exception as e:
            error_msg = f"评测失败: {str(e)}"
            print(f"✗ {error_msg}")
            results["errors"].append(error_msg)
        
        return results
    
    def benchmark_reranking(self):
        """评测重排序效果"""
        print(f"\n{'='*60}")
        print("评测重排序效果")
        print(f"{'='*60}")
        
        # 使用 Chroma 进行测试（最简单）
        vector_store = ChromaVectorStore("rerank_test")
        vector_store.create_collection(dimension=settings.embedding_dimension)
        
        rag_engine = AdvancedRAGEngine(vector_store, use_parent_child=False)
        
        # 只索引前100条文档
        test_docs = self.test_documents[:100]
        rag_engine.index_documents(test_docs, show_progress=False)
        
        query = self.test_queries[0]
        print(f"\n测试查询: {query}")
        
        # 不启用重排序
        print("\n【不使用重排序】")
        results_no_rerank = rag_engine.search(
            query=query,
            top_k=10,
            enable_hybrid=True
        )
        
        print("Top 5 结果:")
        for i, result in enumerate(results_no_rerank[:5]):
            print(f"  {i+1}. [分数: {result.score:.4f}] {result.document.content[:80]}...")
        
        # 启用重排序
        print("\n【使用 DeepSeek 重排序】")
        results_reranked = rag_engine.rerank(query, results_no_rerank[:10], top_k=5)
        
        print("Top 5 结果:")
        for i, result in enumerate(results_reranked):
            print(f"  {i+1}. [分数: {result.score:.4f}] {result.document.content[:80]}...")
        
        # 清理
        vector_store.drop_collection()
    
    def run_full_benchmark(self):
        """运行完整评测"""
        print("\n" + "="*60)
        print("RAG 系统完整性能评测")
        print("="*60)
        
        all_results = []
        
        # 评测 Chroma（最容易设置，不需要额外服务）
        print("\n\n>>> 评测 Chroma <<<")
        try:
            chroma_store = ChromaVectorStore("benchmark_chroma")
            chroma_results = self.benchmark_vector_store("Chroma", chroma_store)
            all_results.append(chroma_results)
            chroma_store.drop_collection()
        except Exception as e:
            print(f"✗ Chroma 评测跳过: {e}")
        
        # 评测 Qdrant（需要 Qdrant 服务运行）
        print("\n\n>>> 评测 Qdrant <<<")
        try:
            qdrant_store = QdrantVectorStore("benchmark_qdrant")
            qdrant_results = self.benchmark_vector_store("Qdrant", qdrant_store)
            all_results.append(qdrant_results)
            qdrant_store.drop_collection()
        except Exception as e:
            print(f"✗ Qdrant 评测跳过: {e}")
        
        # 评测 Milvus（需要 Milvus 服务运行）
        print("\n\n>>> 评测 Milvus <<<")
        try:
            milvus_store = MilvusVectorStore("benchmark_milvus")
            milvus_results = self.benchmark_vector_store("Milvus", milvus_store)
            all_results.append(milvus_results)
            milvus_store.drop_collection()
        except Exception as e:
            print(f"✗ Milvus 评测跳过: {e}")
        
        # 打印汇总报告
        self._print_summary(all_results)
        
        # 评测重排序效果
        try:
            self.benchmark_reranking()
        except Exception as e:
            print(f"✗ 重排序评测失败: {e}")
    
    def _print_summary(self, results: List[Dict]):
        """打印评测汇总"""
        print("\n\n" + "="*60)
        print("评测汇总报告")
        print("="*60)
        
        if not results:
            print("没有成功完成的评测")
            return
        
        print(f"\n测试配置:")
        print(f"  - 文档数量: {self.num_documents}")
        print(f"  - 查询数量: {len(self.test_queries)}")
        print(f"  - 向量维度: {settings.embedding_dimension}")
        
        print(f"\n性能对比:")
        print(f"{'数据库':<15} {'索引时间(秒)':<15} {'平均检索时间(秒)':<20}")
        print("-" * 60)
        
        for result in results:
            if not result.get("errors"):
                print(f"{result['name']:<15} {result['index_time']:<15.2f} {result['avg_search_time']:<20.3f}")
        
        # 找出最快的
        if results:
            fastest_index = min(results, key=lambda x: x.get('index_time', float('inf')))
            fastest_search = min(results, key=lambda x: x.get('avg_search_time', float('inf')))
            
            print(f"\n🏆 索引速度最快: {fastest_index['name']} ({fastest_index['index_time']:.2f}秒)")
            print(f"🏆 检索速度最快: {fastest_search['name']} ({fastest_search['avg_search_time']:.3f}秒)")


def main():
    """主函数"""
    # 创建评测实例
    benchmark = RAGBenchmark(num_documents=1000)
    
    # 运行完整评测
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
