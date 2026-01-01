"""
高级 RAG 引擎
整合所有功能的主引擎
"""
from typing import List, Optional, Dict, Any
from src.vectorstores.vector_store_base import RAGVectorStore
from src.core.models import Document, SearchResult, QueryRequest
from src.llm.deepseek_client import DeepSeekClient
from src.llm.embedding_manager import EmbeddingManager
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_search import HybridSearchEngine
from src.retrievers.chunking_strategy import ChunkingStrategy
from tqdm import tqdm


class AdvancedRAGEngine:
    """高级 RAG 引擎"""
    
    def __init__(
        self,
        vector_store: RAGVectorStore,
        use_parent_child: bool = False
    ):
        """
        初始化 RAG 引擎
        
        Args:
            vector_store: 向量数据库实例
            use_parent_child: 是否使用父子分块策略
        """
        self.vector_store = vector_store
        self.use_parent_child = use_parent_child
        self.child_to_parent: Dict[str, str] = {}  # 子块到父块的映射
        
        # 初始化各个组件
        self.embedding_manager = EmbeddingManager()
        self.deepseek_client = DeepSeekClient()
        self.bm25_retriever = BM25Retriever()
        self.hybrid_engine = HybridSearchEngine()
        
        print("✓ RAG 引擎初始化完成")
    
    def index_documents(
        self,
        documents: List[Document],
        show_progress: bool = True
    ) -> bool:
        """
        索引文档到向量数据库
        
        Args:
            documents: 文档列表
            show_progress: 是否显示进度条
        
        Returns:
            是否成功
        """
        print(f"\n开始索引 {len(documents)} 条文档...")
        
        # 如果使用父子分块
        if self.use_parent_child:
            all_child_docs = []
            for doc in tqdm(documents, desc="分块处理", disable=not show_progress):
                child_docs, child_to_parent = ChunkingStrategy.create_parent_child_documents(
                    doc_id=doc.id,
                    text=doc.content,
                    metadata=doc.metadata
                )
                all_child_docs.extend(child_docs)
                self.child_to_parent.update(child_to_parent)
            
            documents = all_child_docs
            print(f"父子分块后共 {len(documents)} 个子块")
        
        # 生成 embeddings
        texts = [doc.content for doc in documents]
        print("正在生成 embeddings...")
        embeddings = self.embedding_manager.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress
        )
        
        # 将 embeddings 赋值给文档
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        # 插入向量数据库
        success = self.vector_store.batch_upsert(documents)
        
        # 为 BM25 索引准备数据
        bm25_docs = [
            {"id": doc.id, "content": doc.content}
            for doc in documents
        ]
        self.bm25_retriever.index_documents(bm25_docs)
        
        return success
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        enable_hybrid: bool = True,
        enable_multi_query: bool = False,
        enable_hyde: bool = False
    ) -> List[SearchResult]:
        """
        高级检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 元数据过滤
            enable_hybrid: 是否启用混合检索
            enable_multi_query: 是否启用 Multi-Query
            enable_hyde: 是否启用 HyDE
        
        Returns:
            检索结果列表
        """
        queries_to_search = []
        
        # Multi-Query: 查询扩展
        if enable_multi_query:
            print("🔄 Multi-Query: 生成同义查询...")
            queries_to_search = self.deepseek_client.generate_multi_queries(query, num_queries=3)
            print(f"生成的查询: {queries_to_search}")
        else:
            queries_to_search = [query]
        
        # HyDE: 假设性文档生成
        if enable_hyde:
            print("🔄 HyDE: 生成假设性文档...")
            hypothetical_doc = self.deepseek_client.generate_hypothetical_document(query)
            print(f"假设性文档: {hypothetical_doc[:200]}...")
            queries_to_search.append(hypothetical_doc)
        
        # 对所有查询进行检索
        all_results = []
        for q in queries_to_search:
            if enable_hybrid:
                results = self._hybrid_search(q, top_k, filters)
            else:
                results = self._vector_search(q, top_k, filters)
            all_results.extend(results)
        
        # 去重并合并结果
        unique_results = self._merge_results(all_results)
        
        # 如果使用父子分块，替换为父块内容
        if self.use_parent_child:
            unique_results = self._replace_with_parent(unique_results)
        
        return unique_results[:top_k]
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """纯向量检索"""
        query_embedding = self.embedding_manager.encode(query)
        return self.vector_store.search(query_embedding, top_k, filters)
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """混合检索（向量 + BM25）"""
        # 向量检索
        vector_results = self._vector_search(query, top_k, filters)
        
        # BM25 检索
        bm25_results = self.bm25_retriever.search(query, top_k)
        
        # RRF 融合
        hybrid_results = self.hybrid_engine.reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            vector_weight=0.6
        )
        
        return hybrid_results
    
    def _merge_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """合并和去重检索结果"""
        # 使用文档ID去重
        doc_map = {}
        for result in results:
            doc_id = result.document.id
            if doc_id not in doc_map:
                doc_map[doc_id] = result
            else:
                # 取更高的分数
                if result.score > doc_map[doc_id].score:
                    doc_map[doc_id] = result
        
        # 按分数排序
        merged = list(doc_map.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
    
    def _replace_with_parent(self, results: List[SearchResult]) -> List[SearchResult]:
        """将子块替换为父块"""
        replaced_results = []
        for result in results:
            child_id = result.document.id
            if child_id in self.child_to_parent:
                # 替换为父块内容
                result.document.content = self.child_to_parent[child_id]
                result.document.metadata["replaced_with_parent"] = True
            replaced_results.append(result)
        return replaced_results
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        使用 DeepSeek 重排序
        
        Args:
            query: 查询文本
            results: 检索结果
            top_k: 最终返回数量
        
        Returns:
            重排序后的结果
        """
        if not results:
            return []
        
        print(f"🔄 正在使用 DeepSeek 重排序 {len(results)} 条结果...")
        
        documents = [r.document.content for r in results]
        rerank_scores = self.deepseek_client.rerank_documents(query, documents, top_k)
        
        # 根据新的排名重新组织结果
        reranked_results = []
        for idx, score in rerank_scores:
            result = results[idx]
            result.score = score
            result.rank = len(reranked_results) + 1
            reranked_results.append(result)
        
        return reranked_results
    
    def query(
        self,
        request: QueryRequest,
        return_answer: bool = True
    ) -> Dict[str, Any]:
        """
        完整的查询流程
        
        Args:
            request: 查询请求
            return_answer: 是否生成最终答案
        
        Returns:
            包含检索结果和答案的字典
        """
        print(f"\n{'='*60}")
        print(f"查询: {request.query}")
        print(f"{'='*60}")
        
        # 1. 检索
        results = self.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            enable_hybrid=request.enable_hybrid,
            enable_multi_query=True,
            enable_hyde=False
        )
        
        print(f"✓ 检索到 {len(results)} 条结果")
        
        # 2. 重排序
        if request.enable_rerank and len(results) > 0:
            results = self.rerank(request.query, results, top_k=5)
            print(f"✓ 重排序完成，保留前 {len(results)} 条")
        
        # 3. 生成答案
        answer = ""
        if return_answer and len(results) > 0:
            print("🤖 正在生成答案...")
            contexts = [r.document.content for r in results[:5]]
            answer = self.deepseek_client.answer_with_context(request.query, contexts)
        
        return {
            "query": request.query,
            "results": results,
            "answer": answer,
            "num_results": len(results)
        }
