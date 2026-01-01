"""
混合检索（向量检索 + BM25）
"""
from typing import List, Dict, Any
from src.core.models import SearchResult


class HybridSearchEngine:
    """混合检索引擎"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: List[SearchResult],
        bm25_results: List[tuple],
        k: int = 60,
        vector_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        RRF (Reciprocal Rank Fusion) 算法融合检索结果
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果 [(doc_id, score), ...]
            k: RRF 参数
            vector_weight: 向量检索权重（0-1之间）
        
        Returns:
            融合后的检索结果
        """
        # 构建文档 ID 到 SearchResult 的映射
        doc_map = {result.document.id: result for result in vector_results}
        
        # 计算 RRF 分数
        rrf_scores = {}
        
        # 向量检索贡献
        for rank, result in enumerate(vector_results):
            doc_id = result.document.id
            rrf_scores[doc_id] = vector_weight / (k + rank + 1)
        
        # BM25 贡献
        bm25_weight = 1.0 - vector_weight
        for rank, (doc_id, bm25_score) in enumerate(bm25_results):
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += bm25_weight / (k + rank + 1)
            else:
                # 如果 BM25 找到了向量检索没找到的文档
                # 注意：这种情况下我们没有完整的文档信息，所以跳过
                pass
        
        # 按 RRF 分数排序
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # 构建最终结果
        final_results = []
        for rank, doc_id in enumerate(sorted_doc_ids):
            result = doc_map[doc_id]
            # 更新分数和排名
            result.score = rrf_scores[doc_id]
            result.rank = rank + 1
            final_results.append(result)
        
        return final_results
    
    @staticmethod
    def normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
        """
        归一化检索分数到 [0, 1] 区间
        
        Args:
            results: 检索结果列表
        
        Returns:
            归一化后的结果
        """
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for result in results:
                result.score = 1.0
        else:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
