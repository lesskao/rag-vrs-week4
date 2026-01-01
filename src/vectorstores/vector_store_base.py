"""
向量数据库抽象接口定义
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.core.models import Document, SearchResult


class RAGVectorStore(ABC):
    """RAG 向量数据库抽象接口类"""
    
    def __init__(self, collection_name: str = "rag_collection"):
        """
        初始化向量数据库
        
        Args:
            collection_name: 集合/表名称
        """
        self.collection_name = collection_name
    
    @abstractmethod
    def batch_upsert(self, documents: List[Document]) -> bool:
        """
        批量插入或更新文档
        
        Args:
            documents: 文档列表，每个文档包含 id, content, embedding, metadata
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        相似度检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件，例如 {"category": "tech"}
        
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def delete(self, doc_ids: List[str]) -> bool:
        """
        删除指定文档
        
        Args:
            doc_ids: 文档 ID 列表
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典（包含文档数量等）
        """
        pass
    
    @abstractmethod
    def create_collection(self, dimension: int) -> bool:
        """
        创建集合
        
        Args:
            dimension: 向量维度
        
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def drop_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否成功
        """
        pass
