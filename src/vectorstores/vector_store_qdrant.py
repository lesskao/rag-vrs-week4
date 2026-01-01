"""
Qdrant 向量数据库实现
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.vectorstores.vector_store_base import RAGVectorStore
from src.core.models import Document, SearchResult
from src.core.config import settings


class QdrantVectorStore(RAGVectorStore):
    """Qdrant 向量数据库实现"""
    
    def __init__(self, collection_name: str = "rag_collection"):
        super().__init__(collection_name)
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        print(f"✓ 成功连接到 Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    
    def create_collection(self, dimension: int) -> bool:
        """创建 Qdrant 集合"""
        try:
            # 如果集合已存在，先删除
            collections = self.client.get_collections().collections
            if any(col.name == self.collection_name for col in collections):
                self.client.delete_collection(self.collection_name)
                print(f"已删除旧集合: {self.collection_name}")
            
            # 创建新集合
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )
            
            print(f"✓ 成功创建 Qdrant 集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 创建集合失败: {e}")
            return False
    
    def batch_upsert(self, documents: List[Document]) -> bool:
        """批量插入文档"""
        try:
            points = []
            for doc in documents:
                point = PointStruct(
                    id=hash(doc.id) % (10 ** 10),  # 转换为整数 ID
                    vector=doc.embedding,
                    payload={
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                )
                points.append(point)
            
            # 批量插入
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✓ 成功插入 {len(documents)} 条文档到 Qdrant")
            return True
        except Exception as e:
            print(f"✗ 批量插入失败: {e}")
            return False
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """相似度检索"""
        try:
            # 构建过滤条件
            query_filter = None
            if filters:
                must_conditions = []
                for key, value in filters.items():
                    must_conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                if must_conditions:
                    query_filter = Filter(must=must_conditions)
            
            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter
            )
            
            # 转换结果
            search_results = []
            for result in results:
                doc = Document(
                    id=result.payload["id"],
                    content=result.payload["content"],
                    metadata=result.payload.get("metadata", {})
                )
                search_results.append(
                    SearchResult(
                        document=doc,
                        score=float(result.score)
                    )
                )
            
            return search_results
        except Exception as e:
            print(f"✗ 检索失败: {e}")
            return []
    
    def delete(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        try:
            # 将字符串 ID 转换为整数 ID
            point_ids = [hash(doc_id) % (10 ** 10) for doc_id in doc_ids]
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            print(f"✓ 成功删除 {len(doc_ids)} 条文档")
            return True
        except Exception as e:
            print(f"✗ 删除失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "num_entities": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            print(f"✗ 获取统计信息失败: {e}")
            return {}
    
    def drop_collection(self) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ 成功删除集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 删除集合失败: {e}")
            return False
