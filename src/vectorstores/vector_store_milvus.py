"""
Milvus 向量数据库实现
"""
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from src.vectorstores.vector_store_base import RAGVectorStore
from src.core.models import Document, SearchResult
from src.core.config import settings


class MilvusVectorStore(RAGVectorStore):
    """Milvus 向量数据库实现"""
    
    def __init__(self, collection_name: str = "rag_collection"):
        super().__init__(collection_name)
        self._connect()
        self.collection: Optional[Collection] = None
    
    def _connect(self):
        """连接到 Milvus 服务"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            print(f"✓ 成功连接到 Milvus: {settings.milvus_host}:{settings.milvus_port}")
        except Exception as e:
            print(f"✗ Milvus 连接失败: {e}")
            raise
    
    def create_collection(self, dimension: int) -> bool:
        """创建 Milvus 集合"""
        try:
            # 如果集合已存在，先删除
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"已删除旧集合: {self.collection_name}")
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=2000),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="RAG 文档集合"
            )
            
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # 创建索引（IVF_FLAT 索引）
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            print(f"✓ 成功创建 Milvus 集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 创建集合失败: {e}")
            return False
    
    def batch_upsert(self, documents: List[Document]) -> bool:
        """批量插入文档"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            import json
            
            # 准备数据
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            categories = [doc.metadata.get("category", "default") for doc in documents]
            metadata_jsons = [json.dumps(doc.metadata, ensure_ascii=False) for doc in documents]
            
            entities = [
                ids,
                contents,
                embeddings,
                categories,
                metadata_jsons
            ]
            
            # 插入数据
            self.collection.insert(entities)
            self.collection.flush()
            
            # 加载集合到内存
            self.collection.load()
            
            print(f"✓ 成功插入 {len(documents)} 条文档到 Milvus")
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
            if not self.collection:
                self.collection = Collection(self.collection_name)
                self.collection.load()
            
            # 构建过滤表达式
            expr = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key == "category":
                        conditions.append(f'category == "{value}"')
                if conditions:
                    expr = " && ".join(conditions)
            
            # 执行搜索
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "content", "category", "metadata_json"]
            )
            
            # 转换结果
            import json
            search_results = []
            for hits in results:
                for hit in hits:
                    metadata = json.loads(hit.entity.get("metadata_json", "{}"))
                    doc = Document(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        metadata=metadata
                    )
                    search_results.append(
                        SearchResult(
                            document=doc,
                            score=float(hit.distance)
                        )
                    )
            
            return search_results
        except Exception as e:
            print(f"✗ 检索失败: {e}")
            return []
    
    def delete(self, doc_ids: List[str]) -> bool:
        """删除文档"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            expr = f'id in {doc_ids}'
            self.collection.delete(expr)
            print(f"✓ 成功删除 {len(doc_ids)} 条文档")
            return True
        except Exception as e:
            print(f"✗ 删除失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            return {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "description": self.collection.description
            }
        except Exception as e:
            print(f"✗ 获取统计信息失败: {e}")
            return {}
    
    def drop_collection(self) -> bool:
        """删除集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                print(f"✓ 成功删除集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 删除集合失败: {e}")
            return False
