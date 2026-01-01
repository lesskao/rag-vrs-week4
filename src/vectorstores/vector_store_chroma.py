"""
Chroma 向量数据库实现
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.vectorstores.vector_store_base import RAGVectorStore
from src.core.models import Document, SearchResult
from src.core.config import settings as app_settings


class ChromaVectorStore(RAGVectorStore):
    """Chroma 向量数据库实现"""
    
    def __init__(self, collection_name: str = "rag_collection"):
        super().__init__(collection_name)
        self.client = chromadb.PersistentClient(
            path=app_settings.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        print(f"✓ 成功初始化 Chroma: {app_settings.chroma_persist_directory}")
    
    def create_collection(self, dimension: int) -> bool:
        """创建 Chroma 集合"""
        try:
            # 如果集合已存在，先删除
            try:
                self.client.delete_collection(self.collection_name)
                print(f"已删除旧集合: {self.collection_name}")
            except:
                pass
            
            # 创建新集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"✓ 成功创建 Chroma 集合: {self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 创建集合失败: {e}")
            return False
    
    def batch_upsert(self, documents: List[Document]) -> bool:
        """批量插入文档"""
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)
            
            # 准备数据
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = []
            documents_text = []
            
            for doc in documents:
                # Chroma 的 metadata 只支持基本类型
                metadata = {
                    "category": doc.metadata.get("category", "default"),
                }
                # 将其他元数据转为字符串存储
                for key, value in doc.metadata.items():
                    if key != "category":
                        metadata[key] = str(value)
                
                metadatas.append(metadata)
                documents_text.append(doc.content)
            
            # 批量插入
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            print(f"✓ 成功插入 {len(documents)} 条文档到 Chroma")
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
                self.collection = self.client.get_collection(self.collection_name)
            
            # 构建过滤条件
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    where[key] = value
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # 转换结果
            search_results = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i]
                    )
                    # Chroma 返回的是距离，需要转换为相似度（距离越小相似度越高）
                    distance = results["distances"][0][i]
                    score = 1.0 / (1.0 + distance)  # 转换为相似度分数
                    
                    search_results.append(
                        SearchResult(
                            document=doc,
                            score=float(score)
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
                self.collection = self.client.get_collection(self.collection_name)
            
            self.collection.delete(ids=doc_ids)
            print(f"✓ 成功删除 {len(doc_ids)} 条文档")
            return True
        except Exception as e:
            print(f"✗ 删除失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                self.collection = self.client.get_collection(self.collection_name)
            
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "num_entities": count
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
