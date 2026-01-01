"""
Pydantic 数据模型定义
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class Document(BaseModel):
    """文档数据模型"""
    id: str = Field(..., description="文档唯一标识符")
    content: str = Field(..., description="文档文本内容")
    embedding: Optional[List[float]] = Field(None, description="文档向量表示")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "content": "这是一篇关于人工智能的文章...",
                "metadata": {
                    "category": "tech",
                    "author": "张三",
                    "created_at": "2024-01-01"
                }
            }
        }


class SearchResult(BaseModel):
    """检索结果模型"""
    document: Document = Field(..., description="检索到的文档")
    score: float = Field(..., description="相似度得分")
    rank: Optional[int] = Field(None, description="排名位置")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document": {
                    "id": "doc_001",
                    "content": "这是一篇关于人工智能的文章...",
                    "metadata": {"category": "tech"}
                },
                "score": 0.95,
                "rank": 1
            }
        }


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询文本")
    top_k: int = Field(20, description="返回结果数量")
    filters: Optional[Dict[str, Any]] = Field(None, description="元数据过滤条件")
    enable_hybrid: bool = Field(True, description="是否启用混合检索")
    enable_rerank: bool = Field(True, description="是否启用重排序")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "什么是深度学习？",
                "top_k": 5,
                "filters": {"category": "tech"},
                "enable_hybrid": True,
                "enable_rerank": True
            }
        }


class ChunkStrategy(BaseModel):
    """分块策略配置"""
    chunk_size: int = Field(512, description="子块大小（字符数）")
    chunk_overlap: int = Field(50, description="块之间的重叠")
    enable_parent_child: bool = Field(False, description="是否启用父子分块")
    parent_size: int = Field(2048, description="父块大小（字符数）")
