"""
核心模块 - 配置和数据模型
"""
from .config import settings, Settings
from .models import Document, SearchResult, QueryRequest, ChunkStrategy

__all__ = [
    "settings",
    "Settings",
    "Document",
    "SearchResult",
    "QueryRequest",
    "ChunkStrategy",
]
