"""
配置管理模块
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import Field


class Settings(BaseSettings):
    """系统配置"""
    
    # DeepSeek API 配置（可选，仅高级功能需要）
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    
    # 向量数据库配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    chroma_persist_directory: str = "./chroma_db"
    
    # Embedding 配置
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dimension: int = 768
    
    # 检索配置
    top_k: int = 20
    final_top_k: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()
