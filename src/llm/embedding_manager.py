"""
Embedding 管理器
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
from src.core.config import settings


class EmbeddingManager:
    """Embedding 模型管理器"""
    
    def __init__(self):
        print(f"正在加载 Embedding 模型: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        print("✓ Embedding 模型加载完成")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> Union[List[float], List[List[float]]]:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
        
        Returns:
            向量或向量列表
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        # 转换为列表
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        if is_single:
            return embeddings_list[0]
        return embeddings_list
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
