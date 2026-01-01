"""
文档分块策略
"""
from typing import List, Dict, Tuple
from src.core.models import Document


class ChunkingStrategy:
    """文档分块策略"""
    
    @staticmethod
    def simple_chunk(
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        简单分块策略
        
        Args:
            text: 待分块文本
            chunk_size: 块大小（字符数）
            chunk_overlap: 块之间的重叠
        
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
        
        return chunks
    
    @staticmethod
    def parent_child_chunk(
        text: str,
        child_size: int = 512,
        child_overlap: int = 50,
        parent_size: int = 2048
    ) -> List[Tuple[str, str]]:
        """
        父子分块策略
        索引小块，但检索时返回大块以提供更多上下文
        
        Args:
            text: 待分块文本
            child_size: 子块大小
            child_overlap: 子块重叠
            parent_size: 父块大小
        
        Returns:
            [(子块, 父块), ...] 列表
        """
        # 先生成父块
        parent_chunks = ChunkingStrategy.simple_chunk(
            text,
            chunk_size=parent_size,
            chunk_overlap=child_size
        )
        
        # 为每个父块生成子块
        result = []
        for parent in parent_chunks:
            child_chunks = ChunkingStrategy.simple_chunk(
                parent,
                chunk_size=child_size,
                chunk_overlap=child_overlap
            )
            for child in child_chunks:
                result.append((child, parent))
        
        return result
    
    @staticmethod
    def create_parent_child_documents(
        doc_id: str,
        text: str,
        metadata: Dict,
        child_size: int = 512,
        parent_size: int = 2048
    ) -> Tuple[List[Document], Dict[str, str]]:
        """
        创建父子分块文档
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            metadata: 元数据
            child_size: 子块大小
            parent_size: 父块大小
        
        Returns:
            (子文档列表, 子ID到父文本的映射)
        """
        chunks = ChunkingStrategy.parent_child_chunk(
            text,
            child_size=child_size,
            parent_size=parent_size
        )
        
        child_documents = []
        child_to_parent = {}
        
        for i, (child, parent) in enumerate(chunks):
            # 为子块生成唯一ID
            child_id = f"{doc_id}_child_{i}"
            
            # 创建子文档
            child_doc = Document(
                id=child_id,
                content=child,
                metadata={
                    **metadata,
                    "parent_doc_id": doc_id,
                    "chunk_index": i,
                    "is_child": True
                }
            )
            child_documents.append(child_doc)
            
            # 记录父块内容
            child_to_parent[child_id] = parent
        
        return child_documents, child_to_parent
