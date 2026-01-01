"""
BM25 关键词检索器
"""
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import re


class BM25Retriever:
    """BM25 关键词检索器"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def index_documents(self, documents: List[dict]):
        """
        索引文档
        
        Args:
            documents: 文档列表，每个文档包含 id 和 content
        """
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # 简单的中文分词（按字符分）
        tokenized_corpus = [self._tokenize(doc['content']) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✓ BM25 索引完成，共 {len(documents)} 条文档")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单分词（支持中英文）
        
        Args:
            text: 待分词文本
        
        Returns:
            词列表
        """
        # 移除标点符号
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 分离中英文
        tokens = []
        words = text.split()
        for word in words:
            # 如果是英文单词，直接添加
            if re.match(r'^[a-zA-Z]+$', word):
                tokens.append(word.lower())
            else:
                # 中文按字符分
                tokens.extend(list(word))
        
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            (文档ID, BM25分数) 列表
        """
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k 结果
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return results
