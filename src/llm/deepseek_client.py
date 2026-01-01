"""
DeepSeek API 客户端
"""
from openai import OpenAI
from typing import List, Optional
from src.core.config import settings


class DeepSeekClient:
    """DeepSeek API 客户端封装"""
    
    def __init__(self):
        if not settings.deepseek_api_key:
            print("⚠️  警告: 未配置 DeepSeek API Key，高级功能将不可用")
            print("   配置方法: 在 .env 文件中设置 DEEPSEEK_API_KEY")
            self.client = None
            self.model = None
        else:
            self.client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url
            )
            self.model = settings.deepseek_model
    
    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        调用 DeepSeek Chat API
        
        Args:
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大生成长度
        
        Returns:
            生成的文本
        """
        if not self.client:
            print("✗ DeepSeek API 未配置")
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"✗ DeepSeek API 调用失败: {e}")
            return ""
    
    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Multi-Query: 将一个查询改写为多个同义查询
        
        Args:
            query: 原始查询
            num_queries: 生成的查询数量
        
        Returns:
            改写后的查询列表
        """
        prompt = f"""你是一个专业的查询优化助手。请将以下用户查询改写为 {num_queries} 个语义相近但表达不同的查询，用于提高检索召回率。

原始查询: {query}

要求：
1. 保持核心语义不变
2. 使用不同的表达方式和同义词
3. 每行一个查询
4. 不要添加序号或其他标记

改写后的查询："""

        messages = [
            {"role": "system", "content": "你是一个专业的查询优化助手。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages, temperature=0.8, max_tokens=300)
        
        # 解析返回的查询列表
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        # 包含原始查询
        all_queries = [query] + queries[:num_queries-1]
        return all_queries[:num_queries]
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE: 生成假设性文档（伪答案）
        
        Args:
            query: 用户查询
        
        Returns:
            生成的假设性文档
        """
        prompt = f"""请针对以下问题，生成一个详细、专业的回答。不要说"我不知道"，请直接生成一个可能的答案。

问题: {query}

回答:"""

        messages = [
            {"role": "system", "content": "你是一个知识丰富的AI助手，擅长生成高质量的答案。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages, temperature=0.7, max_tokens=500)
        return response
    
    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        使用 DeepSeek 对文档进行重排序
        
        Args:
            query: 用户查询
            documents: 文档列表
            top_k: 返回的文档数量
        
        Returns:
            (文档索引, 相关性分数) 列表
        """
        if not documents:
            return []
        
        # 构建打分提示
        docs_text = "\n\n".join([f"文档{i+1}:\n{doc[:500]}" for i, doc in enumerate(documents)])
        
        prompt = f"""请对以下文档与查询的相关性进行评分（0-10分，10分最相关）。

查询: {query}

{docs_text}

请按照以下格式输出，每行一个文档的评分：
1: 分数
2: 分数
3: 分数
...

评分结果："""

        messages = [
            {"role": "system", "content": "你是一个专业的信息检索评估专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages, temperature=0.3, max_tokens=200)
        
        # 解析评分结果
        scores = []
        for line in response.split('\n'):
            if ':' in line:
                try:
                    idx_str, score_str = line.split(':')
                    idx = int(idx_str.strip()) - 1
                    score = float(score_str.strip())
                    if 0 <= idx < len(documents):
                        scores.append((idx, score))
                except:
                    continue
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def answer_with_context(self, query: str, contexts: List[str]) -> str:
        """
        基于检索上下文回答问题
        
        Args:
            query: 用户查询
            contexts: 检索到的文档列表
        
        Returns:
            生成的答案
        """
        context_text = "\n\n".join([f"[文档{i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""请基于以下检索到的文档，回答用户的问题。如果文档中没有相关信息，请诚实地说明。

检索到的文档：
{context_text}

用户问题: {query}

回答:"""

        messages = [
            {"role": "system", "content": "你是一个专业的AI助手，擅长基于给定的文档回答问题。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat(messages, temperature=0.7, max_tokens=1000)
        return response
