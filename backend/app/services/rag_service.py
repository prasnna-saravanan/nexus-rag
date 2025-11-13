"""
RAG (Retrieval-Augmented Generation) service.
Orchestrates the full RAG pipeline: retrieve context + generate answer.
"""
from typing import List, Dict, Any
import openai
from app.core.config import Settings


class RAGService:
    """
    RAG service for context-aware question answering.
    
    Flow:
    1. Query → Embed
    2. Search vector DB → Retrieve relevant chunks
    3. Build context from chunks
    4. Generate answer using LLM with context
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize RAG service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo"
    ) -> str:
        """
        Generate an answer using LLM with retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved context chunks from vector search
            model: LLM model to use
            
        Returns:
            Generated answer
        """
        # Build context from chunks
        context = self._build_context(context_chunks)
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say so honestly.
Always cite which parts of the context you used."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context provided."""
        
        # Call LLM
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build a formatted context string from chunks.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            doc_id = chunk.get("document_id", "unknown")
            
            context_parts.append(
                f"[Source {idx}] (relevance: {score:.2f}, doc: {doc_id})\n{text}\n"
            )
        
        return "\n---\n".join(context_parts)

