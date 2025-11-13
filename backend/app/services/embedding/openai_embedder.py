"""
OpenAI embedding provider.
Uses OpenAI's embedding models (text-embedding-3-small, text-embedding-3-large, etc.)
"""
from typing import List, Dict, Any
import openai
from .base import EmbedderBase


class OpenAIEmbedder(EmbedderBase):
    """
    OpenAI embedding provider.
    
    Supports:
    - text-embedding-3-small (1536 dimensions) - Fast, cheap, good quality
    - text-embedding-3-large (3072 dimensions) - Highest quality
    - text-embedding-ada-002 (1536 dimensions) - Legacy model
    """
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text using OpenAI API."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a batch."""
        # OpenAI supports batch embeddings
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        # Extract embeddings in order
        return [item.embedding for item in response.data]
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimensions.get(self.model, 1536)
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this embedding provider."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "dimension": self.get_dimension(),
            "max_tokens": 8191,
            "description": "OpenAI's state-of-the-art embedding models"
        }

