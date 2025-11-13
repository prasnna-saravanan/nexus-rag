"""
Base embedding interface.
All embedding providers implement this protocol.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class EmbedderBase(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in a batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return information about this embedding provider."""
        pass

