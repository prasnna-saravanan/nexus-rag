"""
Base chunking interface for text splitting strategies.
All chunking strategies implement this protocol.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel


class Chunk(BaseModel):
    """Represents a text chunk with metadata."""
    text: str
    metadata: Dict[str, Any] = {}
    chunk_index: int = 0


class ChunkerBase(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return information about this chunking strategy."""
        pass

