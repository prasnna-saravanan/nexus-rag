"""
Fixed-size chunking - simplest strategy.
Splits text into fixed-size chunks with optional overlap.
"""
from typing import List, Dict, Any
from .base import ChunkerBase, Chunk


class FixedSizeChunker(ChunkerBase):
    """
    Fixed-size chunking strategy.
    
    The simplest approach - just split into fixed character lengths.
    Fast but may split in the middle of sentences or words.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the fixed-size chunker.
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text into fixed-size chunks."""
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        idx = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Create Chunk object
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                    "chunking_strategy": "fixed",
                    "start_pos": start,
                    "end_pos": end
                },
                chunk_index=idx
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            idx += 1
            
            # Prevent infinite loop if overlap >= chunk_size
            if self.chunk_overlap >= self.chunk_size:
                break
        
        return chunks
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this chunking strategy."""
        return {
            "name": "Fixed Size Chunker",
            "description": "Splits text into fixed character-length chunks",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "best_for": "Uniform processing, simple documents, benchmarking"
        }

