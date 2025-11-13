"""
Recursive character text splitter - the industry standard.
Tries to split on paragraphs, then sentences, then words, then characters.
"""
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import ChunkerBase, Chunk


class RecursiveChunker(ChunkerBase):
    """
    Recursive character-based chunking strategy.
    
    This is the most common approach and works well for most documents.
    It tries to keep semantically related text together by splitting on
    natural boundaries (paragraphs, sentences) before resorting to character splits.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the recursive chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Separators in order of preference
        # LangChain will try these in order: paragraph, newline, sentence, word, character
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text using recursive character splitting."""
        if metadata is None:
            metadata = {}
        
        # Use LangChain's splitter
        text_chunks = self.splitter.split_text(text)
        
        # Convert to Chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                    "chunking_strategy": "recursive"
                },
                chunk_index=idx
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this chunking strategy."""
        return {
            "name": "Recursive Character Text Splitter",
            "description": "Splits on paragraphs, then sentences, then words, then characters",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "best_for": "General purpose text, articles, documentation"
        }

