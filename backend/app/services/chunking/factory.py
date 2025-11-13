"""
Factory for creating chunking strategies.
Makes it easy to swap between different chunking approaches.
"""
from typing import Dict, Any
from app.models.schemas import ChunkingStrategy
from .base import ChunkerBase
from .recursive import RecursiveChunker
from .fixed import FixedSizeChunker
from .email import EmailThreadAwareChunker
from .hierarchical import HierarchicalChunker
from .table_aware import TableAwareChunker


class ChunkerFactory:
    """Factory to create chunking strategies."""
    
    @staticmethod
    def create(
        strategy: ChunkingStrategy,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> ChunkerBase:
        """
        Create a chunker based on the strategy name.
        
        Args:
            strategy: The chunking strategy to use
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            A ChunkerBase instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy == ChunkingStrategy.FIXED:
            return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        elif strategy == ChunkingStrategy.EMAIL_THREAD_AWARE:
            return EmailThreadAwareChunker()
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return HierarchicalChunker(max_chunk_size=chunk_size)
        elif strategy == ChunkingStrategy.TABLE_AWARE:
            return TableAwareChunker()
        elif strategy == ChunkingStrategy.SEMANTIC:
            # TODO: Implement semantic chunking
            raise NotImplementedError("Semantic chunking coming soon!")
        elif strategy == ChunkingStrategy.MARKDOWN:
            # TODO: Implement markdown-aware chunking
            raise NotImplementedError("Markdown chunking coming soon!")
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    @staticmethod
    def list_strategies() -> Dict[str, Any]:
        """List all available chunking strategies with descriptions."""
        return {
            "recursive": {
                "name": "Recursive Character Text Splitter",
                "description": "Splits on natural boundaries (paragraphs → sentences → words)",
                "status": "available",
                "use_case": "General documents"
            },
            "fixed": {
                "name": "Fixed Size Chunker",
                "description": "Simple fixed-length character chunks",
                "status": "available",
                "use_case": "Benchmarking, uniform processing"
            },
            "email_thread_aware": {
                "name": "Email Thread-Aware Chunker",
                "description": "Strips replies & signatures, injects context (subject/sender)",
                "status": "available",
                "use_case": "Supply chain emails, operational communication"
            },
            "hierarchical": {
                "name": "Hierarchical Chunker",
                "description": "Splits on headers, preserves parent context",
                "status": "available",
                "use_case": "SOPs, policy documents, structured docs"
            },
            "table_aware": {
                "name": "Table-Aware Chunker",
                "description": "Extracts PDF tables, converts to markdown, preserves structure",
                "status": "available",
                "use_case": "Invoices, POs, transactional documents"
            },
            "semantic": {
                "name": "Semantic Chunker",
                "description": "Groups text by semantic similarity",
                "status": "coming_soon",
                "use_case": "TBD"
            },
            "markdown": {
                "name": "Markdown-Aware Chunker",
                "description": "Respects markdown structure (headers, code blocks, lists)",
                "status": "coming_soon",
                "use_case": "TBD"
            }
        }

