"""
Hierarchical Chunking Strategy for SOPs (Standard Operating Procedures).

Respects document structure:
- Splits on headers (H1, H2, H3)
- Preserves parent section context
- Each chunk knows its place in the hierarchy
"""
import re
from typing import List, Dict, Any, Tuple
from .base import ChunkerBase, Chunk


class HierarchicalChunker(ChunkerBase):
    """
    Hierarchical chunking for structured documents.
    Splits on headers and preserves document hierarchy.
    """
    
    def __init__(self, max_chunk_size: int = 2000):
        """
        Initialize hierarchical chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
        """
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk document hierarchically based on headers.
        
        Args:
            text: Document text (preferably markdown)
            metadata: Optional metadata
        """
        if metadata is None:
            metadata = {}
        
        # Parse document structure
        sections = self._parse_hierarchical_structure(text)
        
        # Create chunks with context injection
        chunks = []
        for idx, section in enumerate(sections):
            level, title, content, parent_titles = section
            
            # Build context-injected chunk
            context_parts = []
            
            # Add parent context (breadcrumb)
            if parent_titles:
                context_parts.append(f"Context: {' > '.join(parent_titles)}")
            
            # Add current section
            if title:
                context_parts.append(f"\n{'#' * level} {title}")
            
            context_parts.append(content)
            
            full_text = '\n'.join(context_parts).strip()
            
            # If chunk is too large, split it further
            if len(full_text) > self.max_chunk_size:
                # Split on paragraphs but maintain context
                sub_chunks = self._split_large_section(
                    full_text, 
                    title or "Untitled Section",
                    parent_titles
                )
                for sub_idx, sub_chunk_text in enumerate(sub_chunks):
                    chunk = Chunk(
                        text=sub_chunk_text,
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "section_level": level,
                            "section_title": title,
                            "parent_sections": parent_titles,
                            "is_subsection": True,
                            "subsection_index": sub_idx,
                            "chunking_strategy": "hierarchical"
                        },
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
            else:
                chunk = Chunk(
                    text=full_text,
                    metadata={
                        **metadata,
                        "chunk_index": idx,
                        "section_level": level,
                        "section_title": title,
                        "parent_sections": parent_titles,
                        "chunking_strategy": "hierarchical"
                    },
                    chunk_index=idx
                )
                chunks.append(chunk)
        
        return chunks
    
    def _parse_hierarchical_structure(self, text: str) -> List[Tuple[int, str, str, List[str]]]:
        """
        Parse document into hierarchical sections.
        
        Returns:
            List of (level, title, content, parent_titles)
        """
        lines = text.split('\n')
        sections = []
        current_section = None
        section_stack = []  # Stack of (level, title)
        
        for line in lines:
            # Check for markdown header
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if header_match:
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Parse new header
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update section stack
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                
                # Parent titles for context
                parent_titles = [t for _, t in section_stack]
                
                # Push current section to stack
                section_stack.append((level, title))
                
                # Start new section
                current_section = (level, title, "", parent_titles)
            else:
                # Add line to current section content
                if current_section:
                    level, title, content, parent_titles = current_section
                    content += line + '\n'
                    current_section = (level, title, content, parent_titles)
                else:
                    # No header yet - treat as intro section
                    if not sections:
                        current_section = (0, None, line + '\n', [])
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _split_large_section(
        self, 
        text: str, 
        section_title: str,
        parent_titles: List[str]
    ) -> List[str]:
        """
        Split a large section into smaller chunks while preserving context.
        """
        # Split on double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = f"Context: {' > '.join(parent_titles + [section_title])}\n\n"
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.max_chunk_size and len(current_chunk) > 100:
                chunks.append(current_chunk.strip())
                current_chunk = f"Context: {' > '.join(parent_titles + [section_title])}\n\n"
            
            current_chunk += para + '\n\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this chunking strategy."""
        return {
            "name": "Hierarchical Chunker",
            "description": "Splits on document structure (headers), preserves hierarchy",
            "features": [
                "Header-based splitting (H1, H2, H3)",
                "Parent context injection",
                "Breadcrumb navigation in metadata",
                "Large section splitting"
            ],
            "best_for": "SOPs, Policy documents, Structured documentation",
            "max_chunk_size": self.max_chunk_size
        }

