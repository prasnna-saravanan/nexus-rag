"""
Document processing utilities.
Handles file reading and text extraction from various formats.
"""
import os
from typing import Optional
from pathlib import Path
import pypdf
import markdown


class DocumentProcessor:
    """
    Process various document formats and extract text.
    
    Supported formats:
    - Plain text (.txt)
    - PDF (.pdf)
    - Markdown (.md)
    """
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == ".txt":
            return DocumentProcessor._read_text(file_path)
        elif extension == ".pdf":
            return DocumentProcessor._read_pdf(file_path)
        elif extension in [".md", ".markdown"]:
            return DocumentProcessor._read_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    @staticmethod
    def _read_text(file_path: str) -> str:
        """Read plain text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def _read_pdf(file_path: str) -> str:
        """Extract text from PDF using pypdf."""
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
    
    @staticmethod
    def _read_markdown(file_path: str) -> str:
        """
        Read markdown file.
        For now, just return raw markdown.
        Could convert to HTML or plain text if needed.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def get_supported_extensions() -> list[str]:
        """Return list of supported file extensions."""
        return [".txt", ".pdf", ".md", ".markdown"]

