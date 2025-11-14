"""
Table-Aware Chunking Strategy for Invoices & POs.

Key Features:
- Extract tables from PDFs using pdfplumber
- Convert tables to markdown format (preserves structure)
- Keep entire tables as single chunks (don't split rows)
- Extract structured metadata (invoice ID, total, vendor)
"""
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import pdfplumber
from tabulate import tabulate
from .base import ChunkerBase, Chunk


class TableAwareChunker(ChunkerBase):
    """
    Table-aware chunking for invoices and POs.
    Extracts tables from PDFs and preserves structure.
    """
    
    def __init__(self):
        """Initialize table-aware chunker."""
        pass
    
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk document while preserving table structure.
        
        Args:
            text: Can be raw text OR a file path to PDF
            metadata: Must include document_type, filename
        """
        if metadata is None:
            metadata = {}
        
        filename = metadata.get("filename", "")
        
        # Check if this is a PDF path
        if filename.lower().endswith('.pdf') and Path(text).exists():
            return self._chunk_pdf(text, metadata)
        else:
            return self._chunk_text(text, metadata)
    
    def _chunk_pdf(self, pdf_path: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Extract and chunk PDF with table awareness.
        """
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extract document metadata
            doc_metadata = self._extract_invoice_metadata(pdf)
            metadata.update(doc_metadata)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                
                # Extract tables
                tables = page.extract_tables()
                
                if tables:
                    # Process each table
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:
                            # Convert table to markdown
                            table_md = self._table_to_markdown(table)
                            
                            # Create table chunk
                            chunk = Chunk(
                                text=table_md,
                                metadata={
                                    **metadata,
                                    "chunk_index": len(chunks),
                                    "page_number": page_num,
                                    "table_index": table_idx,
                                    "content_type": "table",
                                    "chunking_strategy": "table_aware"
                                },
                                chunk_index=len(chunks)
                            )
                            chunks.append(chunk)
                            
                            # Remove table text from page text (approximate)
                            # This prevents duplication
                            for row in table:
                                for cell in row:
                                    if cell:
                                        text = text.replace(str(cell), "", 1)
                
                # Create text chunk for remaining content
                text = text.strip()
                if text:
                    chunk = Chunk(
                        text=text,
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "page_number": page_num,
                            "content_type": "text",
                            "chunking_strategy": "table_aware"
                        },
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk plain text with table detection.
        """
        # Detect markdown tables
        table_pattern = r'\|(.+)\|[\r\n]+\|[\s:|-]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)'
        
        chunks = []
        last_end = 0
        
        for match in re.finditer(table_pattern, text):
            # Text before table
            before_text = text[last_end:match.start()].strip()
            if before_text:
                chunk = Chunk(
                    text=before_text,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "content_type": "text",
                        "chunking_strategy": "table_aware"
                    },
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
            
            # Table chunk
            table_text = match.group(0)
            chunk = Chunk(
                text=table_text,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "content_type": "table",
                    "chunking_strategy": "table_aware"
                },
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
            
            last_end = match.end()
        
        # Remaining text
        remaining = text[last_end:].strip()
        if remaining:
            chunk = Chunk(
                text=remaining,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "content_type": "text",
                    "chunking_strategy": "table_aware"
                },
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        return chunks if chunks else [Chunk(
            text=text,
            metadata={**metadata, "chunk_index": 0, "chunking_strategy": "table_aware"},
            chunk_index=0
        )]
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table to markdown format.
        
        Args:
            table: 2D list of table cells
            
        Returns:
            Markdown-formatted table string
        """
        if not table or len(table) < 2:
            return ""
        
        # Clean cells
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        # Use tabulate for clean markdown
        headers = cleaned_table[0]
        rows = cleaned_table[1:]
        
        return tabulate(rows, headers=headers, tablefmt="github")
    
    def _extract_invoice_metadata(self, pdf: pdfplumber.PDF) -> Dict[str, Any]:
        """
        Extract invoice-specific metadata from PDF.
        
        Looks for common patterns:
        - Invoice ID / Number
        - Total Amount
        - Vendor Name
        - Date
        """
        metadata = {}
        
        # Get first page text for metadata extraction
        if pdf.pages:
            text = pdf.pages[0].extract_text() or ""
            
            # Invoice Number patterns
            invoice_patterns = [
                r'Invoice\s*#?\s*:?\s*(\w+[-\w]*)',
                r'Invoice\s+Number\s*:?\s*(\w+[-\w]*)',
                r'PO\s*#?\s*:?\s*(\w+[-\w]*)'
            ]
            for pattern in invoice_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metadata['invoice_number'] = match.group(1)
                    break
            
            # Total Amount patterns
            total_patterns = [
                r'Total\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'Amount\s+Due\s*:?\s*\$?\s*([\d,]+\.?\d*)',
                r'Grand\s+Total\s*:?\s*\$?\s*([\d,]+\.?\d*)'
            ]
            for pattern in total_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        metadata['total_amount'] = float(amount_str)
                    except:
                        metadata['total_amount_raw'] = match.group(1)
                    break
            
            # Date patterns
            date_pattern = r'Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            match = re.search(date_pattern, text, re.IGNORECASE)
            if match:
                metadata['invoice_date'] = match.group(1)
        
        return metadata
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about this chunking strategy."""
        return {
            "name": "Table-Aware Chunker",
            "description": "Extracts tables from PDFs, converts to markdown, preserves structure",
            "features": [
                "PDF table extraction via pdfplumber",
                "Table-to-markdown conversion",
                "Whole-table chunking (no row splits)",
                "Invoice metadata extraction"
            ],
            "best_for": "Invoices, Purchase Orders, Transactional documents",
            "supported_formats": ["PDF", "Markdown tables"]
        }

