"""
Pydantic models for request/response schemas.
These define the API contract between frontend and backend.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    RECURSIVE = "recursive"
    FIXED = "fixed"
    SEMANTIC = "semantic"
    MARKDOWN = "markdown"
    EMAIL_THREAD_AWARE = "email_thread_aware"
    HIERARCHICAL = "hierarchical"
    TABLE_AWARE = "table_aware"


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"


# Upload & Indexing


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    document_id: str
    filename: str
    size: int
    message: str


class IndexRequest(BaseModel):
    """Request to index a document."""

    document_id: str
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI


class IndexResponse(BaseModel):
    """Response after indexing."""

    document_id: str
    num_chunks: int
    collection_name: str
    message: str


# Search & RAG


class SearchRequest(BaseModel):
    """Request for RAG search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Response from search."""

    query: str
    results: List[SearchResult]
    total_results: int


class RAGRequest(BaseModel):
    """Request for RAG with LLM generation."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    model: str = "gpt-3.5-turbo"


class RAGResponse(BaseModel):
    """Response from RAG."""

    query: str
    answer: str
    sources: List[SearchResult]
    model: str


# Health & Status


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_connected: bool
    neo4j_connected: bool
    collection_exists: bool
    num_vectors: Optional[int] = None


# Graph RAG


class GraphEntity(BaseModel):
    """Entity node in the knowledge graph."""

    id: str
    type: str  # e.g., "Supplier", "Port", "Product", "Event"
    name: str
    properties: Dict[str, Any] = {}


class GraphRelationship(BaseModel):
    """Relationship between entities."""

    from_entity: str
    to_entity: str
    relationship_type: str
    properties: Dict[str, Any] = {}


class GraphRAGRequest(BaseModel):
    """Request for Graph RAG query."""

    query: str
    max_hops: int = Field(default=3, ge=1, le=5)
    include_vector_context: bool = True


class GraphRAGResponse(BaseModel):
    """Response from Graph RAG."""

    query: str
    answer: str
    graph_paths: List[List[str]]  # List of paths through the graph
    entities_involved: List[GraphEntity]
    relationships: List[GraphRelationship]
    vector_context: Optional[List[SearchResult]] = None


# Document Type Specific


class DocumentType(str, Enum):
    """Document types with specialized processing."""

    MASTER_DATA = "master_data"  # SKUs, Vendors
    SOP = "sop"  # Standard Operating Procedures
    INVOICE = "invoice"  # Invoices & POs
    EMAIL = "email"  # Communication threads


class HybridSearchRequest(BaseModel):
    """Request for hybrid search (vector + keyword)."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata_filters: Optional[Dict[str, Any]] = None


class HyDERequest(BaseModel):
    """Request using Hypothetical Document Embeddings."""

    query: str
    document_type: DocumentType
    top_k: int = Field(default=5, ge=1, le=20)


class RerankedSearchRequest(BaseModel):
    """Request for hybrid search with cross-encoder reranking."""

    query: str
    top_k: int = Field(default=10, ge=1, le=50)
    candidates_multiplier: int = Field(default=3, ge=2, le=10)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    use_reranker: bool = True
    metadata_filters: Optional[Dict[str, Any]] = None
