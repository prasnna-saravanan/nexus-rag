"""
FastAPI routes for the RAG API.
All endpoints for document upload, indexing, and search.
"""
import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.models.schemas import (
    DocumentUploadResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    RAGRequest,
    RAGResponse,
    HealthResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.chunking.factory import ChunkerFactory
from app.services.embedding.factory import EmbedderFactory
from app.services.vector.qdrant_client import QdrantClient
from app.services.rag_service import RAGService

router = APIRouter()


# Initialize services (will be dependency injected)
def get_qdrant_client(settings: Settings = Depends(get_settings)) -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(settings)


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    """Get RAG service instance."""
    return RAGService(settings)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    """
    Health check endpoint.
    Verifies Qdrant and Neo4j connection and collection status.
    """
    from app.services.graph.neo4j_client import Neo4jClient
    
    # Check Qdrant
    is_qdrant_connected = qdrant.health_check()
    collection_info = qdrant.get_collection_info()
    
    # Check Neo4j
    neo4j = Neo4jClient(settings)
    is_neo4j_connected = neo4j.health_check()
    neo4j.close()
    
    return HealthResponse(
        status="healthy" if (is_qdrant_connected and is_neo4j_connected) else "unhealthy",
        qdrant_connected=is_qdrant_connected,
        neo4j_connected=is_neo4j_connected,
        collection_exists=collection_info.get("exists", False),
        num_vectors=collection_info.get("vectors_count")
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    """
    Upload a document for processing.
    Supports: TXT, PDF, Markdown
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in DocumentProcessor.get_supported_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {DocumentProcessor.get_supported_extensions()}"
        )
    
    # Create upload directory if needed
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    
    # Save file with document ID as prefix
    file_path = upload_dir / f"{doc_id}_{file.filename}"
    
    # Write file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        size=len(content),
        message=f"File uploaded successfully. Use document_id '{doc_id}' to index."
    )


@router.post("/index", response_model=IndexResponse)
async def index_document(
    request: IndexRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    """
    Index a document into the vector database.
    
    Steps:
    1. Load document
    2. Chunk using specified strategy
    3. Embed chunks
    4. Store in Qdrant
    """
    # Find the uploaded document
    upload_dir = Path(settings.upload_dir)
    doc_files = list(upload_dir.glob(f"{request.document_id}_*"))
    
    if not doc_files:
        raise HTTPException(
            status_code=404,
            detail=f"Document {request.document_id} not found"
        )
    
    file_path = str(doc_files[0])
    
    # Extract text
    try:
        text = DocumentProcessor.extract_text(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text: {str(e)}"
        )
    
    # Chunk the text
    chunker = ChunkerFactory.create(
        strategy=request.chunking_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    
    chunks = chunker.chunk(
        text=text,
        metadata={
            "document_id": request.document_id,
            "filename": doc_files[0].name
        }
    )
    
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks generated from document"
        )
    
    # Embed chunks
    embedder = EmbedderFactory.create(
        provider=request.embedding_provider,
        settings=settings
    )
    
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = await embedder.embed_batch(chunk_texts)
    
    # Ensure collection exists
    qdrant.ensure_collection(dimension=embedder.get_dimension())
    
    # Index to Qdrant
    chunk_metadata = [chunk.metadata for chunk in chunks]
    num_indexed = await qdrant.index_chunks(
        chunks=chunk_texts,
        embeddings=embeddings,
        document_id=request.document_id,
        metadata=chunk_metadata
    )
    
    return IndexResponse(
        document_id=request.document_id,
        num_chunks=num_indexed,
        collection_name=settings.qdrant_collection_name,
        message=f"Successfully indexed {num_indexed} chunks"
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client)
):
    """
    Search for relevant chunks using semantic similarity.
    """
    # Embed the query
    embedder = EmbedderFactory.create(
        provider="openai",  # Default to OpenAI for now
        settings=settings
    )
    
    query_embedding = await embedder.embed_text(request.query)
    
    # Search Qdrant
    results = await qdrant.search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )
    
    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            text=r["text"],
            score=r["score"],
            metadata=r["metadata"]
        )
        for r in results
    ]
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        total_results=len(search_results)
    )


@router.post("/rag", response_model=RAGResponse)
async def rag_query(
    request: RAGRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    RAG endpoint: Retrieve context + Generate answer.
    
    This is the full RAG pipeline:
    1. Embed query
    2. Search for relevant chunks
    3. Generate answer using LLM with context
    """
    # Embed the query
    embedder = EmbedderFactory.create(
        provider="openai",
        settings=settings
    )
    
    query_embedding = await embedder.embed_text(request.query)
    
    # Search for context
    results = await qdrant.search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        score_threshold=request.score_threshold
    )
    
    # Generate answer using RAG
    answer = await rag_service.generate_answer(
        query=request.query,
        context_chunks=results,
        model=request.model
    )
    
    # Convert results to SearchResult objects
    search_results = [
        SearchResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            text=r["text"],
            score=r["score"],
            metadata=r["metadata"]
        )
        for r in results
    ]
    
    return RAGResponse(
        query=request.query,
        answer=answer,
        sources=search_results,
        model=request.model
    )


@router.get("/strategies")
async def list_strategies():
    """
    List all available chunking strategies and their descriptions.
    Useful for learning and experimentation.
    """
    return {
        "chunking_strategies": ChunkerFactory.list_strategies()
    }

