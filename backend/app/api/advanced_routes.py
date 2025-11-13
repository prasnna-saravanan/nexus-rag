"""
Advanced RAG API routes.

Specialized endpoints for enterprise RAG:
- Graph RAG (supply chain risk analysis)
- Hybrid Search (dense + sparse)
- HyDE (hypothetical document embeddings)
- Document-type-specific operations
"""

from typing import Optional

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.schemas import (
    GraphEntity,
    GraphRAGRequest,
    GraphRAGResponse,
    GraphRelationship,
    HybridSearchRequest,
    HyDERequest,
    SearchResponse,
    SearchResult,
)
from app.services.embedding.factory import EmbedderFactory
from app.services.graph.graph_rag_service import GraphRAGService
from app.services.graph.neo4j_client import Neo4jClient
from app.services.hybrid_search import HybridSearchService
from app.services.hyde_service import HyDEService
from app.services.vector.qdrant_client import QdrantClient

router = APIRouter()


# Dependency injection
def get_neo4j_client(settings: Settings = Depends(get_settings)) -> Neo4jClient:
    """Get Neo4j client instance."""
    return Neo4jClient(settings)


def get_qdrant_client(settings: Settings = Depends(get_settings)) -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(settings)


def get_graph_rag_service(
    settings: Settings = Depends(get_settings),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
) -> GraphRAGService:
    """Get Graph RAG service instance."""
    return GraphRAGService(settings, neo4j)


def get_hyde_service(settings: Settings = Depends(get_settings)) -> HyDEService:
    """Get HyDE service instance."""
    return HyDEService(settings)


# Graph RAG Endpoints


@router.post("/graph/entity", tags=["Graph RAG"])
async def create_graph_entity(
    entity_id: str,
    entity_type: str,
    name: str,
    properties: Optional[dict] = None,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Create an entity in the knowledge graph.

    Entity Types:
    - Supplier
    - Product
    - Port
    - Event (e.g., Strike, Natural Disaster)
    - PurchaseOrder
    - LogisticsProvider
    """
    success = neo4j.create_entity(entity_id, entity_type, name, properties)
    return {"success": success, "entity_id": entity_id}


@router.post("/graph/relationship", tags=["Graph RAG"])
async def create_graph_relationship(
    from_id: str,
    to_id: str,
    relationship_type: str,
    properties: Optional[dict] = None,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Create a relationship in the knowledge graph.

    Relationship Types:
    - SUPPLIES (Supplier -> Product)
    - SHIPS_VIA (Product -> Port)
    - DEPENDS_ON (PO -> Supplier)
    - AFFECTS (Event -> Entity)
    - USES (Supplier -> LogisticsProvider)
    """
    success = neo4j.create_relationship(from_id, to_id, relationship_type, properties)
    return {"success": success}


@router.post("/graph/rag", response_model=GraphRAGResponse, tags=["Graph RAG"])
async def graph_rag_query(
    request: GraphRAGRequest,
    settings: Settings = Depends(get_settings),
    graph_rag: GraphRAGService = Depends(get_graph_rag_service),
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    """
    Graph RAG query for supply chain risk analysis.

    Example: "How does the strike in Germany affect us?"

    This will:
    1. Find "strike" and "Germany" entities in the graph
    2. Traverse relationships to find affected purchase orders
    3. Optionally retrieve vector context
    4. Generate comprehensive answer
    """
    # Optional: Get vector context
    vector_context = None
    if request.include_vector_context:
        embedder = EmbedderFactory.create("openai", settings)
        query_embedding = await embedder.embed_text(request.query)
        vector_results = await qdrant.search(query_embedding=query_embedding, top_k=5)
        vector_context = vector_results

    # Execute Graph RAG
    result = await graph_rag.query_graph_rag(
        query=request.query, max_hops=request.max_hops, vector_context=vector_context
    )

    # Convert to response model
    return GraphRAGResponse(
        query=request.query,
        answer=result["answer"],
        graph_paths=result["graph_paths"],
        entities_involved=[GraphEntity(**e) for e in result["entities_involved"]],
        relationships=[GraphRelationship(**r) for r in result["relationships"]],
        vector_context=[SearchResult(**v) for v in vector_context]
        if vector_context
        else None,
    )


# Hybrid Search Endpoint


@router.post("/search/hybrid", response_model=SearchResponse, tags=["Advanced Search"])
async def hybrid_search(
    request: HybridSearchRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    """
    Hybrid Search: Dense (Vector) + Sparse (BM25) retrieval.

    Use Cases:
    - Master Data: Exact SKU match + semantic product search
    - Invoices: Exact invoice number + fuzzy vendor name
    - Emails: Exact sender + semantic content match

    Parameters:
    - keyword_weight: 0.3 = 30% BM25, 70% vector (default)
    - Increase keyword_weight for exact matches
    - Increase vector_weight for semantic similarity
    """
    # Embed query
    embedder = EmbedderFactory.create("openai", settings)
    query_embedding = await embedder.embed_text(request.query)

    # Initialize hybrid search
    hybrid_service = HybridSearchService(qdrant)

    # TODO: Load BM25 index (in production, pre-index all documents)
    # For now, just use vector search
    # In production: hybrid_service.index_for_bm25(documents, metadata)

    # Perform hybrid search
    results = await hybrid_service.hybrid_search(
        query=request.query,
        query_embedding=query_embedding,
        top_k=request.top_k,
        keyword_weight=request.keyword_weight,
        vector_weight=request.vector_weight,
        metadata_filters=request.metadata_filters,
    )

    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id", ""),
            text=r["text"],
            score=r["hybrid_score"],
            metadata={
                **r.get("metadata", {}),
                "vector_score": r["vector_score"],
                "bm25_score": r["bm25_score"],
            },
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query, results=search_results, total_results=len(search_results)
    )


# HyDE Endpoint


@router.post("/search/hyde", response_model=SearchResponse, tags=["Advanced Search"])
async def hyde_search(
    request: HyDERequest,
    settings: Settings = Depends(get_settings),
    hyde_service: HyDEService = Depends(get_hyde_service),
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    """
    HyDE (Hypothetical Document Embeddings) Search.

    Perfect for SOPs and formal documents where user questions
    don't match document vocabulary.

    Example:
    - Query: "What do I do if supplier fails audit?"
    - HyDE generates: "The procedure for supplier audit failure is..."
    - Searches with hypothetical document embedding
    - Returns actual SOPs that match
    """
    # Generate hypothetical document
    hypothetical_doc = await hyde_service.generate_hypothetical_document(
        query=request.query, document_type=request.document_type
    )

    # Embed hypothetical document
    embedder = EmbedderFactory.create("openai", settings)
    hyde_embedding = await embedder.embed_text(hypothetical_doc)

    # Search with HyDE embedding
    results = await qdrant.search(query_embedding=hyde_embedding, top_k=request.top_k)

    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            text=r["text"],
            score=r["score"],
            metadata={
                **r["metadata"],
                "hyde_query": hypothetical_doc[:200] + "...",  # Include HyDE query
            },
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query, results=search_results, total_results=len(search_results)
    )


# Document Type Specific Endpoints


@router.get("/document-types", tags=["Document Types"])
async def list_document_types():
    """
    List all supported document types with their processing strategies.
    """
    return {
        "master_data": {
            "name": "Master Data (SKUs, Vendors)",
            "chunking_strategy": "Small-to-Big",
            "retrieval_strategy": "Hybrid Search (Keyword + Vector)",
            "best_practices": [
                "Use strict metadata filtering",
                "Pre-filter by VendorID before search",
                "Chunk: Store granular fields as vectors",
            ],
        },
        "sop": {
            "name": "Standard Operating Procedures",
            "chunking_strategy": "Hierarchical (Header-based)",
            "retrieval_strategy": "HyDE (Hypothetical Document Embeddings)",
            "best_practices": [
                "Split by headers (H1, H2)",
                "Inject parent section context",
                "Use HyDE to bridge question-policy gap",
            ],
        },
        "invoice": {
            "name": "Invoices & Purchase Orders",
            "chunking_strategy": "Table-Aware (Layout preservation)",
            "retrieval_strategy": "Metadata Filtering + Vector",
            "best_practices": [
                "Extract tables as single chunks",
                "Convert to markdown",
                "Filter by Date/Amount before vector search",
            ],
        },
        "email": {
            "name": "Emails & Communication",
            "chunking_strategy": "Thread-Aware (Clean & Link)",
            "retrieval_strategy": "Recency-weighted + Entity Filtering",
            "best_practices": [
                "Strip quoted replies",
                "Remove signatures",
                "Inject Subject + Sender context",
                "Apply recency decay",
            ],
        },
    }
