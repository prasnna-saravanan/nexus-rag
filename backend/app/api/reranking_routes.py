"""
Reranking API Routes.

The "Kill Shot" endpoint: Hybrid Search + Cross-Encoder Reranking.
"""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.models.schemas import RerankedSearchRequest, SearchResponse, SearchResult
from app.services.embedding.factory import EmbedderFactory
from app.services.hybrid_search import HybridSearchService
from app.services.reranker import CrossEncoderReranker
from app.services.vector.qdrant_client import QdrantClient

router = APIRouter()


def get_qdrant_client(settings: Settings = Depends(get_settings)) -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(settings)


@router.post("/search/reranked", response_model=SearchResponse, tags=["Reranking"])
async def reranked_search(
    request: RerankedSearchRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client),
):
    """
    The "Kill Shot": Hybrid Search + Cross-Encoder Reranking.

    Pipeline:
    1. Hybrid Search (BM25 + Vector) → Get top 30-50 candidates
    2. Cross-Encoder Reranking → Score each (query, doc) pair precisely
    3. Return top K with highest cross-encoder scores

    Use Case: Master Data (SKUs, Products, Vendors)
    - BM25 catches exact SKU matches ("XJ-900")
    - Vector catches semantic descriptions ("steel rod")
    - Cross-encoder picks the best overall match

    Parameters:
    - candidates_multiplier: How many candidates to retrieve before reranking
      (default 3x = if top_k=10, retrieve 30 candidates, then rerank to 10)
    - use_reranker: Set to False to skip reranking (faster but less accurate)

    Example:
    ```
    Query: "SKU XJ-900 steel rod"

    Step 1 - Hybrid Search (top 30):
      1. "Steel Rod XJ-900" (hybrid: 0.95)
      2. "Iron Pipe XJ-901" (hybrid: 0.89) ← Semantically similar but wrong
      3. "Steel Rod Premium" (hybrid: 0.85)
      ...

    Step 2 - Cross-Encoder Reranking (top 10):
      1. "Steel Rod XJ-900" (rerank: 0.98) ✅ CORRECT - Exact match
      2. "Steel Rod Premium" (rerank: 0.72)
      3. "Iron Pipe XJ-901" (rerank: 0.45) ← Demoted by reranker
    ```
    """
    # Calculate number of candidates to retrieve
    num_candidates = request.top_k * request.candidates_multiplier

    # Step 1: Embed query
    embedder = EmbedderFactory.create("openai", settings)
    query_embedding = await embedder.embed_text(request.query)

    # Step 2: Hybrid Search (get more candidates than needed)
    hybrid_service = HybridSearchService(qdrant)

    # TODO: Pre-build BM25 index in production
    # For now, hybrid search falls back to vector-only

    results = await hybrid_service.hybrid_search(
        query=request.query,
        query_embedding=query_embedding,
        top_k=num_candidates,
        keyword_weight=request.keyword_weight,
        vector_weight=request.vector_weight,
        metadata_filters=request.metadata_filters,
    )

    # Step 3: Cross-Encoder Reranking (optional)
    if request.use_reranker and len(results) > 0:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(
            query=request.query, results=results, top_k=request.top_k, score_key="text"
        )
    else:
        results = results[: request.top_k]

    # Convert to SearchResult objects
    search_results = [
        SearchResult(
            chunk_id=r.get("chunk_id", ""),
            document_id=r.get("document_id", ""),
            text=r["text"],
            score=r.get("cross_encoder_score", r.get("hybrid_score", 0.0)),
            metadata={
                **r.get("metadata", {}),
                "hybrid_score": r.get("hybrid_score", 0.0),
                "vector_score": r.get("vector_score", 0.0),
                "bm25_score": r.get("bm25_score", 0.0),
                "cross_encoder_score": r.get("cross_encoder_score"),
            },
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query, results=search_results, total_results=len(search_results)
    )


@router.get("/reranker/info", tags=["Reranking"])
async def reranker_info():
    """
    Get information about the reranking system.
    """
    reranker = CrossEncoderReranker()

    return {
        "reranker": reranker.get_info(),
        "pipeline": {
            "step_1": "Hybrid Search (BM25 + Vector)",
            "step_2": "Cross-Encoder Reranking",
            "step_3": "Return top K",
        },
        "why_it_works": {
            "hybrid_search": "Combines exact keyword matches with semantic similarity",
            "cross_encoder": "Scores (query, doc) pairs together for precise relevance",
            "result": "High accuracy without sacrificing recall",
        },
        "use_cases": [
            "Master Data: Exact SKU + product description matching",
            "Invoice Search: Exact invoice number + vendor name",
            "Email Search: Exact sender + semantic content",
        ],
    }
