"""
Hybrid Search: Dense (Vector) + Sparse (BM25) Search.

The "Senior" Pattern:
- Use BM25 for exact keyword matches (SKU: XJ-900)
- Use Vector Search for semantic matches ("metal rods")
- Combine scores with configurable weights
"""
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from app.services.vector.qdrant_client import QdrantClient
from app.services.embedding.base import EmbedderBase


class HybridSearchService:
    """
    Hybrid search combining dense and sparse retrieval.
    
    Use Cases:
    - Master data search (exact SKU + semantic product description)
    - Invoice search (exact invoice number + fuzzy vendor name)
    - Email search (exact sender + semantic content)
    """
    
    def __init__(self, qdrant_client: QdrantClient):
        """
        Initialize hybrid search service.
        
        Args:
            qdrant_client: Qdrant client for dense retrieval
        """
        self.qdrant = qdrant_client
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
    
    def index_for_bm25(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """
        Build BM25 index for keyword search.
        
        Args:
            documents: List of document texts
            metadata: Metadata for each document
        """
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.bm25_documents = documents
        self.bm25_metadata = metadata
    
    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            top_k: Number of results
            keyword_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)
            metadata_filters: Optional filters
        
        Returns:
            Combined and reranked results
        """
        # Normalize weights
        total_weight = keyword_weight + vector_weight
        keyword_weight = keyword_weight / total_weight
        vector_weight = vector_weight / total_weight
        
        # 1. Dense retrieval (vector search)
        vector_results = await self.qdrant.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more candidates
            score_threshold=None  # No threshold for hybrid
        )
        
        # 2. Sparse retrieval (BM25)
        bm25_results = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top BM25 results
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k * 2]
            
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    bm25_results.append({
                        "text": self.bm25_documents[idx],
                        "metadata": self.bm25_metadata[idx],
                        "bm25_score": bm25_scores[idx]
                    })
        
        # 3. Combine results with weighted scores
        combined_results = self._combine_results(
            vector_results,
            bm25_results,
            vector_weight,
            keyword_weight
        )
        
        # 4. Apply metadata filters if provided
        if metadata_filters:
            combined_results = self._apply_filters(combined_results, metadata_filters)
        
        # 5. Sort by hybrid score and return top-k
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return combined_results[:top_k]
    
    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        keyword_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and BM25 results with weighted scores.
        """
        # Normalize vector scores to 0-1 range
        if vector_results:
            max_vector_score = max(r["score"] for r in vector_results)
            min_vector_score = min(r["score"] for r in vector_results)
            score_range = max_vector_score - min_vector_score
            
            if score_range > 0:
                for r in vector_results:
                    r["normalized_vector_score"] = (
                        (r["score"] - min_vector_score) / score_range
                    )
            else:
                for r in vector_results:
                    r["normalized_vector_score"] = 1.0
        
        # Normalize BM25 scores to 0-1 range
        if bm25_results:
            max_bm25_score = max(r["bm25_score"] for r in bm25_results)
            
            if max_bm25_score > 0:
                for r in bm25_results:
                    r["normalized_bm25_score"] = r["bm25_score"] / max_bm25_score
            else:
                for r in bm25_results:
                    r["normalized_bm25_score"] = 0.0
        
        # Combine results
        results_dict = {}
        
        # Add vector results
        for r in vector_results:
            text = r["text"]
            if text not in results_dict:
                results_dict[text] = {
                    "text": text,
                    "chunk_id": r.get("chunk_id"),
                    "document_id": r.get("document_id"),
                    "metadata": r.get("metadata", {}),
                    "vector_score": r["normalized_vector_score"],
                    "bm25_score": 0.0
                }
        
        # Add BM25 results
        for r in bm25_results:
            text = r["text"]
            if text in results_dict:
                results_dict[text]["bm25_score"] = r["normalized_bm25_score"]
            else:
                results_dict[text] = {
                    "text": text,
                    "chunk_id": None,
                    "document_id": r["metadata"].get("document_id"),
                    "metadata": r.get("metadata", {}),
                    "vector_score": 0.0,
                    "bm25_score": r["normalized_bm25_score"]
                }
        
        # Calculate hybrid scores
        combined_results = []
        for result in results_dict.values():
            hybrid_score = (
                result["vector_score"] * vector_weight +
                result["bm25_score"] * keyword_weight
            )
            result["hybrid_score"] = hybrid_score
            combined_results.append(result)
        
        return combined_results
    
    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters to results."""
        filtered = []
        
        for result in results:
            metadata = result.get("metadata", {})
            matches = True
            
            for key, value in filters.items():
                if metadata.get(key) != value:
                    matches = False
                    break
            
            if matches:
                filtered.append(result)
        
        return filtered

