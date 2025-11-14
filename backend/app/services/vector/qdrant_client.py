"""
Qdrant vector database client.
Handles all interactions with Qdrant for indexing and searching.
"""
from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import QdrantClient as QdrantSDK
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    SearchParams
)
from app.core.config import Settings


class QdrantClient:
    """
    Qdrant client for vector operations.
    Handles collections, indexing, and similarity search.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Qdrant client.
        
        Args:
            settings: Application settings with Qdrant config
        """
        self.settings = settings
        self.client = QdrantSDK(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
    
    def ensure_collection(self, dimension: int) -> None:
        """
        Ensure collection exists with correct configuration.
        
        Args:
            dimension: Embedding vector dimension
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE  # Cosine similarity for semantic search
                )
            )
            print(f"✅ Created collection: {self.collection_name}")
        else:
            print(f"✅ Collection already exists: {self.collection_name}")
    
    async def index_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        document_id: str,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Index chunks with their embeddings.
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            document_id: ID of the source document
            metadata: Optional metadata for each chunk
            
        Returns:
            Number of chunks indexed
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if metadata is None:
            metadata = [{} for _ in chunks]
        
        # Create points for Qdrant
        points = []
        for idx, (chunk, embedding, meta) in enumerate(zip(chunks, embeddings, metadata)):
            point_id = str(uuid.uuid4())
            
            # Combine metadata
            payload = {
                "document_id": document_id,
                "chunk_id": point_id,
                "chunk_index": idx,
                "text": chunk,
                **meta
            }
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upsert to Qdrant (batch operation)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return len(points)
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=None  # TODO: Add filter support
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.payload.get("chunk_id"),
                "document_id": result.payload.get("document_id"),
                "text": result.payload.get("text"),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() 
                           if k not in ["chunk_id", "document_id", "text"]}
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "error": str(e),
                "exists": False
            }
    
    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

