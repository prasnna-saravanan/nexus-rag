"""
Factory for creating embedding providers.
Makes it easy to swap between different embedding models.
"""
from app.models.schemas import EmbeddingProvider
from app.core.config import Settings
from .base import EmbedderBase
from .openai_embedder import OpenAIEmbedder


class EmbedderFactory:
    """Factory to create embedding providers."""
    
    @staticmethod
    def create(provider: EmbeddingProvider, settings: Settings) -> EmbedderBase:
        """
        Create an embedder based on the provider name.
        
        Args:
            provider: The embedding provider to use
            settings: Application settings
            
        Returns:
            An EmbedderBase instance
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMER:
            # TODO: Implement sentence-transformers
            raise NotImplementedError("Sentence Transformers coming soon!")
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

