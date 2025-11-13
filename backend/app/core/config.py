"""
Application configuration using Pydantic Settings.
All config is type-safe and loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    app_name: str = "Enterprise RAG Platform"
    environment: str = "development"
    debug: bool = True
    
    # OpenAI
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
    # Qdrant (Vector DB)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"
    
    # Neo4j (Graph DB)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Upload
    upload_dir: str = "./uploads"
    max_upload_size: int = 52428800  # 50MB
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

