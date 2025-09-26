"""
Configuration management for the AI-powered financial document processing system.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/financial_docs"
    
    # OpenAI configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-1106-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_OUTPUT_TOKENS: int = 3000
    
    # Pinecone configuration
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "financial-documents"
    PINECONE_DIMENSION: int = 1536

    # LangSmith configuration
    LANGSMITH_TRACING: bool = True
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str = "pr-new-circumstance-21"
    
    # Application configuration
    APP_NAME: str = "Docufi Demo"
    DEBUG: bool = False
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: list = [".pdf"]
    
    # Upload directory
    UPLOAD_DIRECTORY: str = "/tmp/uploads"
    
    # Chunk settings for document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Metadata extraction settings
    USE_VECTOR_REGEX_EXTRACTION: bool = True
    EXTRACTION_SIMILARITY_THRESHOLD: float = 0.6
    EXTRACTION_TOP_K_CHUNKS: int = 5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings