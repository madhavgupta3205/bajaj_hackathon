"""Configuration settings for the RAG system"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""

    # API Keys and Authentication
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key")
    HACKRX_BEARER_TOKEN: str = Field(default="a928ab38f03560bdb4b9c3930ca021cf0f1c753febc6a637fb996cb4f30c35c8",
                                     description="Bearer token for HackRX API authentication")

    # Model Configuration

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Database Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

    # Performance Settings
    MAX_CHUNKS_PER_QUERY: int = int(os.getenv("MAX_CHUNKS_PER_QUERY", "10"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL: str = "gemini-1.5-flash"
    TEMPERATURE: float = 0.1

    # Cache Settings
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
