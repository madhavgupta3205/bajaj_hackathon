"""Pydantic models for the RAG system"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, HttpUrl, Field, validator
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for document queries"""
    documents: Union[HttpUrl,
                     str] = Field(..., description="URL to document or document content")
    questions: List[str] = Field(..., min_items=1, max_items=50,
                                 description="List of questions to answer")

    @validator('questions')
    def validate_questions(cls, v):
        for question in v:
            if len(question.strip()) < 3:
                raise ValueError(
                    "Each question must be at least 3 characters long")
        return v


class QueryResponse(BaseModel):
    """Response model for individual query"""
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str = "2.0.0"
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """Document information model"""
    document_id: str
    url: str
    title: Optional[str] = None
    chunk_count: int
    processed_at: datetime
    file_size: Optional[int] = None


class CacheInfo(BaseModel):
    """Cache information model"""
    total_documents: int
    total_chunks: int
    cache_size_mb: float
    last_updated: datetime
