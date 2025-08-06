"""
High-Performance RAG System with LlamaIndex and Gemini
Fast, reliable, and accurate document Q&A system
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
from dotenv import load_dotenv

from app.config import settings
from app.rag_engine import RAGEngine
from app.models import QueryRequest, QueryResponse, HealthResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global RAG engine instance
rag_engine: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global rag_engine

    logger.info("🚀 Starting High-Performance RAG System...")

    try:
        # Initialize RAG engine
        rag_engine = RAGEngine()
        await rag_engine.initialize()
        logger.info("✅ RAG Engine initialized successfully")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG engine: {e}")
        raise
    finally:
        # Cleanup
        if rag_engine:
            await rag_engine.cleanup()
        logger.info("🔄 Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="High-Performance RAG System",
    description="Lightning-fast document Q&A with LlamaIndex and Gemini",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "High-Performance RAG System v2.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if not rag_engine:
            raise HTTPException(
                status_code=503, detail="RAG engine not initialized")

        status = await rag_engine.get_health_status()
        return HealthResponse(**status)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/hackrx/run", response_model=Dict[str, List[str]])
async def process_queries(request: QueryRequest) -> Dict[str, List[str]]:
    """
    HackRX API endpoint - Process document and answer queries
    Returns clean format with just answers array
    """
    try:
        logger.info(
            f"📥 Processing request with {len(request.questions)} questions")
        logger.info(f"📄 Document URL: {request.documents}")

        # Process document and get answers
        answers = await rag_engine.process_document_and_query(
            document_url=str(request.documents),
            questions=request.questions
        )

        logger.info(f"✅ Successfully processed {len(answers)} answers")

        return {"answers": answers}

    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        # Return error answers that maintain the expected format
        error_message = f"Error processing questions: {str(e)}"
        return {"answers": [error_message] * len(request.questions)}


@app.post("/query", response_model=List[QueryResponse])
async def detailed_query(request: QueryRequest) -> List[QueryResponse]:
    """
    Detailed query endpoint - Returns full response with metadata
    """
    try:
        logger.info(
            f"📥 Processing detailed request with {len(request.questions)} questions")

        # Process document and get detailed responses
        responses = await rag_engine.process_document_and_query_detailed(
            document_url=str(request.documents),
            questions=request.questions
        )

        logger.info(
            f"✅ Successfully processed {len(responses)} detailed responses")

        return responses

    except Exception as e:
        logger.error(f"❌ Error processing detailed request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
