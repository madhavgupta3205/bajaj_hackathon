"""
High-Performance RAG Engine using Gemini and ChromaDB
Built from scratch for maximum reliability and speed
"""

import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import re

import google.generativeai as genai
import chromadb
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import QueryResponse, DocumentInfo, CacheInfo

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Fast document processing with caching"""

    @staticmethod
    async def download_pdf(url: str) -> bytes:
        """Download PDF from URL with error handling"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            raise

    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            import PyPDF2
            import io

            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise

    @staticmethod
    def get_document_id(url: str) -> str:
        """Generate unique document ID from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    @staticmethod
    def smart_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Smart text chunking with semantic   awareness"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + \
                        "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If chunks are too small, combine them
        final_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]

            # Combine small chunks
            while i + 1 < len(chunks) and len(chunk) < chunk_size // 2:
                chunk += "\n\n" + chunks[i + 1]
                i += 1

            final_chunks.append(chunk)
            i += 1

        return final_chunks


class GeminiLLM:
    """Gemini LLM wrapper with optimized prompts"""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.9,
            top_k=20,
            max_output_tokens=1024,
        )

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Gemini with retry logic"""

        prompt = f"""Based on the insurance policy document provided, answer the following question accurately and concisely.

Context from insurance policy:
{context}

Question: {question}

Please provide a direct, factual answer based only on the information in the context. If the information is not available in the context, say so.

Answer:"""

        try:
            response = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate_content(
                        prompt,
                        generation_config=self.generation_config
                    )
                ),
                timeout=30.0
            )
            return response.text.strip()
        except asyncio.TimeoutError:
            logger.error("Gemini API timeout")
            return "Answer generation timed out. Please try again."
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Unable to generate answer due to API error."


class RAGEngine:
    """High-performance RAG engine with ChromaDB and Gemini"""

    def __init__(self):
        self.collection = None
        self.chroma_client = None
        self.embedding_model = None
        self.llm = None
        self.doc_processor = DocumentProcessor()
        self.processed_docs: Dict[str, DocumentInfo] = {}

    async def initialize(self):
        """Initialize the RAG engine"""
        try:
            logger.info("🔧 Initializing RAG Engine...")

            # Configure Gemini API
            if not settings.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is required")

            # Initialize LLM
            self.llm = GeminiLLM(
                api_key=settings.GEMINI_API_KEY,
                model=settings.LLM_MODEL
            )

            # Initialize embedding model
            logger.info("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL)

            # Initialize ChromaDB
            os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH)

            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(
                    "documents")
                logger.info(
                    f"📚 Found existing collection with {self.collection.count()} documents")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"description": "Document collection for RAG"}
                )
                logger.info("📚 Created new document collection")

            logger.info("🚀 RAG Engine initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Engine: {e}")
            raise

    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            doc_count = self.collection.count() if self.collection else 0

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow(),
                "components": {
                    "vector_store": {
                        "status": "healthy",
                        "document_count": doc_count,
                        "collection_ready": self.collection is not None
                    },
                    "llm": {
                        "status": "healthy",
                        "model": settings.LLM_MODEL
                    },
                    "embeddings": {
                        "status": "healthy",
                        "model": settings.EMBEDDING_MODEL
                    }
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow(),
                "error": str(e)
            }

    async def is_document_cached(self, document_url: str) -> bool:
        """Check if document is already processed and cached"""
        doc_id = self.doc_processor.get_document_id(document_url)

        try:
            results = self.collection.get(
                where={"document_id": doc_id},
                limit=1
            )
            return len(results["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking cache for {document_url}: {e}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_document(self, document_url: str) -> DocumentInfo:
        """Process and index a document"""
        start_time = time.time()
        doc_id = self.doc_processor.get_document_id(document_url)

        try:
            logger.info(f"📄 Processing document: {document_url}")

            # Download document
            pdf_content = await self.doc_processor.download_pdf(document_url)
            logger.info(f"⬇️ Downloaded document ({len(pdf_content)} bytes)")

            # Extract text
            text = self.doc_processor.extract_text_from_pdf(pdf_content)
            logger.info(f"📝 Extracted text ({len(text)} characters)")

            # Create smart chunks
            chunks = self.doc_processor.smart_chunk_text(
                text,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
            logger.info(f"🧩 Created {len(chunks)} smart chunks")

            # Generate embeddings
            logger.info("🔄 Generating embeddings...")
            embeddings = self.embedding_model.encode(
                chunks, show_progress_bar=True)

            # Prepare data for ChromaDB
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": doc_id,
                    "source_url": document_url,
                    "chunk_index": i,
                    "processed_at": datetime.utcnow().isoformat(),
                    "file_size": len(pdf_content)
                }
                for i in range(len(chunks))
            ]

            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"✅ Indexed document with {len(chunks)} chunks")

            # Store document info
            doc_info = DocumentInfo(
                document_id=doc_id,
                url=document_url,
                title=f"Document_{doc_id[:8]}",
                chunk_count=len(chunks),
                processed_at=datetime.utcnow(),
                file_size=len(pdf_content)
            )

            self.processed_docs[doc_id] = doc_info

            processing_time = time.time() - start_time
            logger.info(
                f"🎯 Document processed successfully in {processing_time:.2f}s")

            return doc_info

        except Exception as e:
            logger.error(f"❌ Failed to process document {document_url}: {e}")
            raise

    async def retrieve_relevant_chunks(self, question: str, top_k: int = None) -> List[str]:
        """Retrieve relevant chunks for a question"""
        if top_k is None:
            top_k = settings.MAX_CHUNKS_PER_QUERY

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

            # Extract documents
            if results["documents"] and len(results["documents"]) > 0:
                return results["documents"][0]
            else:
                return []

        except Exception as e:
            logger.error(f"❌ Failed to retrieve chunks: {e}")
            return []

    async def query_single(self, question: str) -> QueryResponse:
        """Process a single query"""
        start_time = time.time()

        try:
            logger.debug(f"🔍 Processing query: {question[:50]}...")

            # Retrieve relevant chunks
            relevant_chunks = await self.retrieve_relevant_chunks(question)

            if not relevant_chunks:
                return QueryResponse(
                    question=question,
                    answer="I don't have enough information to answer this question.",
                    confidence_score=0.0,
                    sources=[],
                    metadata={"chunks_used": 0},
                    processing_time=time.time() - start_time
                )

            # Combine chunks into context
            context = "\n\n".join(relevant_chunks)

            # Generate answer using Gemini
            answer = await self.llm.generate_answer(question, context)

            processing_time = time.time() - start_time

            return QueryResponse(
                question=question,
                answer=answer,
                confidence_score=0.85,  # Default confidence
                sources=[],  # Would need to extract from metadata
                metadata={
                    "model": settings.LLM_MODEL,
                    "chunks_used": len(relevant_chunks)
                },
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return QueryResponse(
                question=question,
                answer=f"Sorry, I encountered an error processing your question: {str(e)}",
                confidence_score=0.0,
                sources=[],
                metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )

    async def process_document_and_query(
        self,
        document_url: str,
        questions: List[str]
    ) -> List[str]:
        """
        Process document and answer questions - returns simple answers list
        """
        try:
            # Check if document is cached
            if not await self.is_document_cached(document_url):
                logger.info("📄 Document not cached, processing...")
                await self.process_document(document_url)
            else:
                logger.info("⚡ Using cached document")

            # Process queries
            answers = []
            for question in questions:
                response = await self.query_single(question)
                answers.append(response.answer)

            return answers

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            # Return error messages for all questions
            error_msg = f"Error processing document: {str(e)}"
            return [error_msg] * len(questions)

    async def process_document_and_query_detailed(
        self,
        document_url: str,
        questions: List[str]
    ) -> List[QueryResponse]:
        """
        Process document and answer questions - returns detailed responses
        """
        try:
            # Check if document is cached
            if not await self.is_document_cached(document_url):
                logger.info("📄 Document not cached, processing...")
                await self.process_document(document_url)
            else:
                logger.info("⚡ Using cached document")

            # Process queries
            responses = []
            for question in questions:
                response = await self.query_single(question)
                responses.append(response)

            return responses

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            # Return error responses for all questions
            error_responses = []
            for question in questions:
                error_responses.append(QueryResponse(
                    question=question,
                    answer=f"Error processing document: {str(e)}",
                    confidence_score=0.0,
                    sources=[],
                    metadata={"error": str(e)},
                    processing_time=0.0
                ))
            return error_responses

    async def add_document(self, file_path: str) -> DocumentInfo:
        """Add a local document file to the index"""
        try:
            with open(file_path, 'rb') as file:
                pdf_content = file.read()

            text = self.doc_processor.extract_text_from_pdf(pdf_content)
            doc_id = self.doc_processor.get_document_id(file_path)

            chunks = self.doc_processor.smart_chunk_text(text)
            embeddings = self.embedding_model.encode(chunks)

            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "document_id": doc_id,
                    "source_url": file_path,
                    "chunk_index": i,
                    "processed_at": datetime.utcnow().isoformat(),
                    "file_size": len(pdf_content)
                }
                for i in range(len(chunks))
            ]

            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )

            doc_info = DocumentInfo(
                document_id=doc_id,
                url=file_path,
                title=os.path.basename(file_path),
                chunk_count=len(chunks),
                processed_at=datetime.utcnow(),
                file_size=len(pdf_content)
            )

            self.processed_docs[doc_id] = doc_info
            return doc_info

        except Exception as e:
            logger.error(f"❌ Failed to add document {file_path}: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cleanup is automatic for ChromaDB and SentenceTransformers
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
