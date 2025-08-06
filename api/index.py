"""
Vercel deployment entry point for High-Performance RAG System
"""

from app.main import app

# This is required for Vercel to detect the FastAPI app
handler = app
