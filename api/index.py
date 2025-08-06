"""
Vercel deployment entry point for High-Performance RAG System
"""

import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the FastAPI app
from app.main import app

# Export for Vercel
# Vercel looks for 'app' or 'handler' in the entry file
app = app
