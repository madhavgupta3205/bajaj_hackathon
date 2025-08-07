"""
Lightweight embedding service for Vercel deployment
Uses scikit-learn's TF-IDF instead of heavy sentence-transformers
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import pickle
import os
import hashlib
import diskcache


class LightweightEmbedding:
    """Lightweight embedding using TF-IDF vectorization"""

    def __init__(self, cache_dir: str = "/tmp/embedding_cache"):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )
        self.cache = diskcache.Cache(cache_dir)
        self.document_vectors = None
        self.documents = []

    def fit_documents(self, documents: List[str]):
        """Fit the vectorizer on documents"""
        self.documents = documents
        self.document_vectors = self.vectorizer.fit_transform(documents)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors"""
        if isinstance(texts, str):
            texts = [texts]

        # Check cache first
        cache_key = hashlib.md5(str(texts).encode()).hexdigest()
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Transform texts
        vectors = self.vectorizer.transform(texts)
        result = vectors.toarray()

        # Cache result
        self.cache.set(cache_key, result)
        return result

    def search_similar(self, query: str, top_k: int = 5) -> List[str]:
        """Find most similar documents to query"""
        if self.document_vectors is None:
            return []

        query_vector = self.encode([query])
        similarities = cosine_similarity(
            query_vector, self.document_vectors)[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return corresponding documents
        return [self.documents[i] for i in top_indices if similarities[i] > 0.1]


class SimpleLLM:
    """Simple LLM wrapper for Gemini"""

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    async def generate_answer(self, question: str, context: str) -> str:
        """Generate answer with lightweight prompting"""
        prompt = f"""Answer the question based on the context provided.

Context: {context[:2000]}  # Limit context to avoid token limits

Question: {question}

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
