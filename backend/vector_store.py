import faiss
import numpy as np
from mistral_client import get_embedding

class VectorStore:
    def __init__(self):
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        # TODO: build your embedding index if using FAISS, etc.

    def get_relevant_chunks(self, query, top_k=5):
        """
        Return the most relevant text chunks for the given query.
        For now, just return the first `top_k` chunks as a placeholder.
        """
        if not self.chunks:
            return []
        # TODO: implement semantic search with embeddings
        return self.chunks[:top_k]

