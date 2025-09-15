import faiss
import numpy as np
from mistral_client import get_embedding

# class VectorStore:
#     def __init__(self):
#         self.index = None
#         self.chunks = []

#     def build_index(self, chunks):
#         self.chunks = chunks
#         embeddings = [get_embedding(chunk) for chunk in chunks]
#         dim = len(embeddings[0])
#         self.index = faiss.IndexFlatL2(dim)
#         self.index.add(np.array(embeddings).astype("float32"))

#     def search(self, query, k=3):
#         q_emb = np.array([get_embedding(query)]).astype("float32")
#         distances, indices = self.index.search(q_emb, k)
#         return [self.chunks[i] for i in indices[0]]

# vector_store.py
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

