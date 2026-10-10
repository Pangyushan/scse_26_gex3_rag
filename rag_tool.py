"""RAG retrieval tool: embed policy documents and retrieve top-k relevant chunks."""

import numpy as np
from langchain_ollama import OllamaEmbeddings

from policy_loader import load_policy_documents

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://127.0.0.1:11434"
TOP_K = 3


class RagTool:
    def __init__(self):
        self.documents = load_policy_documents()
        self.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_URL,
        )
        print(f"[rag_tool] Loaded {len(self.documents)} policy documents. Embedding...")
        self.doc_vectors = self._embed_documents(self.documents)
        print(f"[rag_tool] Done. Vector shape: {self.doc_vectors.shape}")

    def _embed_documents(self, documents):
        texts = [d.page_content for d in documents]
        vectors = self.embeddings.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def _embed_query(self, query):
        vec = self.embeddings.embed_query(query)
        return np.array(vec, dtype=np.float32)

    def retrieve(self, query, k=TOP_K):
        q_vec = self._embed_query(query)
        doc_norms = np.linalg.norm(self.doc_vectors, axis=1)
        q_norm = np.linalg.norm(q_vec)
        sims = (self.doc_vectors @ q_vec) / (doc_norms * q_norm + 1e-8)
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.documents[i] for i in top_idx]