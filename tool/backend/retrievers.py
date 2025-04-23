from sentence_transformers import SentenceTransformer
import faiss

class FaissRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunk_texts = None

    def build_index(self, chunks):
        self.chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.encode(self.chunk_texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query, top_k: int = 3):
        q_embed = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_embed)
        distances, indices = self.index.search(q_embed, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({"text": self.chunk_texts[idx], "score": float(dist)})
        return results
