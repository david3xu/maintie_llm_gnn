from typing import List
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """LLM embedding generation for MaintIE data"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text corpus"""
        return self.model.encode(texts, batch_size=32, show_progress_bar=True)

    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """Save embeddings to disk"""
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings, f)
