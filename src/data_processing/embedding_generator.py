import logging
import pickle
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingGenerator:
    """Simple LLM embedding generation"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with lightweight model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logging.info(f"Embedding model '{model_name}' loaded.")

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for text list"""
        if not texts:
            logging.warning("Input text list is empty. Returning empty array.")
            return np.array([])

        logging.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        logging.info(f"Embeddings generated with shape: {embeddings.shape}")
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], output_path: str) -> None:
        """Save embeddings to disk"""
        logging.info(f"Saving embeddings to {output_path}...")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump({'texts': texts, 'embeddings': embeddings}, f)
            logging.info("Embeddings saved successfully.")
        except IOError as e:
            logging.error(f"Failed to save embeddings to {output_path}: {e}")

    def load_embeddings(self, input_path: str) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings from disk"""
        logging.info(f"Loading embeddings from {input_path}...")
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            logging.info("Embeddings loaded successfully.")
            return data['embeddings'], data['texts']
        except FileNotFoundError:
            logging.error(f"Embedding file not found at {input_path}.")
            return np.array([]), []
        except (IOError, pickle.UnpicklingError) as e:
            logging.error(f"Failed to load embeddings from {input_path}: {e}")
            return np.array([]), []

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        dim = self.model.get_sentence_embedding_dimension()
        logging.info(f"Embedding dimension: {dim}")
        return dim

def main():
    """CLI entry point for make generate-embeddings"""
    import yaml
    from src.data_processing.data_loader import load_maintie_data

    logging.info("Starting embedding generation process from CLI...")
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        data = load_maintie_data(config)
        all_texts = [sample['text'] for sample in data['gold_corpus']] + \
                    [sample['text'] for sample in data['silver_corpus']]

        unique_texts = sorted(list(set(all_texts)))

        if not unique_texts:
            logging.warning("No unique texts found to generate embeddings for. Exiting.")
            return

        generator = EmbeddingGenerator()
        embeddings = generator.generate_embeddings(unique_texts)

        output_path = config['paths']['embeddings']
        generator.save_embeddings(embeddings, unique_texts, output_path)

    except FileNotFoundError:
        logging.error("Configuration file 'config/config.yaml' not found.")
    except Exception as e:
        logging.error(f"An error occurred during embedding generation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
