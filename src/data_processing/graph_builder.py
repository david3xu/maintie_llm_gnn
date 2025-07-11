import logging
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleGraphBuilder:
    """Simple graph construction with semantic similarity"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize graph builder"""
        self.config = config['data']
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.k_neighbors = self.config.get('k_neighbors', 10)
        logging.info(f"GraphBuilder initialized with threshold={self.similarity_threshold} and k={self.k_neighbors}")

    def build_graph(self, embeddings: np.ndarray, texts: List[str], corpus: List[Dict], entity_map: Dict, relation_map: Dict, gold_texts: set) -> Data:
        """Build PyTorch Geometric graph with labels and masks"""
        if embeddings.size == 0:
            logging.warning("Embeddings array is empty. Returning an empty graph.")
            return Data()

        logging.info("Building graph from embeddings...")
        edge_index, edge_weight = self._build_similarity_edges(embeddings)

        y_entity, y_relation = self._create_labels(texts, corpus, entity_map, relation_map)

        # Create train/val masks based on gold/silver corpus
        num_nodes = embeddings.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for i, text in enumerate(texts):
            if text in gold_texts:
                val_mask[i] = True
            else:
                train_mask[i] = True

        graph = Data(
            x=torch.from_numpy(embeddings).float(),
            edge_index=edge_index,
            edge_attr=edge_weight,
            y_entity=y_entity,
            y_relation=y_relation,
            train_mask=train_mask,
            val_mask=val_mask,
            texts=texts,
            entity_map=entity_map,
            relation_map=relation_map
        )

        logging.info(f"Graph built with {graph.num_nodes} nodes, {graph.num_edges} edges. "
                     f"Training nodes: {train_mask.sum()}, Validation nodes: {val_mask.sum()}")
        return graph

    def _create_labels(self, texts: List[str], corpus: List[Dict], entity_map: Dict, relation_map: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create multi-hot encoded labels for entities and relations"""
        num_nodes = len(texts)
        num_entities = len(entity_map)
        num_relations = len(relation_map)

        y_entity = torch.zeros((num_nodes, num_entities), dtype=torch.float)
        y_relation = torch.zeros((num_nodes, num_relations), dtype=torch.float)

        text_to_idx = {text: i for i, text in enumerate(texts)}

        for sample in corpus:
            text = sample['text']
            if text in text_to_idx:
                node_idx = text_to_idx[text]
                # Encode entities
                for entity in sample.get('entities', []):
                    entity_type = entity['type']
                    if entity_type in entity_map:
                        y_entity[node_idx, entity_map[entity_type]] = 1
                # Encode relations
                for relation in sample.get('relations', []):
                    relation_type = relation['type']
                    if relation_type in relation_map:
                        y_relation[node_idx, relation_map[relation_type]] = 1

        return y_entity, y_relation

    def _build_similarity_edges(self, embeddings: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges based on cosine similarity"""
        logging.info(f"Building edges using k-NN (k={self.k_neighbors}) and similarity threshold...")

        # Use k-NN for efficiency to find potential neighbors
        knn_graph = kneighbors_graph(
            embeddings, self.k_neighbors, mode='connectivity', include_self=False
        )

        # Get source and target nodes from the k-NN graph
        sources, targets = knn_graph.nonzero()

        # Calculate cosine similarity only for the k-NN pairs
        similarities = cosine_similarity(embeddings[sources], embeddings[targets]).diagonal()

        # Filter edges based on the similarity threshold
        mask = similarities >= self.similarity_threshold

        edge_index = torch.tensor([sources[mask], targets[mask]], dtype=torch.long)
        edge_weight = torch.tensor(similarities[mask], dtype=torch.float)

        return edge_index, edge_weight

    def save_graph(self, graph: Data, output_path: str) -> None:
        """Save graph to disk"""
        logging.info(f"Saving graph to {output_path}...")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
            logging.info("Graph saved successfully.")
        except IOError as e:
            logging.error(f"Failed to save graph to {output_path}: {e}")

    def load_graph(self, input_path: str) -> Data:
        """Load graph from disk"""
        logging.info(f"Loading graph from {input_path}...")
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        logging.info("Graph loaded successfully.")
        return graph

    def get_graph_statistics(self, graph: Data) -> Dict[str, Any]:
        """Calculate graph statistics"""
        if not isinstance(graph, Data) or graph.num_nodes == 0:
            return {'nodes': 0, 'edges': 0, 'avg_degree': 0}

        stats = {
            'nodes': graph.num_nodes,
            'edges': graph.num_edges,
            'avg_degree': graph.num_edges / graph.num_nodes if graph.num_nodes > 0 else 0,
            'is_directed': graph.is_directed(),
        }
        logging.info(f"Graph statistics: {stats}")
        return stats

def main():
    """CLI entry point for make build-graphs"""
    import yaml
    from src.data_processing.embedding_generator import EmbeddingGenerator
    from src.data_processing.data_loader import load_maintie_data

    logging.info("Starting graph building process from CLI...")
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Load data to get corpus and maps
        data = load_maintie_data(config)
        corpus = data['gold_corpus'] + data['silver_corpus']
        entity_map = data['entity_map']
        relation_map = data['relation_map']
        gold_texts = {s['text'] for s in data['gold_corpus']}

        # Load embeddings
        emb_path = config['paths']['embeddings']
        embed_gen = EmbeddingGenerator()
        embeddings, texts = embed_gen.load_embeddings(emb_path)

        if embeddings.size == 0:
            logging.error("Embeddings are empty. Cannot build graph.")
            return

        builder = SimpleGraphBuilder(config)
        graph = builder.build_graph(embeddings, texts, corpus, entity_map, relation_map, gold_texts)

        output_path = config['paths']['graphs']
        builder.save_graph(graph, output_path)

        # Print stats at the end
        builder.get_graph_statistics(graph)

    except FileNotFoundError:
        logging.error(f"Configuration file or embedding file not found.")
    except Exception as e:
        logging.error(f"An error occurred during graph building: {e}", exc_info=True)

if __name__ == "__main__":
    main()
