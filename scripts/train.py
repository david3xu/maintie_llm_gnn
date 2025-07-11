import logging
import yaml
from typing import Any, Dict, List, Tuple
from torch_geometric.data import Data
from src.data_processing.data_loader import load_maintie_data
from src.data_processing.embedding_generator import EmbeddingGenerator
from src.data_processing.graph_builder import SimpleGraphBuilder
from src.models.llm_gnn_hybrid import MaintIELLMGNNHybrid, create_model
from src.training.simple_trainer import SimpleTrainer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_data_pipeline(config: Dict[str, Any]) -> Tuple[Data, Dict, Dict]:
    """Setup complete data pipeline, returning a single graph Data object."""

    graph_path = config['paths']['graphs']
    try:
        graph = SimpleGraphBuilder(config).load_graph(graph_path)
        logging.info(f"Loaded pre-built graph from {graph_path}.")
        # Entity and relation maps are now stored on the graph object
        return graph, graph.entity_map, graph.relation_map
    except FileNotFoundError:
        logging.info("Pre-built graph not found. Building new graph...")

    # If graph doesn't exist, build it
    data = load_maintie_data(config)
    corpus = data['gold_corpus'] + data['silver_corpus']
    gold_texts = {s['text'] for s in data['gold_corpus']}

    all_texts = sorted(list(set([s['text'] for s in corpus])))

    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(all_texts)

    builder = SimpleGraphBuilder(config)
    graph = builder.build_graph(embeddings, all_texts, corpus, data['entity_map'], data['relation_map'], gold_texts)
    builder.save_graph(graph, graph_path)

    return graph, data['entity_map'], data['relation_map']

def initialize_training(config: Dict[str, Any], num_entity_classes: int, num_relation_classes: int) -> Tuple[MaintIELLMGNNHybrid, SimpleTrainer]:
    """Initialize model and trainer"""
    model = create_model(config, num_entity_classes, num_relation_classes)
    trainer = SimpleTrainer(config)
    return model, trainer

def main():
    """Main training script"""
    logging.info("--- Starting Training Pipeline ---")

    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        logging.info("Configuration loaded.")

        # 1. Setup Data Pipeline
        logging.info("Setting up data pipeline...")
        graph_data, entity_map, relation_map = setup_data_pipeline(config)

        if graph_data.num_nodes == 0:
            logging.error("Data pipeline did not produce a valid graph. Halting training.")
            return

        # 2. Initialize Model and Trainer
        logging.info("Initializing model and trainer...")
        model, trainer = initialize_training(config, len(entity_map), len(relation_map))

        # 3. Run Training
        logging.info("Starting model training...")
        training_results = trainer.train(model, graph_data)

        logging.info(f"Training finished. Best validation loss: {training_results['best_val_loss']:.4f}")

        # 4. Save final model
        final_model_path = f"{config['paths']['models']}final_model.pt"
        model.save_model(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    except FileNotFoundError:
        logging.error("Configuration file 'config/config.yaml' not found.")
    except Exception as e:
        logging.error(f"An error occurred during the training pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()
