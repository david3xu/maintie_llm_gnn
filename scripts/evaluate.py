import logging
import yaml
from typing import Dict
from torch_geometric.data import Data
from src.models.llm_gnn_hybrid import MaintIELLMGNNHybrid, load_model
from src.evaluation.evaluator import GoldStandardEvaluator
from src.data_processing.graph_builder import SimpleGraphBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_trained_model(model_path: str) -> MaintIELLMGNNHybrid:
    """Load trained model from checkpoint"""
    logging.info(f"Loading trained model from {model_path}...")
    return load_model(model_path)

def setup_evaluation_data(config: Dict) -> Data:
    """Load the main graph for evaluation."""
    logging.info("Loading main graph for evaluation...")
    graph_path = config['paths']['graphs']
    try:
        graph = SimpleGraphBuilder(config).load_graph(graph_path)
        return graph
    except FileNotFoundError:
        logging.error(f"Graph file not found at {graph_path}. Cannot perform evaluation.")
        return Data()

def main():
    """Main evaluation script"""
    logging.info("--- Starting Evaluation Pipeline ---")

    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded.")

        # 1. Load Trained Model
        model_path = "results/models/final_model.pt"
        model = load_trained_model(model_path)

        # 2. Load Evaluation Data (the full graph)
        graph_data = setup_evaluation_data(config)

        if not graph_data.num_nodes:
            logging.error("Halting evaluation due to empty graph.")
            return

        # 3. Run Evaluation
        logging.info("Running evaluation...")
        evaluator = GoldStandardEvaluator(config)
        results = evaluator.evaluate(model, graph_data)

        # 4. Display and Save Results
        report = evaluator.generate_report(results)
        print(report)

        results_path = "results/evaluation/evaluation_results.json"
        evaluator.save_results(results, results_path)
        logging.info(f"Evaluation report saved to {results_path}")

    except FileNotFoundError:
        logging.error(f"Configuration file or model file not found. Ensure 'config/config.yaml' and '{model_path}' exist.")
    except Exception as e:
        logging.error(f"An error occurred during the evaluation pipeline: {e}", exc_info=True)

if __name__ == "__main__":
    main()
