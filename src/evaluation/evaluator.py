import json
import logging
from typing import Any, Dict, List, Tuple
import torch
from torch_geometric.data import Data
from src.models.llm_gnn_hybrid import MaintIELLMGNNHybrid
from src.evaluation.metrics import MetricsCalculator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GoldStandardEvaluator:
    """Simple gold standard evaluation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator"""
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("GoldStandardEvaluator initialized.")

    def evaluate(self, model: MaintIELLMGNNHybrid, graph: Data) -> Dict[str, float]:
        """Evaluate model on the validation set using the val_mask."""
        model.eval()
        model.to(self.device)
        graph.to(self.device)

        all_entity_preds, all_entity_targets = [], []
        all_relation_preds, all_relation_targets = [], []

        with torch.no_grad():
            predictions = model(graph)

            # Use the validation mask to get predictions and targets for the gold data
            val_mask = graph.val_mask
            entity_preds, relation_preds = self._extract_predictions(predictions, val_mask)
            entity_targets, relation_targets = self._extract_targets(graph, val_mask)

            all_entity_preds.extend(entity_preds)
            all_entity_targets.extend(entity_targets)
            all_relation_preds.extend(relation_preds)
            all_relation_targets.extend(relation_targets)

        entity_metrics = self.metrics_calculator.calculate_entity_metrics(all_entity_preds, all_entity_targets)
        relation_metrics = self.metrics_calculator.calculate_relation_metrics(all_relation_preds, all_relation_targets)

        results = {
            'entity_f1': entity_metrics['f1'],
            'entity_precision': entity_metrics['precision'],
            'entity_recall': entity_metrics['recall'],
            'relation_f1': relation_metrics['f1'],
            'relation_precision': relation_metrics['precision'],
            'relation_recall': relation_metrics['recall'],
        }

        logging.info(f"Evaluation results: {results}")
        return results

    def _extract_predictions(self, model_output: Dict[str, torch.Tensor], mask: torch.Tensor) -> Tuple[List[int], List[int]]:
        """Extract entity and relation predictions for the masked nodes."""
        # Apply sigmoid to get probabilities and threshold at 0.5 for multi-label classification
        entity_probs = torch.sigmoid(model_output['entity_logits'][mask])
        relation_probs = torch.sigmoid(model_output['relation_logits'][mask])

        entity_preds = (entity_probs > 0.5).int().cpu().tolist()
        relation_preds = (relation_probs > 0.5).int().cpu().tolist()

        return entity_preds, relation_preds

    def _extract_targets(self, graph: Data, mask: torch.Tensor) -> Tuple[List[int], List[int]]:
        """Extract ground truth labels for the masked nodes."""
        entity_targets = graph.y_entity[mask].int().cpu().tolist()
        relation_targets = graph.y_relation[mask].int().cpu().tolist()
        return entity_targets, relation_targets

    def generate_report(self, results: Dict[str, float]) -> str:
        """Generate evaluation report"""
        report = "--- Evaluation Report ---\n"
        for key, value in results.items():
            report += f"{key.replace('_', ' ').title()}: {value:.4f}\n"
        report += "-------------------------"
        return report

    def save_results(self, results: Dict[str, float], output_path: str) -> None:
        """Save evaluation results"""
        logging.info(f"Saving evaluation results to {output_path}...")
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            logging.info("Results saved successfully.")
        except IOError as e:
            logging.error(f"Failed to save results to {output_path}: {e}")

def main():
    """CLI entry point for make evaluate"""
    import yaml
    from src.models.llm_gnn_hybrid import load_model
    from src.data_processing.graph_builder import SimpleGraphBuilder

    logging.info("--- Starting Evaluation Pipeline ---")
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Load trained model
        model_path = "results/models/final_model.pt"
        model = load_model(model_path)

        # Load gold data
        graph_path = config['paths']['graphs']
        graph_data = SimpleGraphBuilder(config).load_graph(graph_path)

        if not graph_data.num_nodes:
            logging.warning("Evaluation graph is empty. Cannot perform evaluation.")
            return

        # Run evaluation
        evaluator = GoldStandardEvaluator(config)
        results = evaluator.evaluate(model, graph_data)

        # Display and save results
        report = evaluator.generate_report(results)
        print(report)

        results_path = f"results/evaluation/evaluation_results.json"
        evaluator.save_results(results, results_path)

    except FileNotFoundError:
        logging.error("Configuration file or model file not found.")
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
