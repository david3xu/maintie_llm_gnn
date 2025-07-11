import logging
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Any, Dict, List, Tuple
from src.training.utils import setup_optimizer
from src.models.llm_gnn_hybrid import MaintIELLMGNNHybrid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleTrainer:
    """Simple training implementation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer"""
        self.config = config['training']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.patience = self.config.get('patience', 5)
        logging.info(f"Trainer initialized on device: {self.device}")

    def train(self, model: MaintIELLMGNNHybrid, graph: Data) -> Dict[str, Any]:
        """Main training loop for a single graph with masks"""
        optimizer = setup_optimizer(model, self.config)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        model.to(self.device)
        graph.to(self.device)

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(model, graph, optimizer)
            val_metrics = self.validate(model, graph)

            val_loss = val_metrics['val_loss']
            logging.info(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                         f"Train Loss: {train_loss:.4f} | "
                         f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Consider saving the best model state
                # torch.save(model.state_dict(), 'best_model.pt')
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        return {'best_val_loss': best_val_loss}

    def train_epoch(self, model: MaintIELLMGNNHybrid, graph: Data, optimizer: torch.optim.Optimizer) -> float:
        """Single training epoch"""
        model.train()
        optimizer.zero_grad()

        predictions = model(graph)
        loss = self.compute_loss(predictions, graph, graph.train_mask)

        loss.backward()
        optimizer.step()

        return loss.item()

    def validate(self, model: MaintIELLMGNNHybrid, graph: Data) -> Dict[str, float]:
        """Validation loop"""
        model.eval()
        with torch.no_grad():
            predictions = model(graph)
            loss = self.compute_loss(predictions, graph, graph.val_mask)

        return {'val_loss': loss.item()}

    def compute_loss(self, predictions: Dict[str, torch.Tensor], graph: Data, mask: torch.Tensor) -> torch.Tensor:
        """Compute training loss for multi-label classification."""
        entity_loss = F.binary_cross_entropy_with_logits(
            predictions['entity_logits'][mask],
            graph.y_entity[mask]
        )

        relation_loss = F.binary_cross_entropy_with_logits(
            predictions['relation_logits'][mask],
            graph.y_relation[mask]
        )

        # You can weight the losses if desired, e.g., return 0.5 * entity_loss + 0.5 * relation_loss
        return entity_loss + relation_loss

    def save_checkpoint(self, model: MaintIELLMGNNHybrid, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, path: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        return torch.load(path)
