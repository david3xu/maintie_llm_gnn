from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from src.models.simple_gnn import SimpleGNN
from torch_geometric.data import Data

class MaintIELLMGNNHybrid(torch.nn.Module):
    """Simple LLM+GNN hybrid model"""

    def __init__(self, config: Dict[str, Any], num_entity_classes: int, num_relation_classes: int):
        """Initialize hybrid model"""
        super(MaintIELLMGNNHybrid, self).__init__()
        self.config = config
        self.model_config = config['model']

        # GNN component
        self.gnn = SimpleGNN(
            input_dim=self.model_config['embedding_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            output_dim=self.model_config['hidden_dim'], # GNN outputs hidden features
            num_layers=self.model_config['num_gnn_layers'],
            dropout=self.model_config['dropout']
        )

        # Output heads
        self.num_entity_classes = num_entity_classes
        self.num_relation_classes = num_relation_classes
        self.output_heads = self._create_output_heads()

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid model"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Process node features through GNN
        node_representations = self.gnn(x, edge_index, edge_attr)

        # Apply classification heads
        entity_logits = self.output_heads['entity'](node_representations)
        relation_logits = self.output_heads['relation'](node_representations) # Simplified: should process pairs

        return {
            'entity_logits': entity_logits,
            'relation_logits': relation_logits
        }

    def _create_output_heads(self) -> nn.ModuleDict:
        """Create entity and relation classification heads"""
        hidden_dim = self.model_config['hidden_dim']

        entity_head = nn.Linear(hidden_dim, self.num_entity_classes)

        # Simplified relation head: predicts relations from individual node embeddings
        # A more complex model would consider pairs of node embeddings.
        relation_head = nn.Linear(hidden_dim, self.num_relation_classes)

        return nn.ModuleDict({
            'entity': entity_head,
            'relation': relation_head
        })

    def save_model(self, path: str, optimizer_state: Optional[Dict] = None) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'num_entity_classes': self.num_entity_classes,
            'num_relation_classes': self.num_relation_classes,
            'optimizer_state_dict': optimizer_state
        }
        torch.save(checkpoint, path)

def create_model(config: Dict[str, Any], num_entity_classes: int, num_relation_classes: int) -> MaintIELLMGNNHybrid:
    """Factory function to create model"""
    model = MaintIELLMGNNHybrid(config, num_entity_classes, num_relation_classes)
    return model

def load_model(checkpoint_path: str) -> MaintIELLMGNNHybrid:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    num_entity_classes = checkpoint['num_entity_classes']
    num_relation_classes = checkpoint['num_relation_classes']

    model = MaintIELLMGNNHybrid(config, num_entity_classes, num_relation_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
