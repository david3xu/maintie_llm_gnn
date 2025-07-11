from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv

class SimpleGNN(torch.nn.Module):
    """Simple Graph Neural Network"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
        """Initialize simple GNN"""
        super(SimpleGNN, self).__init__()

        self.convs = ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GNN layers"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get node embeddings without final classification"""
        # In this simple model, the final layer is the output embedding layer.
        # If we wanted intermediate embeddings, we would stop before the last layer.
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
