from typing import List, Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_mean, scatter_max
import numpy as np
import logging
from collections import OrderedDict
import json

class MaintenanceGNN(nn.Module):
    """
    Advanced Graph Neural Network for maintenance information extraction.
    
    Combines graph structure learning with multi-task prediction for entity
    recognition and relation extraction in maintenance texts.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_entity_classes: int = 224,
                 num_relation_classes: int = 6,
                 num_gnn_layers: int = 2,
                 gnn_type: str = "GAT",
                 num_attention_heads: int = 8,
                 dropout: float = 0.2,
                 use_domain_constraints: bool = True,
                 complexity_level: str = "FG-3"):
        """
        Initialize the MaintenanceGNN model.
        
        Args:
            input_dim: Input feature dimension (LLM embeddings + domain features)
            hidden_dim: Hidden layer dimension
            num_entity_classes: Number of entity types to predict
            num_relation_classes: Number of relation types to predict
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ("GAT", "GCN", "GraphSAGE")
            num_attention_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
            use_domain_constraints: Whether to apply domain constraints
            complexity_level: MaintIE complexity level (FG-0, FG-1, FG-2, FG-3)
        """
        super(MaintenanceGNN, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_entity_classes = num_entity_classes
        self.num_relation_classes = num_relation_classes
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.use_domain_constraints = use_domain_constraints
        self.complexity_level = complexity_level
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Graph neural network layers
        self.gnn_layers = self._build_gnn_layers(gnn_type, hidden_dim, num_gnn_layers, num_attention_heads)
        
        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Entity classification head
        self.entity_classifier = self._build_entity_classifier(hidden_dim, num_entity_classes)
        
        # Relation classification head
        self.relation_classifier = self._build_relation_classifier(hidden_dim, num_relation_classes)
        
        # Graph-level pooling for global features
        self.global_pool = nn.ModuleDict({
            'mean': lambda x, batch: global_mean_pool(x, batch),
            'max': lambda x, batch: global_max_pool(x, batch)
        })
        
        # Domain constraint validator
        if use_domain_constraints:
            self.domain_validator = MaintenanceDomainValidator(
                num_entity_classes, num_relation_classes, complexity_level
            )
        
        # Attention mechanism for relation prediction
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(f"Initialized MaintenanceGNN with {gnn_type} architecture")
    
    def _build_gnn_layers(self, gnn_type: str, hidden_dim: int, num_layers: int, num_heads: int) -> nn.ModuleList:
        """Build GNN layers based on specified type."""
        layers = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == "GAT":
                # Graph Attention Network
                in_channels = hidden_dim
                out_channels = hidden_dim // num_heads if i < num_layers - 1 else hidden_dim
                
                layer = GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=self.dropout,
                    add_self_loops=True,
                    bias=True
                )
                
            elif gnn_type == "GCN":
                # Graph Convolutional Network
                layer = GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    add_self_loops=True,
                    bias=True
                )
                
            elif gnn_type == "GraphSAGE":
                # GraphSAGE
                layer = GraphSAGE(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_layers=1,
                    aggr='mean'
                )
                
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
            layers.append(layer)
        
        return layers
    
    def _build_entity_classifier(self, hidden_dim: int, num_classes: int) -> nn.Module:
        """Build entity classification head."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def _build_relation_classifier(self, hidden_dim: int, num_classes: int) -> nn.Module:
        """Build relation classification head."""
        # Relation classifier takes concatenated node pairs
        input_dim = hidden_dim * 2
        
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the MaintenanceGNN.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, edge_attr_dim]
            batch: Batch indices for multiple graphs [N]
        
        Returns:
            Dictionary containing entity and relation predictions
        """
        # Input projection and normalization
        h = self.input_projection(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Store initial features for residual connections
        initial_h = h
        
        # Graph neural network processing
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Apply GNN layer
            if self.gnn_type == "GAT":
                h_new = gnn_layer(h, edge_index)
            elif self.gnn_type == "GCN":
                h_new = gnn_layer(h, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
            elif self.gnn_type == "GraphSAGE":
                h_new = gnn_layer(h, edge_index)
            
            # Apply normalization and activation
            h_new = layer_norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Residual connection (skip connection)
            if h.size() == h_new.size():
                h = h + h_new
            else:
                h = h_new
        
        # Final node representations
        node_embeddings = h
        
        # Entity prediction
        entity_logits = self.entity_classifier(node_embeddings)
        
        # Relation prediction
        relation_logits = self.predict_relations(node_embeddings, edge_index, batch)
        
        # Apply domain constraints if enabled
        outputs = {
            'entity_logits': entity_logits,
            'relation_logits': relation_logits,
            'node_embeddings': node_embeddings
        }
        
        if self.use_domain_constraints:
            outputs = self.domain_validator.validate_predictions(outputs)
        
        return outputs
    
    def predict_relations(self, 
                         node_embeddings: torch.Tensor, 
                         edge_index: torch.Tensor,
                         batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict relations between entity pairs.
        
        Args:
            node_embeddings: Node representations [N, hidden_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch indices [N]
        
        Returns:
            Relation logits [E, num_relation_classes]
        """
        # Extract source and target node embeddings
        src_embeddings = node_embeddings[edge_index[0]]  # [E, hidden_dim]
        dst_embeddings = node_embeddings[edge_index[1]]  # [E, hidden_dim]
        
        # Concatenate node pair representations
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=1)  # [E, hidden_dim * 2]
        
        # Apply attention mechanism for relation prediction
        edge_embeddings_expanded = edge_embeddings.unsqueeze(0)  # [1, E, hidden_dim * 2]
        attended_embeddings, attention_weights = self.relation_attention(
            edge_embeddings_expanded, edge_embeddings_expanded, edge_embeddings_expanded
        )
        attended_embeddings = attended_embeddings.squeeze(0)  # [E, hidden_dim * 2]
        
        # Predict relations
        relation_logits = self.relation_classifier(attended_embeddings)
        
        return relation_logits
    
    def get_node_embeddings(self, 
                           x: torch.Tensor, 
                           edge_index: torch.Tensor,
                           edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get node embeddings without classification heads."""
        # Forward pass up to node embeddings
        h = self.input_projection(x)
        h = self.input_norm(h)
        h = F.relu(h)
        
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            if self.gnn_type == "GAT":
                h = gnn_layer(h, edge_index)
            elif self.gnn_type == "GCN":
                h = gnn_layer(h, edge_index, edge_weight=edge_attr.squeeze() if edge_attr is not None else None)
            elif self.gnn_type == "GraphSAGE":
                h = gnn_layer(h, edge_index)
            
            h = layer_norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def compute_attention_weights(self, 
                                 x: torch.Tensor, 
                                 edge_index: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for interpretability."""
        if self.gnn_type != "GAT":
            raise ValueError("Attention weights only available for GAT models")
        
        # Get attention weights from first GAT layer
        gat_layer = self.gnn_layers[0]
        h = self.input_projection(x)
        h = F.relu(h)
        
        # Forward pass through GAT layer to get attention weights
        _, attention_weights = gat_layer(h, edge_index, return_attention_weights=True)
        
        return attention_weights


class MaintenanceDomainValidator:
    """
    Domain constraint validator for maintenance information extraction.
    
    Applies engineering domain knowledge to validate and refine model predictions.
    """
    
    def __init__(self, 
                 num_entity_classes: int, 
                 num_relation_classes: int,
                 complexity_level: str = "FG-3"):
        """
        Initialize domain validator.
        
        Args:
            num_entity_classes: Number of entity classes
            num_relation_classes: Number of relation classes  
            complexity_level: MaintIE complexity level
        """
        self.num_entity_classes = num_entity_classes
        self.num_relation_classes = num_relation_classes
        self.complexity_level = complexity_level
        
        # Load domain constraint rules
        self.entity_constraints = self._load_entity_constraints()
        self.relation_constraints = self._load_relation_constraints()
        
        # Entity type compatibility matrix
        self.entity_compatibility = self._build_entity_compatibility_matrix()
        
        # Relation type rules
        self.relation_rules = self._build_relation_rules()
    
    def _load_entity_constraints(self) -> Dict[str, Any]:
        """Load entity constraint rules."""
        return {
            'physical_object_indicators': [
                'pump', 'motor', 'valve', 'sensor', 'bearing', 'belt',
                'compressor', 'turbine', 'generator', 'transformer'
            ],
            'activity_indicators': [
                'replace', 'repair', 'inspect', 'clean', 'service', 'test',
                'install', 'remove', 'adjust', 'calibrate', 'maintain'
            ],
            'state_indicators': [
                'failed', 'normal', 'operational', 'broken', 'damaged',
                'working', 'faulty', 'overheated', 'worn', 'corroded'
            ],
            'property_indicators': [
                'pressure', 'temperature', 'vibration', 'flow', 'level',
                'speed', 'current', 'voltage', 'frequency', 'torque'
            ]
        }
    
    def _load_relation_constraints(self) -> Dict[str, Any]:
        """Load relation constraint rules."""
        return {
            'valid_subject_object_pairs': {
                'Activity-PhysicalObject': ['replace', 'repair', 'inspect'],
                'PhysicalObject-Property': ['has', 'measures', 'controls'],
                'PhysicalObject-State': ['is_in', 'exhibits', 'shows'],
                'Activity-State': ['causes', 'results_in', 'prevents']
            },
            'forbidden_combinations': [
                ('Property', 'Activity'),  # Properties don't perform activities
                ('State', 'PhysicalObject'),  # States don't contain objects
            ]
        }
    
    def _build_entity_compatibility_matrix(self) -> torch.Tensor:
        """Build entity type compatibility matrix."""
        # Simplified compatibility matrix (would be loaded from domain knowledge)
        matrix = torch.ones(self.num_entity_classes, self.num_entity_classes)
        
        # Add some domain-specific constraints
        # This is simplified - in practice, would be based on ontology
        
        return matrix
    
    def _build_relation_rules(self) -> Dict[str, List[str]]:
        """Build relation validation rules."""
        return {
            'requires_physical_object': ['located_at', 'part_of', 'connected_to'],
            'requires_activity': ['performed_by', 'scheduled_for', 'results_from'],
            'requires_property': ['measured_by', 'controlled_by', 'affects'],
            'mutual_exclusive': [
                ['normal', 'failed'],
                ['operational', 'broken']
            ]
        }
    
    def validate_predictions(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply domain constraints to model predictions.
        
        Args:
            outputs: Model outputs dictionary
        
        Returns:
            Validated outputs with domain constraints applied
        """
        entity_logits = outputs['entity_logits']
        relation_logits = outputs['relation_logits']
        
        # Apply entity constraints
        constrained_entity_logits = self._apply_entity_constraints(entity_logits)
        
        # Apply relation constraints
        constrained_relation_logits = self._apply_relation_constraints(
            relation_logits, constrained_entity_logits
        )
        
        # Update outputs
        outputs['entity_logits'] = constrained_entity_logits
        outputs['relation_logits'] = constrained_relation_logits
        outputs['constraint_applied'] = True
        
        return outputs
    
    def _apply_entity_constraints(self, entity_logits: torch.Tensor) -> torch.Tensor:
        """Apply entity-level domain constraints."""
        # Apply softmax to get probabilities
        entity_probs = F.softmax(entity_logits, dim=-1)
        
        # Apply domain-specific adjustments
        # For example, boost certain entity types based on context
        
        # Convert back to logits
        constrained_logits = torch.log(entity_probs + 1e-8)
        
        return constrained_logits
    
    def _apply_relation_constraints(self, 
                                  relation_logits: torch.Tensor,
                                  entity_logits: torch.Tensor) -> torch.Tensor:
        """Apply relation-level domain constraints."""
        # Get entity predictions
        entity_preds = torch.argmax(entity_logits, dim=-1)
        
        # Apply relation constraints based on entity types
        # This is simplified - would use actual entity-relation compatibility
        
        return relation_logits


class MaintenanceGNNLoss(nn.Module):
    """
    Multi-task loss function for MaintenanceGNN.
    
    Combines entity classification loss and relation classification loss
    with optional weighting and regularization terms.
    """
    
    def __init__(self, 
                 entity_weight: float = 1.0,
                 relation_weight: float = 1.0,
                 use_focal_loss: bool = False,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Initialize multi-task loss.
        
        Args:
            entity_weight: Weight for entity classification loss
            relation_weight: Weight for relation classification loss
            use_focal_loss: Whether to use focal loss for handling class imbalance
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super(MaintenanceGNNLoss, self).__init__()
        
        self.entity_weight = entity_weight
        self.relation_weight = relation_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        if use_focal_loss:
            self.entity_loss_fn = self._focal_loss
            self.relation_loss_fn = self._focal_loss
        else:
            self.entity_loss_fn = nn.CrossEntropyLoss()
            self.relation_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                entity_targets: torch.Tensor,
                relation_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs
            entity_targets: Entity classification targets [N]
            relation_targets: Relation classification targets [E]
        
        Returns:
            Dictionary containing individual and total losses
        """
        entity_logits = outputs['entity_logits']
        relation_logits = outputs['relation_logits']
        
        # Entity classification loss
        entity_loss = self.entity_loss_fn(entity_logits, entity_targets)
        
        # Relation classification loss
        relation_loss = self.relation_loss_fn(relation_logits, relation_targets)
        
        # Total weighted loss
        total_loss = (self.entity_weight * entity_loss + 
                     self.relation_weight * relation_loss)
        
        return {
            'total_loss': total_loss,
            'entity_loss': entity_loss,
            'relation_loss': relation_loss
        }
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


# Example usage and testing
if __name__ == "__main__":
    # Model configuration
    config = {
        'input_dim': 416,  # 384 (LLM) + 32 (domain features)
        'hidden_dim': 256,
        'num_entity_classes': 224,  # FG-3 complexity
        'num_relation_classes': 6,
        'gnn_type': 'GAT',
        'num_attention_heads': 8,
        'dropout': 0.2
    }
    
    # Initialize model
    model = MaintenanceGNN(**config)
    
    # Sample data
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn(num_nodes, config['input_dim'])
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 1)
    
    # Forward pass
    outputs = model(x, edge_index, edge_attr)
    
    print("Model Outputs:")
    print(f"Entity logits shape: {outputs['entity_logits'].shape}")
    print(f"Relation logits shape: {outputs['relation_logits'].shape}")
    print(f"Node embeddings shape: {outputs['node_embeddings'].shape}")
    
    # Test loss computation
    loss_fn = MaintenanceGNNLoss()
    entity_targets = torch.randint(0, config['num_entity_classes'], (num_nodes,))
    relation_targets = torch.randint(0, config['num_relation_classes'], (num_edges,))
    
    losses = loss_fn(outputs, entity_targets, relation_targets)
    print(f"\nLoss Values:")
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    print(f"Entity loss: {losses['entity_loss'].item():.4f}")
    print(f"Relation loss: {losses['relation_loss'].item():.4f}")
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
