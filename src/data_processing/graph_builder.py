from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, remove_self_loops
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import json
import re
import logging
from collections import defaultdict, Counter
from pathlib import Path
import pickle

class MaintenanceGraphBuilder:
    """
    Advanced graph builder for maintenance text analysis.
    
    Creates multi-type edge graphs that combine semantic similarity from LLM
    embeddings with domain-specific maintenance knowledge and equipment hierarchies.
    """
    
    def __init__(self, config: Dict[str, Any], ontology_path: Optional[str] = None):
        """
        Initialize the maintenance graph builder.
        
        Args:
            config: Graph construction configuration
            ontology_path: Path to MaintIE ontology file (scheme.json)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config)
        
        # Load maintenance ontology and equipment hierarchy
        self.ontology = self._load_ontology(ontology_path) if ontology_path else None
        self.equipment_hierarchy = self._build_equipment_hierarchy()
        
        # Initialize edge type configurations
        self.edge_types = self.config.get('edge_types', [
            'semantic_similarity',
            'entity_cooccurrence', 
            'equipment_hierarchy',
            'procedure_similarity'
        ])
        
        # Setup similarity and connectivity parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.75)
        self.k_neighbors = self.config.get('k_neighbors', 15)
        self.max_edges_per_node = self.config.get('max_edges_per_node', 50)
        
        # Initialize caching for performance
        self.graph_cache = {}
        self.similarity_cache = {}
        
        # Initialize pattern matchers
        self.equipment_patterns = self._compile_equipment_patterns()
        self.entity_extractors = self._setup_entity_extractors()
        
        self.logger.info(f"Initialized MaintenanceGraphBuilder with {len(self.edge_types)} edge types")
    
    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate graph construction configuration."""
        default_config = {
            'similarity_threshold': 0.75,
            'k_neighbors': 15,
            'max_edges_per_node': 50,
            'edge_weights': {
                'semantic_similarity': 1.0,
                'entity_cooccurrence': 0.8,
                'equipment_hierarchy': 0.9,
                'procedure_similarity': 0.7
            },
            'graph_properties': {
                'add_self_loops': True,
                'undirected': True,
                'remove_duplicates': True
            },
            'performance': {
                'use_fast_similarity': True,
                'cache_similarities': True,
                'batch_processing': True
            }
        }
        
        # Update with user configuration
        if config:
            default_config.update(config)
        
        return default_config
    
    def _load_ontology(self, ontology_path: str) -> Dict[str, Any]:
        """Load MaintIE ontology schema."""
        try:
            with open(ontology_path, 'r', encoding='utf-8') as f:
                ontology = json.load(f)
            self.logger.info(f"Loaded ontology with {len(ontology.get('entities', {}))} entity types")
            return ontology
        except Exception as e:
            self.logger.warning(f"Could not load ontology: {e}")
            return {}
    
    def _build_equipment_hierarchy(self) -> Dict[str, List[str]]:
        """Build equipment hierarchy from ontology."""
        hierarchy = defaultdict(list)
        
        if not self.ontology:
            # Fallback hierarchy if ontology not available
            return {
                'PhysicalObject': ['SensingObject', 'DrivingObject', 'ProcessingObject'],
                'SensingObject': ['PressureSensor', 'TemperatureSensor', 'FlowSensor'],
                'DrivingObject': ['Motor', 'Pump', 'Compressor'],
                'ProcessingObject': ['Valve', 'Filter', 'HeatExchanger']
            }
        
        # Extract hierarchy from ontology
        entities = self.ontology.get('entities', {})
        for entity_name, entity_info in entities.items():
            if 'parent' in entity_info:
                parent = entity_info['parent']
                hierarchy[parent].append(entity_name)
        
        return dict(hierarchy)
    
    def _compile_equipment_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for equipment identification."""
        patterns = [
            r'\b\w*pump\w*\b',
            r'\b\w*motor\w*\b',
            r'\b\w*valve\w*\b', 
            r'\b\w*sensor\w*\b',
            r'\b\w*bearing\w*\b',
            r'\b\w*compressor\w*\b',
            r'\b\w*turbine\w*\b',
            r'\b\w*generator\w*\b',
            r'\b[A-Z]{2,4}-?\d{1,4}\b',  # Equipment codes
            r'\b\w+\s*#\s*\d+\b',        # Equipment numbers
            r'\b\w+\s*unit\s*\d+\b'      # Unit numbers
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _setup_entity_extractors(self) -> Dict[str, Any]:
        """Setup entity extraction utilities."""
        return {
            'equipment_terms': [
                'pump', 'motor', 'valve', 'sensor', 'bearing', 'compressor',
                'turbine', 'generator', 'transformer', 'switch', 'relay'
            ],
            'failure_terms': [
                'leak', 'crack', 'wear', 'vibration', 'overheat', 'fail',
                'fault', 'alarm', 'error', 'malfunction'
            ],
            'action_terms': [
                'replace', 'repair', 'inspect', 'clean', 'service', 'test',
                'install', 'remove', 'adjust', 'calibrate'
            ]
        }
    
    def build_maintenance_graph(self, 
                              node_features: np.ndarray,
                              texts: List[str], 
                              annotations: Optional[List[Dict]] = None,
                              node_ids: Optional[List[str]] = None) -> Data:
        """
        Build comprehensive maintenance graph with multiple edge types.
        
        Args:
            node_features: LLM embeddings + domain features [N, feature_dim]
            texts: Original text documents
            annotations: Optional ground truth annotations
            node_ids: Optional node identifiers
        
        Returns:
            PyTorch Geometric Data object
        """
        self.logger.info(f"Building graph for {len(texts)} nodes")
        
        num_nodes = len(texts)
        node_ids = node_ids or list(range(num_nodes))
        
        # Initialize edge collections
        all_edges = []
        edge_weights = []
        edge_types = []
        
        # Build different types of edges
        if 'semantic_similarity' in self.edge_types:
            sem_edges, sem_weights = self.build_semantic_edges(node_features)
            all_edges.extend(sem_edges)
            edge_weights.extend(sem_weights)
            edge_types.extend(['semantic'] * len(sem_edges))
        
        if 'entity_cooccurrence' in self.edge_types:
            ent_edges, ent_weights = self.build_entity_cooccurrence_edges(texts, annotations)
            all_edges.extend(ent_edges)
            edge_weights.extend(ent_weights)
            edge_types.extend(['entity'] * len(ent_edges))
        
        if 'equipment_hierarchy' in self.edge_types:
            eq_edges, eq_weights = self.build_equipment_hierarchy_edges(texts)
            all_edges.extend(eq_edges)
            edge_weights.extend(eq_weights)
            edge_types.extend(['equipment'] * len(eq_edges))
        
        if 'procedure_similarity' in self.edge_types:
            proc_edges, proc_weights = self.build_procedure_similarity_edges(texts)
            all_edges.extend(proc_edges)
            edge_weights.extend(proc_weights)
            edge_types.extend(['procedure'] * len(proc_edges))
        
        # Combine and process edges
        edge_index, edge_attr = self._process_edges(
            all_edges, edge_weights, edge_types, num_nodes
        )
        
        # Create PyTorch Geometric Data object
        graph_data = self._create_pytorch_geometric_data(
            node_features, edge_index, edge_attr, texts, annotations, node_ids
        )
        
        # Apply post-processing
        graph_data = self._postprocess_graph(graph_data)
        
        self.logger.info(f"Built graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
        return graph_data
    
    def build_semantic_edges(self, embeddings: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """
        Build edges based on semantic similarity of embeddings.
        
        Args:
            embeddings: Node feature matrix [N, feature_dim]
        
        Returns:
            Tuple of (edge_list, edge_weights)
        """
        # Check cache
        cache_key = f"semantic_{hash(embeddings.tobytes())}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        edges = []
        weights = []
        
        if self.config['performance']['use_fast_similarity']:
            # Use k-nearest neighbors for efficiency
            edges, weights = self._build_knn_similarity_edges(embeddings)
        else:
            # Full pairwise similarity computation
            edges, weights = self._build_full_similarity_edges(embeddings)
        
        # Cache results
        if self.config['performance']['cache_similarities']:
            self.similarity_cache[cache_key] = (edges, weights)
        
        self.logger.debug(f"Built {len(edges)} semantic similarity edges")
        return edges, weights
    
    def _build_knn_similarity_edges(self, embeddings: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """Build edges using k-nearest neighbors for efficiency."""
        # Initialize k-NN model
        knn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, len(embeddings)),
            metric='cosine'
        )
        knn.fit(embeddings)
        
        edges = []
        weights = []
        
        # Find neighbors for each node
        distances, indices = knn.kneighbors(embeddings)
        
        for i, (node_distances, node_indices) in enumerate(zip(distances, indices)):
            for j, (distance, neighbor_idx) in enumerate(zip(node_distances, node_indices)):
                if i != neighbor_idx and j > 0:  # Skip self
                    similarity = 1.0 - distance  # Convert distance to similarity
                    
                    if similarity >= self.similarity_threshold:
                        edges.append([i, neighbor_idx])
                        weights.append(similarity)
        
        return edges, weights
    
    def _build_full_similarity_edges(self, embeddings: np.ndarray) -> Tuple[List[List[int]], List[float]]:
        """Build edges using full pairwise similarity computation."""
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        edges = []
        weights = []
        
        # Extract edges above threshold
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                
                if similarity >= self.similarity_threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    weights.extend([similarity, similarity])
        
        return edges, weights
    
    def build_entity_cooccurrence_edges(self, 
                                      texts: List[str], 
                                      annotations: Optional[List[Dict]] = None) -> Tuple[List[List[int]], List[float]]:
        """
        Build edges based on entity co-occurrence patterns.
        
        Args:
            texts: Text documents
            annotations: Optional entity annotations
        
        Returns:
            Tuple of (edge_list, edge_weights)
        """
        edges = []
        weights = []
        
        # Extract entities from each text
        text_entities = []
        for text in texts:
            entities = self._extract_entities_from_text(text)
            text_entities.append(entities)
        
        # Build co-occurrence matrix
        cooccurrence_counts = defaultdict(int)
        
        for i, entities_i in enumerate(text_entities):
            for j, entities_j in enumerate(text_entities):
                if i != j:
                    # Count shared entities
                    shared_entities = len(set(entities_i) & set(entities_j))
                    total_entities = len(set(entities_i) | set(entities_j))
                    
                    if shared_entities > 0 and total_entities > 0:
                        # Jaccard similarity for entity overlap
                        jaccard_sim = shared_entities / total_entities
                        
                        if jaccard_sim >= 0.3:  # Threshold for entity similarity
                            edges.append([i, j])
                            weights.append(jaccard_sim)
        
        self.logger.debug(f"Built {len(edges)} entity co-occurrence edges")
        return edges, weights
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text using pattern matching."""
        entities = []
        
        # Extract equipment entities
        for pattern in self.equipment_patterns:
            matches = pattern.findall(text.lower())
            entities.extend(matches)
        
        # Extract term-based entities
        for category, terms in self.entity_extractors.items():
            for term in terms:
                if re.search(rf'\b{term}\w*\b', text, re.IGNORECASE):
                    entities.append(f"{category}:{term}")
        
        return list(set(entities))  # Remove duplicates
    
    def build_equipment_hierarchy_edges(self, texts: List[str]) -> Tuple[List[List[int]], List[float]]:
        """
        Build edges based on equipment hierarchy relationships.
        
        Args:
            texts: Text documents
        
        Returns:
            Tuple of (edge_list, edge_weights)
        """
        edges = []
        weights = []
        
        # Extract equipment mentions from each text
        text_equipment = []
        for text in texts:
            equipment = self._extract_equipment_from_text(text)
            text_equipment.append(equipment)
        
        # Build hierarchy-based edges
        for i, equipment_i in enumerate(text_equipment):
            for j, equipment_j in enumerate(text_equipment):
                if i != j:
                    # Check for hierarchical relationships
                    hierarchy_strength = self._calculate_hierarchy_strength(equipment_i, equipment_j)
                    
                    if hierarchy_strength > 0.5:
                        edges.append([i, j])
                        weights.append(hierarchy_strength)
        
        self.logger.debug(f"Built {len(edges)} equipment hierarchy edges")
        return edges, weights
    
    def _extract_equipment_from_text(self, text: str) -> List[str]:
        """Extract equipment mentions from text."""
        equipment = []
        
        # Use equipment patterns
        for pattern in self.equipment_patterns:
            matches = pattern.findall(text.lower())
            equipment.extend(matches)
        
        # Use equipment terms
        for term in self.entity_extractors['equipment_terms']:
            if re.search(rf'\b{term}\w*\b', text, re.IGNORECASE):
                equipment.append(term)
        
        return list(set(equipment))
    
    def _calculate_hierarchy_strength(self, equipment_i: List[str], equipment_j: List[str]) -> float:
        """Calculate strength of hierarchical relationship between equipment sets."""
        if not equipment_i or not equipment_j:
            return 0.0
        
        hierarchy_score = 0.0
        total_comparisons = 0
        
        for eq_i in equipment_i:
            for eq_j in equipment_j:
                # Check if equipment types are in same hierarchy branch
                if self._are_in_same_hierarchy(eq_i, eq_j):
                    hierarchy_score += 1.0
                total_comparisons += 1
        
        return hierarchy_score / total_comparisons if total_comparisons > 0 else 0.0
    
    def _are_in_same_hierarchy(self, eq1: str, eq2: str) -> bool:
        """Check if two equipment types are in the same hierarchy branch."""
        # Simplified hierarchy checking
        equipment_categories = {
            'pump': 'DrivingObject',
            'motor': 'DrivingObject', 
            'compressor': 'DrivingObject',
            'sensor': 'SensingObject',
            'valve': 'ProcessingObject',
            'filter': 'ProcessingObject'
        }
        
        category1 = equipment_categories.get(eq1.lower())
        category2 = equipment_categories.get(eq2.lower())
        
        return category1 is not None and category1 == category2
    
    def build_procedure_similarity_edges(self, texts: List[str]) -> Tuple[List[List[int]], List[float]]:
        """
        Build edges based on maintenance procedure similarity.
        
        Args:
            texts: Text documents
        
        Returns:
            Tuple of (edge_list, edge_weights)
        """
        edges = []
        weights = []
        
        # Extract maintenance actions from each text
        text_actions = []
        for text in texts:
            actions = self._extract_maintenance_actions(text)
            text_actions.append(actions)
        
        # Build procedure similarity edges
        for i, actions_i in enumerate(text_actions):
            for j, actions_j in enumerate(text_actions):
                if i != j:
                    # Calculate action similarity
                    action_similarity = self._calculate_action_similarity(actions_i, actions_j)
                    
                    if action_similarity >= 0.4:  # Threshold for procedure similarity
                        edges.append([i, j])
                        weights.append(action_similarity)
        
        self.logger.debug(f"Built {len(edges)} procedure similarity edges")
        return edges, weights
    
    def _extract_maintenance_actions(self, text: str) -> List[str]:
        """Extract maintenance actions from text."""
        actions = []
        
        for action_term in self.entity_extractors['action_terms']:
            if re.search(rf'\b{action_term}\w*\b', text, re.IGNORECASE):
                actions.append(action_term)
        
        return list(set(actions))
    
    def _calculate_action_similarity(self, actions_i: List[str], actions_j: List[str]) -> float:
        """Calculate similarity between action sets."""
        if not actions_i or not actions_j:
            return 0.0
        
        # Jaccard similarity
        intersection = len(set(actions_i) & set(actions_j))
        union = len(set(actions_i) | set(actions_j))
        
        return intersection / union if union > 0 else 0.0
    
    def _process_edges(self, 
                      all_edges: List[List[int]], 
                      edge_weights: List[float],
                      edge_types: List[str], 
                      num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process and combine all edge types."""
        if not all_edges:
            # Create empty edge tensors
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            return edge_index, edge_attr
        
        # Convert to tensors
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)
        
        # Remove duplicate edges and combine weights
        edge_index, edge_weights_tensor = self._deduplicate_edges(edge_index, edge_weights_tensor)
        
        # Limit edges per node if specified
        if self.max_edges_per_node:
            edge_index, edge_weights_tensor = self._limit_edges_per_node(
                edge_index, edge_weights_tensor, num_nodes
            )
        
        # Add self loops if configured
        if self.config['graph_properties']['add_self_loops']:
            edge_index, edge_weights_tensor = add_self_loops(
                edge_index, edge_weights_tensor, num_nodes=num_nodes
            )
        
        # Reshape edge attributes
        edge_attr = edge_weights_tensor.unsqueeze(1)
        
        return edge_index, edge_attr
    
    def _deduplicate_edges(self, 
                          edge_index: torch.Tensor, 
                          edge_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove duplicate edges and combine their weights."""
        # Create edge dictionary for deduplication
        edge_dict = defaultdict(list)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            edge_key = (src.item(), dst.item())
            edge_dict[edge_key].append(edge_weights[i].item())
        
        # Reconstruct deduplicated edges
        new_edges = []
        new_weights = []
        
        for (src, dst), weights in edge_dict.items():
            new_edges.append([src, dst])
            # Combine weights (average)
            new_weights.append(sum(weights) / len(weights))
        
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        new_edge_weights = torch.tensor(new_weights, dtype=torch.float)
        
        return new_edge_index, new_edge_weights
    
    def _limit_edges_per_node(self, 
                             edge_index: torch.Tensor, 
                             edge_weights: torch.Tensor,
                             num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Limit the number of edges per node to prevent over-connectivity."""
        # Group edges by source node
        node_edges = defaultdict(list)
        
        for i, (src, dst) in enumerate(edge_index.t()):
            node_edges[src.item()].append((i, edge_weights[i].item()))
        
        # Keep only top edges per node
        keep_indices = []
        
        for node_id, edges in node_edges.items():
            # Sort by weight (descending)
            edges.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top edges
            for i, (edge_idx, weight) in enumerate(edges[:self.max_edges_per_node]):
                keep_indices.append(edge_idx)
        
        # Filter edges
        keep_indices = sorted(keep_indices)
        filtered_edge_index = edge_index[:, keep_indices]
        filtered_edge_weights = edge_weights[keep_indices]
        
        return filtered_edge_index, filtered_edge_weights
    
    def _create_pytorch_geometric_data(self, 
                                     node_features: np.ndarray,
                                     edge_index: torch.Tensor,
                                     edge_attr: torch.Tensor,
                                     texts: List[str],
                                     annotations: Optional[List[Dict]],
                                     node_ids: List[str]) -> Data:
        """Create PyTorch Geometric Data object."""
        # Convert node features to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create labels if annotations provided
        y = None
        if annotations:
            y = self._create_label_tensors(annotations)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(texts)
        )
        
        # Add metadata
        data.texts = texts
        data.node_ids = node_ids
        data.annotations = annotations
        
        return data
    
    def _create_label_tensors(self, annotations: List[Dict]) -> torch.Tensor:
        """Create label tensors from annotations."""
        # This is a simplified version - adapt based on your annotation format
        labels = []
        for annotation in annotations:
            # Extract entity labels (simplified)
            entity_label = annotation.get('entities', [])
            if entity_label:
                labels.append(1)  # Has entities
            else:
                labels.append(0)  # No entities
        
        return torch.tensor(labels, dtype=torch.long)
    
    def _postprocess_graph(self, data: Data) -> Data:
        """Apply post-processing to the graph."""
        # Normalize edge weights
        if data.edge_attr is not None and len(data.edge_attr) > 0:
            edge_weights = data.edge_attr.squeeze()
            normalized_weights = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-8)
            data.edge_attr = normalized_weights.unsqueeze(1)
        
        # Add graph statistics
        data.graph_stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'avg_degree': data.num_edges / data.num_nodes if data.num_nodes > 0 else 0,
            'edge_types': self.edge_types
        }
        
        return data
    
    def save_graph(self, data: Data, filepath: str) -> None:
        """Save graph data to disk."""
        torch.save(data, filepath)
        self.logger.info(f"Saved graph to {filepath}")
    
    def load_graph(self, filepath: str) -> Data:
        """Load graph data from disk."""
        data = torch.load(filepath)
        self.logger.info(f"Loaded graph from {filepath}")
        return data
    
    def analyze_graph_properties(self, data: Data) -> Dict[str, Any]:
        """Analyze graph properties and connectivity."""
        edge_index = data.edge_index.numpy()
        
        # Convert to NetworkX for analysis
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(edge_index.T)
        
        properties = {
            'num_nodes': data.num_nodes,
            'num_edges': data.num_edges,
            'avg_degree': sum(dict(G.degree()).values()) / data.num_nodes,
            'density': nx.density(G),
            'num_connected_components': nx.number_connected_components(G),
            'avg_clustering': nx.average_clustering(G),
            'diameter': nx.diameter(G) if nx.is_connected(G) else 'N/A (disconnected)'
        }
        
        return properties


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = {
        'similarity_threshold': 0.75,
        'k_neighbors': 15,
        'edge_types': ['semantic_similarity', 'entity_cooccurrence', 'equipment_hierarchy'],
        'edge_weights': {
            'semantic_similarity': 1.0,
            'entity_cooccurrence': 0.8,
            'equipment_hierarchy': 0.9
        }
    }
    
    # Initialize graph builder
    graph_builder = MaintenanceGraphBuilder(config)
    
    # Sample data
    sample_embeddings = np.random.rand(10, 384)  # 10 nodes, 384-dim embeddings
    sample_texts = [
        "Replace faulty pressure sensor in cooling system pump #3",
        "Inspect vibration levels in main turbine bearing assembly", 
        "Clean and lubricate conveyor belt drive motor",
        "Emergency repair of hydraulic leak in valve actuator",
        "Test pressure sensor calibration in cooling system",
        "Replace worn bearing in turbine assembly",
        "Service motor drive system components",
        "Repair hydraulic valve actuator mechanism",
        "Inspect cooling system pump operation",
        "Calibrate pressure monitoring sensors"
    ]
    
    # Build graph
    graph_data = graph_builder.build_maintenance_graph(
        node_features=sample_embeddings,
        texts=sample_texts
    )
    
    # Analyze properties
    properties = graph_builder.analyze_graph_properties(graph_data)
    
    print("Graph Properties:")
    for key, value in properties.items():
        print(f"  {key}: {value}")
    
    print(f"\nGraph shape: {graph_data.x.shape}")
    print(f"Edge index shape: {graph_data.edge_index.shape}")
    print(f"Edge attributes shape: {graph_data.edge_attr.shape}")
