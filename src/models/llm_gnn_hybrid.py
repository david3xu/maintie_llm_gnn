from typing import List, Dict, Optional, Tuple, Union, Any, Iterator
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
import logging
import time
from pathlib import Path
from collections import OrderedDict

# Import custom modules
from .embedders.maintenance_llm_embedder import MaintenanceLLMEmbedder
from ..data_processing.graph_builder import MaintenanceGraphBuilder
from .graph_networks.maintenance_gnn import MaintenanceGNN, MaintenanceGNNLoss
from ..evaluation.metrics_calculator import MaintenanceMetricsCalculator
from ..utils.file_handlers import JSONHandler, PickleHandler
from ..core.config import ConfigManager

class MaintIELLMGNNHybrid:
    """
    Main hybrid architecture combining LLM embeddings with GNN processing
    for maintenance information extraction.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager(config)
        self.device = self._setup_device()
        self.embedder = self._initialize_embedder()
        self.graph_builder = self._initialize_graph_builder()
        self.gnn_model = self._initialize_gnn_model()
        self.loss_function = self._initialize_loss_function()
        self.metrics_calculator = MaintenanceMetricsCalculator()
        self.optimizer = None
        self.scheduler = None
        self.training_state = {
            'epoch': 0,
            'best_score': 0.0,
            'best_model_path': None,
            'training_history': []
        }
        self.embedding_cache = {}
        self.graph_cache = {}
        self.logger.info("Initialized MaintIE LLM-GNN Hybrid Model")

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU")
        return device

    def _initialize_embedder(self) -> MaintenanceLLMEmbedder:
        embedder_config = self.config.get_section('model.llm_embedder')
        return MaintenanceLLMEmbedder(
            model_name=embedder_config.get('model_name'),
            feature_config=embedder_config.get('feature_config'),
            cache_embeddings=embedder_config.get('cache_embeddings', True),
            device=str(self.device)
        )

    def _initialize_graph_builder(self) -> MaintenanceGraphBuilder:
        graph_config = self.config.get_section('graph')
        ontology_path = self.config.get('data.ontology_path')
        return MaintenanceGraphBuilder(
            config=graph_config,
            ontology_path=ontology_path
        )

    def _initialize_gnn_model(self) -> MaintenanceGNN:
        model_config = self.config.get_section('model')
        gnn_config = model_config.get('gnn', {})
        model = MaintenanceGNN(
            input_dim=model_config.get('llm_embedder.combined_features_dim', 416),
            hidden_dim=gnn_config.get('hidden_dim', 256),
            num_entity_classes=model_config.get('num_entity_classes', 224),
            num_relation_classes=model_config.get('num_relation_classes', 6),
            num_gnn_layers=gnn_config.get('num_layers', 2),
            gnn_type=gnn_config.get('type', 'GAT'),
            num_attention_heads=gnn_config.get('num_heads', 8),
            dropout=gnn_config.get('dropout', 0.2),
            use_domain_constraints=model_config.get('use_domain_constraints', True)
        )
        return model.to(self.device)

    def _initialize_loss_function(self) -> MaintenanceGNNLoss:
        training_config = self.config.get_section('training')
        return MaintenanceGNNLoss(
            entity_weight=training_config.get('entity_loss_weight', 1.0),
            relation_weight=training_config.get('relation_loss_weight', 1.0),
            use_focal_loss=training_config.get('use_focal_loss', False)
        )

    def extract_maintenance_info(self, texts: List[str], return_embeddings: bool = False, return_graphs: bool = False) -> Dict[str, Any]:
        self.logger.info(f"Processing {len(texts)} maintenance texts")
        start_time = time.time()
        embeddings = self._get_or_compute_embeddings(texts)
        graph_data = self._get_or_compute_graph(texts, embeddings)
        predictions = self._predict_with_gnn(graph_data)
        results = self._format_extraction_results(predictions, texts, graph_data)
        if return_embeddings:
            results['embeddings'] = embeddings
        if return_graphs:
            results['graph_data'] = graph_data
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.logger.info(f"Extraction completed in {processing_time:.2f} seconds")
        return results

    def _get_or_compute_embeddings(self, texts: List[str]) -> np.ndarray:
        cache_key = hash(str(texts))
        if cache_key in self.embedding_cache:
            self.logger.debug("Retrieved embeddings from cache")
            return self.embedding_cache[cache_key]
        embeddings = self.embedder.create_node_features(texts)
        self.embedding_cache[cache_key] = embeddings
        return embeddings

    def _get_or_compute_graph(self, texts: List[str], embeddings: np.ndarray) -> Data:
        cache_key = hash(str(texts) + str(embeddings.tobytes()))
        if cache_key in self.graph_cache:
            self.logger.debug("Retrieved graph from cache")
            return self.graph_cache[cache_key]
        graph_data = self.graph_builder.build_maintenance_graph(
            node_features=embeddings,
            texts=texts
        )
        self.graph_cache[cache_key] = graph_data
        return graph_data

    def _predict_with_gnn(self, graph_data: Data) -> Dict[str, torch.Tensor]:
        self.gnn_model.eval()
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            outputs = self.gnn_model(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr
            )
        return outputs

    def _format_extraction_results(self, predictions: Dict[str, torch.Tensor], texts: List[str], graph_data: Data) -> Dict[str, Any]:
        entity_logits = predictions['entity_logits'].cpu()
        relation_logits = predictions['relation_logits'].cpu()
        entity_predictions = torch.argmax(entity_logits, dim=-1).numpy()
        entity_probabilities = torch.softmax(entity_logits, dim=-1).numpy()
        relation_predictions = torch.argmax(relation_logits, dim=-1).numpy()
        relation_probabilities = torch.softmax(relation_logits, dim=-1).numpy()
        results = {
            'texts': texts,
            'num_texts': len(texts),
            'entities': [],
            'relations': [],
            'graph_statistics': self._compute_graph_statistics(graph_data)
        }
        for i, (text, entity_pred, entity_prob) in enumerate(zip(texts, entity_predictions, entity_probabilities)):
            entity_info = {
                'text_id': i,
                'text': text,
                'predicted_entity_class': int(entity_pred),
                'entity_confidence': float(entity_prob[entity_pred]),
                'top_entity_predictions': self._get_top_predictions(entity_prob, k=3)
            }
            results['entities'].append(entity_info)
        edge_index = graph_data.edge_index.cpu().numpy()
        for i, (relation_pred, relation_prob) in enumerate(zip(relation_predictions, relation_probabilities)):
            if i < len(edge_index[0]):
                src_node = int(edge_index[0][i])
                dst_node = int(edge_index[1][i])
                relation_info = {
                    'edge_id': i,
                    'source_node': src_node,
                    'target_node': dst_node,
                    'source_text': texts[src_node] if src_node < len(texts) else 'N/A',
                    'target_text': texts[dst_node] if dst_node < len(texts) else 'N/A',
                    'predicted_relation_class': int(relation_pred),
                    'relation_confidence': float(relation_prob[relation_pred]),
                    'top_relation_predictions': self._get_top_predictions(relation_prob, k=3)
                }
                results['relations'].append(relation_info)
        return results

    def _get_top_predictions(self, probabilities: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        top_indices = np.argsort(probabilities)[-k:][::-1]
        return [
            {
                'class_id': int(idx),
                'probability': float(probabilities[idx])
            }
            for idx in top_indices
        ]

    def _compute_graph_statistics(self, graph_data: Data) -> Dict[str, Any]:
        return {
            'num_nodes': int(graph_data.num_nodes),
            'num_edges': int(graph_data.num_edges),
            'avg_degree': float(graph_data.num_edges / graph_data.num_nodes) if graph_data.num_nodes > 0 else 0.0,
            'edge_density': float(graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1))) if graph_data.num_nodes > 1 else 0.0
        }

    def train(self, train_dataset: List[Dict[str, Any]], val_dataset: List[Dict[str, Any]], num_epochs: int = None, save_checkpoints: bool = True) -> Dict[str, Any]:
        self.logger.info(f"Starting training with {len(train_dataset)} training samples")
        training_config = self.config.get_section('training')
        num_epochs = num_epochs or training_config.get('num_epochs', 100)
        self._setup_optimizer_and_scheduler()
        train_loader = self._create_data_loader(train_dataset, shuffle=True)
        val_loader = self._create_data_loader(val_dataset, shuffle=False)
        training_history = []
        best_val_score = 0.0
        patience_counter = 0
        patience = training_config.get('early_stopping_patience', 10)
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate_epoch(val_loader)
            epoch_metrics = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'validation': val_metrics
            }
            training_history.append(epoch_metrics)
            current_val_score = val_metrics['total_f1']
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                patience_counter = 0
                if save_checkpoints:
                    self._save_best_model(epoch, current_val_score)
                self.logger.info(f"New best validation score: {current_val_score:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            if self.scheduler:
                self.scheduler.step(current_val_score)
        self.training_state['epoch'] = epoch + 1
        self.training_state['best_score'] = best_val_score
        self.training_state['training_history'] = training_history
        self.logger.info(f"Training completed. Best validation score: {best_val_score:.4f}")
        return {
            'training_history': training_history,
            'best_validation_score': best_val_score,
            'final_epoch': epoch + 1
        }

    def _setup_optimizer_and_scheduler(self):
        training_config = self.config.get_section('training')
        optimizer_config = training_config.get('optimizer', {})
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        self.optimizer = optim.AdamW(
            self.gnn_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler_config = training_config.get('scheduler', {})
        if scheduler_config.get('type') == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )

    def _create_data_loader(self, dataset: List[Dict[str, Any]], shuffle: bool) -> Iterator[Data]:
        batch_size = self.config.get('training.batch_size', 1)
        graph_objects = []
        for sample in dataset:
            texts = sample['texts'] if isinstance(sample['texts'], list) else [sample['texts']]
            annotations = sample.get('annotations')
            embeddings = self.embedder.create_node_features(texts)
            graph_data = self.graph_builder.build_maintenance_graph(
                node_features=embeddings,
                texts=texts,
                annotations=annotations
            )
            if annotations:
                graph_data = self._add_labels_to_graph(graph_data, annotations)
            graph_objects.append(graph_data)
        return GeometricDataLoader(
            graph_objects,
            batch_size=batch_size,
            shuffle=shuffle,
            follow_batch=['x']
        )

    def _add_labels_to_graph(self, graph_data: Data, annotations: Dict[str, Any]) -> Data:
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.num_edges
        entity_labels = torch.zeros(num_nodes, dtype=torch.long)
        if 'entity_labels' in annotations:
            entity_labels = torch.tensor(annotations['entity_labels'], dtype=torch.long)
        relation_labels = torch.zeros(num_edges, dtype=torch.long)
        if 'relation_labels' in annotations:
            relation_labels = torch.tensor(annotations['relation_labels'], dtype=torch.long)
        graph_data.entity_labels = entity_labels
        graph_data.relation_labels = relation_labels
        return graph_data

    def _train_epoch(self, train_loader: Iterator[Data]) -> Dict[str, float]:
        self.gnn_model.train()
        total_loss = 0.0
        entity_losses = []
        relation_losses = []
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            outputs = self.gnn_model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )
            losses = self.loss_function(
                outputs=outputs,
                entity_targets=batch.entity_labels,
                relation_targets=batch.relation_labels
            )
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                self.gnn_model.parameters(),
                self.config.get('training.gradient_clipping', 1.0)
            )
            self.optimizer.step()
            total_loss += losses['total_loss'].item()
            entity_losses.append(losses['entity_loss'].item())
            relation_losses.append(losses['relation_loss'].item())
            num_batches += 1
        return {
            'total_loss': total_loss / num_batches,
            'entity_loss': np.mean(entity_losses),
            'relation_loss': np.mean(relation_losses),
            'num_batches': num_batches
        }

    def _validate_epoch(self, val_loader: Iterator[Data]) -> Dict[str, float]:
        self.gnn_model.eval()
        all_entity_predictions = []
        all_entity_targets = []
        all_relation_predictions = []
        all_relation_targets = []
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.gnn_model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )
                losses = self.loss_function(
                    outputs=outputs,
                    entity_targets=batch.entity_labels,
                    relation_targets=batch.relation_labels
                )
                total_loss += losses['total_loss'].item()
                entity_preds = torch.argmax(outputs['entity_logits'], dim=-1)
                relation_preds = torch.argmax(outputs['relation_logits'], dim=-1)
                all_entity_predictions.extend(entity_preds.cpu().numpy())
                all_entity_targets.extend(batch.entity_labels.cpu().numpy())
                all_relation_predictions.extend(relation_preds.cpu().numpy())
                all_relation_targets.extend(batch.relation_labels.cpu().numpy())
                num_batches += 1
        entity_metrics = self.metrics_calculator.calculate_entity_metrics(
            predictions=all_entity_predictions,
            targets=all_entity_targets
        )
        relation_metrics = self.metrics_calculator.calculate_relation_metrics(
            predictions=all_relation_predictions,
            targets=all_relation_targets
        )
        combined_metrics = {
            'total_loss': total_loss / num_batches,
            'entity_f1': entity_metrics['f1'],
            'entity_precision': entity_metrics['precision'],
            'entity_recall': entity_metrics['recall'],
            'relation_f1': relation_metrics['f1'],
            'relation_precision': relation_metrics['precision'],
            'relation_recall': relation_metrics['recall'],
            'total_f1': (entity_metrics['f1'] + relation_metrics['f1']) / 2.0
        }
        return combined_metrics

    def _save_best_model(self, epoch: int, score: float):
        models_dir = Path(self.config.get('paths.models_dir', 'models/checkpoints'))
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = models_dir / f'best_model_epoch_{epoch}_score_{score:.4f}.pt'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.gnn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'score': score,
            'training_state': self.training_state
        }
        torch.save(checkpoint, checkpoint_path)
        self.training_state['best_model_path'] = str(checkpoint_path)
        self.logger.info(f"Saved best model to {checkpoint_path}")

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gnn_model.load_state_dict(checkpoint['model_state_dict'])
        if 'training_state' in checkpoint:
            self.training_state = checkpoint['training_state']
        self.logger.info(f"Loaded model from {checkpoint_path}")

    def save_model(self, filepath: str):
        checkpoint = {
            'model_state_dict': self.gnn_model.state_dict(),
            'config': self.config.to_dict(),
            'training_state': self.training_state
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved model to {filepath}")

    def evaluate(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info(f"Evaluating on {len(test_dataset)} test samples")
        test_loader = self._create_data_loader(test_dataset, shuffle=False)
        test_metrics = self._validate_epoch(test_loader)
        detailed_results = {
            'test_metrics': test_metrics,
            'model_info': {
                'num_parameters': sum(p.numel() for p in self.gnn_model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.gnn_model.parameters()) / 1024 / 1024
            },
            'config': self.config.to_dict()
        }
        self.logger.info(f"Test evaluation completed. F1 Score: {test_metrics['total_f1']:.4f}")
        return detailed_results

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'MaintIE LLM-GNN Hybrid',
            'num_parameters': sum(p.numel() for p in self.gnn_model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.gnn_model.parameters()) / 1024 / 1024,
            'device': str(self.device),
            'training_state': self.training_state,
            'embedder_info': {
                'model_name': self.embedder.model.get_sentence_embedding_dimension(),
                'embedding_dim': self.embedder.embedding_dim,
                'combined_dim': self.embedder.combined_dim
            },
            'gnn_info': {
                'type': self.gnn_model.gnn_type,
                'hidden_dim': self.gnn_model.hidden_dim,
                'num_layers': self.gnn_model.num_gnn_layers,
                'num_entity_classes': self.gnn_model.num_entity_classes,
                'num_relation_classes': self.gnn_model.num_relation_classes
            }
        }

# Example usage and testing
if __name__ == "__main__":
    config = {
        'model': {
            'llm_embedder': {
                'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                'feature_config': {'domain_features_dim': 32},
                'combined_features_dim': 416
            },
            'gnn': {
                'hidden_dim': 256,
                'type': 'GAT',
                'num_heads': 8,
                'num_layers': 2,
                'dropout': 0.2
            },
            'num_entity_classes': 224,
            'num_relation_classes': 6
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'early_stopping_patience': 10
        },
        'graph': {
            'similarity_threshold': 0.75,
            'k_neighbors': 15,
            'edge_types': ['semantic_similarity', 'entity_cooccurrence']
        },
        'data': {
            'ontology_path': 'data/raw/scheme.json'
        }
    }
    model = MaintIELLMGNNHybrid(config)
    sample_texts = [
        "Replace faulty pressure sensor in cooling system pump #3",
        "Inspect vibration levels in main turbine bearing assembly",
        "Clean and lubricate conveyor belt drive motor"
    ]
    results = model.extract_maintenance_info(
        texts=sample_texts,
        return_embeddings=True,
        return_graphs=True
    )
    print("Extraction Results:")
    print(f"Number of texts processed: {results['num_texts']}")
    print(f"Number of entities found: {len(results['entities'])}")
    print(f"Number of relations found: {len(results['relations'])}")
    print(f"Processing time: {results['processing_time']:.2f} seconds")
    model_info = model.get_model_info()
    print(f"\nModel Info:")
    print(f"Parameters: {model_info['num_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"Device: {model_info['device']}")
