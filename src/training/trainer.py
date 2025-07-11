from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import numpy as np
import logging
import time
import wandb
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold, StratifiedKFold
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

# Import custom modules
from ..models.llm_gnn_hybrid import MaintIELLMGNNHybrid
from ..evaluation.gold_evaluator import GoldStandardEvaluator
from ..evaluation.metrics_calculator import MaintenanceMetricsCalculator
from ..utils.file_handlers import JSONHandler, PickleHandler
from ..core.config import ConfigManager
from ..data_processing.data_splitter import MaintIEDataSplitter

class MaintIETrainer:
    """
    Comprehensive training pipeline for MaintIE LLM-GNN hybrid models.
    """
    def __init__(self, model: MaintIELLMGNNHybrid, config: Dict[str, Any], experiment_name: Optional[str] = None, use_wandb: bool = True, use_tensorboard: bool = True):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager(config)
        self.model = model
        self.experiment_name = experiment_name or f"maintie_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self._setup_experiment_tracking()
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.training_state = {
            'epoch': 0,
            'global_step': 0,
            'best_scores': {},
            'training_history': [],
            'early_stopping_counter': 0,
            'curriculum_stage': 0
        }
        self.evaluator = GoldStandardEvaluator(config)
        self.metrics_calculator = MaintenanceMetricsCalculator()
        self.data_splitter = MaintIEDataSplitter(config)
        self.performance_tracker = PerformanceTracker()
        self.validation_scores = deque(maxlen=10)
        self.curriculum_config = self.config.get_section('curriculum_learning')
        self.complexity_levels = ['FG-0', 'FG-1', 'FG-2', 'FG-3']
        self.logger.info(f"Initialized MaintIETrainer for experiment: {self.experiment_name}")

    def _setup_experiment_tracking(self):
        self.output_dir = Path(self.config.get('paths.output_dir', 'results'))
        self.checkpoint_dir = self.output_dir / 'checkpoints' / self.experiment_name
        self.log_dir = self.output_dir / 'logs' / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.use_wandb:
            try:
                wandb.init(
                    project="maintie-llm-gnn",
                    name=self.experiment_name,
                    config=self.config.to_dict(),
                    dir=str(self.output_dir)
                )
                self.logger.info("Initialized Weights & Biases tracking")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
                self.use_wandb = False
        if self.use_tensorboard:
            try:
                self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir))
                self.logger.info("Initialized TensorBoard logging")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False

    def train(self, train_dataset: List[Dict[str, Any]], val_dataset: List[Dict[str, Any]], test_dataset: Optional[List[Dict[str, Any]]] = None, num_epochs: Optional[int] = None, curriculum_learning: bool = True) -> Dict[str, Any]:
        self.logger.info(f"Starting training with {len(train_dataset)} training samples")
        training_config = self.config.get_section('training')
        num_epochs = num_epochs or training_config.get('num_epochs', 100)
        self._setup_optimizer_and_scheduler()
        if curriculum_learning and self.curriculum_config.get('enabled', False):
            training_results = self._curriculum_learning_pipeline(train_dataset, val_dataset, test_dataset, num_epochs)
        else:
            training_results = self._standard_training_pipeline(train_dataset, val_dataset, num_epochs)
        if test_dataset:
            test_results = self.evaluate_on_test_set(test_dataset)
            training_results['test_results'] = test_results
        self._save_final_model_and_results(training_results)
        self._cleanup_experiment_tracking()
        self.logger.info("Training pipeline completed successfully")
        return training_results

    def _curriculum_learning_pipeline(self, train_dataset, val_dataset, test_dataset, total_epochs):
        self.logger.info("Starting curriculum learning pipeline")
        curriculum_results = {
            'stages': [],
            'overall_metrics': {},
            'final_model_path': None
        }
        stages = self.curriculum_config.get('stages', [
            {'complexity': 'FG-0', 'epochs': 10, 'entity_classes': 1},
            {'complexity': 'FG-1', 'epochs': 20, 'entity_classes': 5},
            {'complexity': 'FG-2', 'epochs': 30, 'entity_classes': 32},
            {'complexity': 'FG-3', 'epochs': 40, 'entity_classes': 224}
        ])
        accumulated_epochs = 0
        for stage_idx, stage_config in enumerate(stages):
            self.logger.info(f"Starting curriculum stage {stage_idx + 1}: {stage_config['complexity']}")
            # Update model for current complexity level
            self.model.update_for_complexity(stage_config['complexity'])
            # Filter datasets for current complexity
            train_subset = [sample for sample in train_dataset if sample['complexity'] == stage_config['complexity']]
            val_subset = [sample for sample in val_dataset if sample['complexity'] == stage_config['complexity']]
            # Create data loaders
            train_loader = self._create_data_loader(train_subset, shuffle=True, training=True)
            val_loader = self._create_data_loader(val_subset, shuffle=False, training=False)
            # Train for this stage
            stage_results = self._standard_training_pipeline(train_subset, val_subset, stage_config['epochs'])
            curriculum_results['stages'].append({
                'stage': stage_idx + 1,
                'complexity': stage_config['complexity'],
                'epochs_trained': stage_config['epochs'],
                'metrics': stage_results['performance_summary']
            })
            # Early stopping check for curriculum
            if self._should_stop_curriculum(stage_results['performance_summary']):
                self.logger.info(f"Early stopping curriculum at stage {stage_idx + 1}")
                break
            accumulated_epochs += stage_config['epochs']
        if accumulated_epochs < total_epochs:
            remaining_epochs = total_epochs - accumulated_epochs
            self.logger.info(f"Final fine-tuning for {remaining_epochs} epochs on full complexity")
            final_results = self._standard_training_pipeline(train_dataset, val_dataset, remaining_epochs)
            curriculum_results['final_fine_tuning'] = final_results
        return curriculum_results

    def _standard_training_pipeline(self, train_dataset, val_dataset, num_epochs):
        self.logger.info(f"Starting standard training for {num_epochs} epochs")
        train_loader = self._create_data_loader(train_dataset, shuffle=True, training=True)
        val_loader = self._create_data_loader(val_dataset, shuffle=False, training=False)
        training_history = []
        best_val_score = 0.0
        patience_counter = 0
        patience = self.config.get('training.early_stopping_patience', 10)
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self._validate_epoch(val_loader, epoch)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_f1'])
                else:
                    self.scheduler.step()
            epoch_time = time.time() - epoch_start_time
            epoch_metrics = {
                'epoch': epoch + 1,
                'epoch_time': epoch_time,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'train': train_metrics,
                'validation': val_metrics
            }
            training_history.append(epoch_metrics)
            self._log_epoch_metrics(epoch_metrics)
            current_val_score = val_metrics['total_f1']
            self.validation_scores.append(current_val_score)
            if current_val_score > best_val_score:
                best_val_score = current_val_score
                patience_counter = 0
                self._save_checkpoint(epoch, current_val_score, is_best=True)
                self.logger.info(f"Epoch {epoch + 1}: New best validation F1: {current_val_score:.4f}")
            else:
                patience_counter += 1
                self.logger.info(f"Epoch {epoch + 1}: Validation F1: {current_val_score:.4f} (Best: {best_val_score:.4f})")
            if (epoch + 1) % self.config.get('training.checkpoint_frequency', 10) == 0:
                self._save_checkpoint(epoch, current_val_score, is_best=False)
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        training_results = {
            'training_history': training_history,
            'best_validation_f1': best_val_score,
            'total_epochs': epoch + 1,
            'final_learning_rate': self.optimizer.param_groups[0]['lr'],
            'performance_summary': self.performance_tracker.get_summary()
        }
        return training_results

    def _train_epoch(self, train_loader: GeometricDataLoader, epoch: int) -> Dict[str, float]:
        self.model.gnn_model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'entity_loss': 0.0,
            'relation_loss': 0.0,
            'num_batches': 0,
            'num_samples': 0
        }
        batch_times = []
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            batch = batch.to(self.model.device)
            if self.scaler:
                with autocast():
                    outputs = self.model.gnn_model(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        batch=batch.batch
                    )
                    losses = self.model.loss_function(
                        outputs=outputs,
                        entity_targets=batch.entity_labels,
                        relation_targets=batch.relation_labels
                    )
                self.optimizer.zero_grad()
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.gnn_model.parameters(),
                    self.config.get('training.gradient_clipping', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model.gnn_model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )
                losses = self.model.loss_function(
                    outputs=outputs,
                    entity_targets=batch.entity_labels,
                    relation_targets=batch.relation_labels
                )
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.gnn_model.parameters(),
                    self.config.get('training.gradient_clipping', 1.0)
                )
                self.optimizer.step()
            batch_size = batch.num_nodes
            epoch_metrics['total_loss'] += losses['total_loss'].item() * batch_size
            epoch_metrics['entity_loss'] += losses['entity_loss'].item() * batch_size
            epoch_metrics['relation_loss'] += losses['relation_loss'].item() * batch_size
            epoch_metrics['num_batches'] += 1
            epoch_metrics['num_samples'] += batch_size
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            self.training_state['global_step'] += 1
            if batch_idx % self.config.get('training.log_frequency', 50) == 0:
                self._log_batch_metrics(epoch, batch_idx, losses, batch_time)
        num_samples = epoch_metrics['num_samples']
        normalized_metrics = {
            'total_loss': epoch_metrics['total_loss'] / num_samples,
            'entity_loss': epoch_metrics['entity_loss'] / num_samples,
            'relation_loss': epoch_metrics['relation_loss'] / num_samples,
            'avg_batch_time': np.mean(batch_times),
            'samples_per_second': num_samples / sum(batch_times)
        }
        return normalized_metrics

    def _validate_epoch(self, val_loader: GeometricDataLoader, epoch: int) -> Dict[str, float]:
        self.model.gnn_model.eval()
        all_predictions = {
            'entity_predictions': [],
            'entity_targets': [],
            'relation_predictions': [],
            'relation_targets': []
        }
        validation_loss = 0.0
        num_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.model.device)
                outputs = self.model.gnn_model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch
                )
                losses = self.model.loss_function(
                    outputs=outputs,
                    entity_targets=batch.entity_labels,
                    relation_targets=batch.relation_labels
                )
                validation_loss += losses['total_loss'].item() * batch.num_nodes
                num_samples += batch.num_nodes
                entity_preds = torch.argmax(outputs['entity_logits'], dim=-1)
                relation_preds = torch.argmax(outputs['relation_logits'], dim=-1)
                all_predictions['entity_predictions'].extend(entity_preds.cpu().numpy())
                all_predictions['entity_targets'].extend(batch.entity_labels.cpu().numpy())
                all_predictions['relation_predictions'].extend(relation_preds.cpu().numpy())
                all_predictions['relation_targets'].extend(batch.relation_labels.cpu().numpy())
        entity_metrics = self.metrics_calculator.calculate_entity_metrics(
            predictions=all_predictions['entity_predictions'],
            targets=all_predictions['entity_targets']
        )
        relation_metrics = self.metrics_calculator.calculate_relation_metrics(
            predictions=all_predictions['relation_predictions'],
            targets=all_predictions['relation_targets']
        )
        validation_metrics = {
            'validation_loss': validation_loss / num_samples,
            'entity_f1': entity_metrics['f1'],
            'entity_precision': entity_metrics['precision'],
            'entity_recall': entity_metrics['recall'],
            'relation_f1': relation_metrics['f1'],
            'relation_precision': relation_metrics['precision'],
            'relation_recall': relation_metrics['recall'],
            'total_f1': (entity_metrics['f1'] + relation_metrics['f1']) / 2.0,
            'entity_accuracy': entity_metrics.get('accuracy', 0.0),
            'relation_accuracy': relation_metrics.get('accuracy', 0.0)
        }
        return validation_metrics

    def evaluate_on_test_set(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info(f"Evaluating on test set with {len(test_dataset)} samples")
        test_results = self.evaluator.evaluate_model(self.model, test_dataset)
        test_loader = self._create_data_loader(test_dataset, shuffle=False, training=False)
        detailed_metrics = self._validate_epoch(test_loader, epoch=-1)
        test_results['detailed_metrics'] = detailed_metrics
        test_results['model_info'] = self.model.get_model_info()
        return test_results

    def _setup_optimizer_and_scheduler(self):
        training_config = self.config.get_section('training')
        optimizer_config = training_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        if optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.gnn_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', [0.9, 0.999])
            )
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(
                self.model.gnn_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(
                self.model.gnn_model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=optimizer_config.get('momentum', 0.9)
            )
        scheduler_config = training_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 50),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.1)
            )

    def _create_data_loader(self, dataset: List[Dict[str, Any]], shuffle: bool, training: bool) -> GeometricDataLoader:
        batch_size = self.config.get('training.batch_size', 8)
        graph_objects = []
        for sample in dataset:
            graph_data = self._process_sample_to_graph(sample, training=training)
            graph_objects.append(graph_data)
        return GeometricDataLoader(
            graph_objects,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get('training.num_workers', 0),
            pin_memory=torch.cuda.is_available()
        )

    def _process_sample_to_graph(self, sample: Dict[str, Any], training: bool) -> Data:
        texts = sample['texts'] if isinstance(sample['texts'], list) else [sample['texts']]
        annotations = sample.get('annotations')
        embeddings = self.model.embedder.create_node_features(texts)
        graph_data = self.model.graph_builder.build_maintenance_graph(
            node_features=embeddings,
            texts=texts,
            annotations=annotations
        )
        if annotations and training:
            graph_data = self._add_training_labels(graph_data, annotations)
        return graph_data

    def _add_training_labels(self, graph_data: Data, annotations: Dict[str, Any]) -> Data:
        num_nodes = graph_data.num_nodes
        num_edges = graph_data.num_edges
        entity_labels = torch.zeros(num_nodes, dtype=torch.long)
        if 'entity_labels' in annotations:
            entity_labels = torch.tensor(annotations['entity_labels'][:num_nodes], dtype=torch.long)
        relation_labels = torch.zeros(num_edges, dtype=torch.long)
        if 'relation_labels' in annotations:
            relation_labels = torch.tensor(annotations['relation_labels'][:num_edges], dtype=torch.long)
        graph_data.entity_labels = entity_labels
        graph_data.relation_labels = relation_labels
        return graph_data

    def _log_epoch_metrics(self, metrics: Dict[str, Any]):
        epoch = metrics['epoch']
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'learning_rate': metrics['learning_rate'],
                'epoch_time': metrics['epoch_time'],
                'train/total_loss': metrics['train']['total_loss'],
                'train/entity_loss': metrics['train']['entity_loss'],
                'train/relation_loss': metrics['train']['relation_loss'],
                'val/total_f1': metrics['validation']['total_f1'],
                'val/entity_f1': metrics['validation']['entity_f1'],
                'val/relation_f1': metrics['validation']['relation_f1'],
                'val/validation_loss': metrics['validation']['validation_loss']
            })
        if self.use_tensorboard:
            self.tensorboard_writer.add_scalar('Learning_Rate', metrics['learning_rate'], epoch)
            self.tensorboard_writer.add_scalar('Epoch_Time', metrics['epoch_time'], epoch)
            for key, value in metrics['train'].items():
                self.tensorboard_writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in metrics['validation'].items():
                self.tensorboard_writer.add_scalar(f'Validation/{key}', value, epoch)

    def _log_batch_metrics(self, epoch: int, batch_idx: int, losses: Dict[str, torch.Tensor], batch_time: float):
        global_step = self.training_state['global_step']
        if self.use_wandb:
            wandb.log({
                'batch/total_loss': losses['total_loss'].item(),
                'batch/entity_loss': losses['entity_loss'].item(),
                'batch/relation_loss': losses['relation_loss'].item(),
                'batch/time': batch_time
            }, step=global_step)
        if self.use_tensorboard:
            self.tensorboard_writer.add_scalar('Batch/Total_Loss', losses['total_loss'].item(), global_step)
            self.tensorboard_writer.add_scalar('Batch/Entity_Loss', losses['entity_loss'].item(), global_step)
            self.tensorboard_writer.add_scalar('Batch/Relation_Loss', losses['relation_loss'].item(), global_step)
            self.tensorboard_writer.add_scalar('Batch/Time', batch_time, global_step)

    def _save_checkpoint(self, epoch: int, score: float, is_best: bool = False):
        checkpoint_name = f'checkpoint_epoch_{epoch}_score_{score:.4f}.pt'
        if is_best:
            checkpoint_name = f'best_model.pt'
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.gnn_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'score': score,
            'training_state': self.training_state
        }
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            self.training_state['best_model_path'] = str(checkpoint_path)
        self.logger.debug(f"Saved {'best ' if is_best else ''}checkpoint to {checkpoint_path}")

    def _save_final_model_and_results(self, training_results: Dict[str, Any]):
        final_model_path = self.checkpoint_dir / 'final_model.pt'
        self.model.save_model(str(final_model_path))
        results_path = self.checkpoint_dir / 'training_results.json'
        JSONHandler.save(training_results, str(results_path))
        config_path = self.checkpoint_dir / 'config.json'
        JSONHandler.save(self.config.to_dict(), str(config_path))
        self.logger.info(f"Saved final model and results to {self.checkpoint_dir}")

    def _cleanup_experiment_tracking(self):
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard and hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()

    def _should_stop_curriculum(self, stage_performance: Dict[str, float]) -> bool:
        """Determine if curriculum learning should stop based on performance."""
        current_stage = self.training_state['curriculum_stage']
        if current_stage >= len(self.curriculum_config['stages']):
            return True
        target_metric = self.curriculum_config['stages'][current_stage].get('target_metric', 'total_f1')
        threshold = self.curriculum_config['stages'][current_stage].get('threshold', 0.8)
        patience = self.curriculum_config.get('patience', 5)
        if len(self.validation_scores) < patience:
            return False
        worst_score = min(list(self.validation_scores)[-patience:])
        return stage_performance.get(target_metric, 0.0) < worst_score * threshold

class PerformanceTracker:
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.start_time = time.time()
    def record_metric(self, metric_name: str, value: float, step: int):
        self.metrics_history[metric_name].append({
            'step': step,
            'value': value,
            'timestamp': time.time()
        })
    def get_summary(self) -> Dict[str, Any]:
        summary = {
            'total_training_time': time.time() - self.start_time,
            'metrics_summary': {}
        }
        for metric_name, history in self.metrics_history.items():
            values = [entry['value'] for entry in history]
            summary['metrics_summary'][metric_name] = {
                'final_value': values[-1] if values else 0.0,
                'best_value': max(values) if values else 0.0,
                'mean_value': np.mean(values) if values else 0.0,
                'std_value': np.std(values) if values else 0.0
            }
        return summary

# Example usage
if __name__ == "__main__":
    # This would be used with the actual MaintIE model and datasets
    pass
