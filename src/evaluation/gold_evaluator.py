from typing import List, Dict, Optional, Tuple, Union, Any, Set
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix,
    classification_report, matthews_corrcoef
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import warnings

# Import custom modules
from ..models.llm_gnn_hybrid import MaintIELLMGNNHybrid
from ..utils.file_handlers import JSONHandler, PickleHandler
from ..core.config import ConfigManager

class ComplexityLevel(Enum):
    FG_0 = "FG-0"
    FG_1 = "FG-1"
    FG_2 = "FG-2"
    FG_3 = "FG-3"

@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    support: int
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None

@dataclass
class StatisticalAnalysis:
    confidence_interval_95: Tuple[float, float]
    standard_error: float
    bootstrap_samples: List[float]
    significance_test: Optional[Dict[str, float]] = None

class GoldStandardEvaluator:
    """
    Evaluation framework for MaintIE gold standard validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager(config)
        self.complexity_levels = [ComplexityLevel.FG_0, ComplexityLevel.FG_1, ComplexityLevel.FG_2, ComplexityLevel.FG_3]
        self.current_complexity = ComplexityLevel.FG_3
        self.entity_mappings = self._load_entity_mappings()
        self.relation_mappings = self._load_relation_mappings()
        self.baseline_results = self._load_baseline_results()
        self.bootstrap_samples = self.config.get('evaluation.bootstrap_samples', 1000)
        self.confidence_level = self.config.get('evaluation.confidence_level', 0.95)
        self.output_dir = Path(self.config.get('paths.evaluation_output', 'results/evaluation'))
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir = self.output_dir / 'tables'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Initialized GoldStandardEvaluator")

    def _load_entity_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            'FG-0': {'O': 0},
            'FG-1': {
                'PhysicalObject': 0,
                'Activity': 1,
                'Process': 2,
                'State': 3,
                'Property': 4
            },
            'FG-2': {
                'PhysicalObject/SensingObject': 0,
                'PhysicalObject/DrivingObject': 1,
                'PhysicalObject/ProcessingObject': 2,
            },
            'FG-3': {
                'PhysicalObject/SensingObject/PressureSensor': 0,
                'PhysicalObject/SensingObject/TemperatureSensor': 1,
            }
        }

    def _load_relation_mappings(self) -> Dict[str, int]:
        return {
            'located_at': 0,
            'part_of': 1,
            'connected_to': 2,
            'affects': 3,
            'caused_by': 4,
            'no_relation': 5
        }

    def _load_baseline_results(self) -> Dict[str, Dict[str, float]]:
        return {
            'SPERT': {
                'FG-0': {'NER_F1': 96.8, 'RE_F1': 85.2},
                'FG-1': {'NER_F1': 87.39, 'RE_F1': 71.28},
                'FG-2': {'NER_F1': 73.51, 'RE_F1': 51.94},
                'FG-3': {'NER_F1': 62.33, 'RE_F1': 41.07}
            },
            'REBEL': {
                'FG-0': {'NER_F1': 94.1, 'RE_F1': 82.5},
                'FG-1': {'NER_F1': 85.12, 'RE_F1': 67.87},
                'FG-2': {'NER_F1': 70.24, 'RE_F1': 48.33},
                'FG-3': {'NER_F1': 58.91, 'RE_F1': 37.95}
            }
        }

    def evaluate_model(self, model: MaintIELLMGNNHybrid, gold_dataset: List[Dict[str, Any]], complexity_levels: Optional[List[str]] = None, save_results: bool = True) -> Dict[str, Any]:
        self.logger.info(f"Starting gold standard evaluation on {len(gold_dataset)} samples")
        complexity_levels = complexity_levels or ['FG-0', 'FG-1', 'FG-2', 'FG-3']
        evaluation_results = {
            'model_info': model.get_model_info(),
            'dataset_info': self._analyze_dataset(gold_dataset),
            'complexity_results': {},
            'overall_metrics': {},
            'baseline_comparison': {},
            'statistical_analysis': {},
            'error_analysis': {}
        }
        for complexity in complexity_levels:
            self.logger.info(f"Evaluating complexity level: {complexity}")
            complexity_dataset = self._filter_dataset_by_complexity(gold_dataset, complexity)
            complexity_results = self._evaluate_complexity_level(model, complexity_dataset, complexity)
            evaluation_results['complexity_results'][complexity] = complexity_results
        evaluation_results['overall_metrics'] = self._compute_overall_metrics(evaluation_results['complexity_results'])
        evaluation_results['baseline_comparison'] = self._compare_with_baselines(evaluation_results['complexity_results'])
        evaluation_results['statistical_analysis'] = self._perform_statistical_analysis(evaluation_results['complexity_results'])
        evaluation_results['error_analysis'] = self._perform_error_analysis(evaluation_results['complexity_results'])
        if save_results:
            self._save_evaluation_results(evaluation_results)
            self._generate_evaluation_report(evaluation_results)
        self.logger.info("Gold standard evaluation completed")
        return evaluation_results

    def _evaluate_complexity_level(self, model, dataset, complexity):
        self.logger.debug(f"Evaluating {complexity} with {len(dataset)} samples")
        predictions = self._run_model_predictions(model, dataset)
        ground_truth = self._extract_ground_truth(dataset, complexity)
        entity_metrics = self._calculate_entity_metrics(predictions['entities'], ground_truth['entities'], complexity)
        relation_metrics = self._calculate_relation_metrics(predictions['relations'], ground_truth['relations'])
        combined_metrics = self._calculate_combined_metrics(entity_metrics, relation_metrics)
        return {
            'complexity_level': complexity,
            'dataset_size': len(dataset),
            'entity_metrics': entity_metrics,
            'relation_metrics': relation_metrics,
            'combined_metrics': combined_metrics,
            'predictions': predictions,
            'ground_truth': ground_truth
        }

    def _run_model_predictions(self, model, dataset):
        all_predictions = {
            'entities': [],
            'relations': [],
            'texts': [],
            'processing_times': []
        }
        model.gnn_model.eval()
        with torch.no_grad():
            for sample in dataset:
                texts = sample['texts'] if isinstance(sample['texts'], list) else [sample['texts']]
                results = model.extract_maintenance_info(texts)
                for entity_result in results['entities']:
                    all_predictions['entities'].append({
                        'text_id': entity_result['text_id'],
                        'predicted_class': entity_result['predicted_entity_class'],
                        'confidence': entity_result['entity_confidence'],
                        'text': entity_result['text']
                    })
                for relation_result in results['relations']:
                    all_predictions['relations'].append({
                        'source_id': relation_result['source_node'],
                        'target_id': relation_result['target_node'],
                        'predicted_class': relation_result['predicted_relation_class'],
                        'confidence': relation_result['relation_confidence']
                    })
                all_predictions['texts'].extend(texts)
                all_predictions['processing_times'].append(results['processing_time'])
        return all_predictions

    def _extract_ground_truth(self, dataset, complexity):
        ground_truth = {
            'entities': [],
            'relations': []
        }
        entity_mapping = self.entity_mappings.get(complexity, {})
        for sample in dataset:
            annotations = sample.get('annotations', {})
            if 'entities' in annotations:
                for entity in annotations['entities']:
                    entity_type = entity.get('type', 'O')
                    entity_id = entity_mapping.get(entity_type, 0)
                    ground_truth['entities'].append({
                        'text_id': entity.get('text_id', 0),
                        'true_class': entity_id,
                        'entity_type': entity_type,
                        'span': entity.get('span', [0, 0])
                    })
            if 'relations' in annotations:
                for relation in annotations['relations']:
                    relation_type = relation.get('type', 'no_relation')
                    relation_id = self.relation_mappings.get(relation_type, 5)
                    ground_truth['relations'].append({
                        'source_id': relation.get('source', 0),
                        'target_id': relation.get('target', 0),
                        'true_class': relation_id,
                        'relation_type': relation_type
                    })
        return ground_truth

    def _calculate_entity_metrics(self, predictions, ground_truth, complexity):
        pred_labels, true_labels = self._align_entity_predictions(predictions, ground_truth)
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels)
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(true_labels, pred_labels, average=None, zero_division=0)
        entity_mapping = self.entity_mappings.get(complexity, {})
        class_names = list(entity_mapping.keys())
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(per_class_precision):
                per_class_metrics[class_name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i]),
                    'support': int(per_class_support[i])
                }
        return EvaluationMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            support=int(np.sum(support)),
            confusion_matrix=cm,
            per_class_metrics=per_class_metrics
        )

    def _calculate_relation_metrics(self, predictions, ground_truth):
        pred_labels, true_labels = self._align_relation_predictions(predictions, ground_truth)
        precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels)
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(true_labels, pred_labels, average=None, zero_division=0)
        relation_names = list(self.relation_mappings.keys())
        per_class_metrics = {}
        for i, relation_name in enumerate(relation_names):
            if i < len(per_class_precision):
                per_class_metrics[relation_name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i]),
                    'support': int(per_class_support[i])
                }
        return EvaluationMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            support=int(np.sum(support)),
            confusion_matrix=cm,
            per_class_metrics=per_class_metrics
        )

    def _align_entity_predictions(self, predictions, ground_truth):
        pred_by_text = {pred['text_id']: pred for pred in predictions}
        true_by_text = {gt['text_id']: gt for gt in ground_truth}
        pred_labels = []
        true_labels = []
        for text_id in set(pred_by_text.keys()) | set(true_by_text.keys()):
            pred_class = pred_by_text.get(text_id, {}).get('predicted_class', 0)
            true_class = true_by_text.get(text_id, {}).get('true_class', 0)
            pred_labels.append(pred_class)
            true_labels.append(true_class)
        return pred_labels, true_labels

    def _align_relation_predictions(self, predictions, ground_truth):
        pred_by_pair = {(pred['source_id'], pred['target_id']): pred for pred in predictions}
        true_by_pair = {(gt['source_id'], gt['target_id']): gt for gt in ground_truth}
        pred_labels = []
        true_labels = []
        for pair in set(pred_by_pair.keys()) | set(true_by_pair.keys()):
            pred_class = pred_by_pair.get(pair, {}).get('predicted_class', 5)
            true_class = true_by_pair.get(pair, {}).get('true_class', 5)
            pred_labels.append(pred_class)
            true_labels.append(true_class)
        return pred_labels, true_labels

    def _calculate_combined_metrics(self, entity_metrics, relation_metrics):
        return {
            'combined_f1': (entity_metrics.f1 + relation_metrics.f1) / 2.0,
            'combined_precision': (entity_metrics.precision + relation_metrics.precision) / 2.0,
            'combined_recall': (entity_metrics.recall + relation_metrics.recall) / 2.0,
            'entity_f1': entity_metrics.f1,
            'relation_f1': relation_metrics.f1,
            'entity_accuracy': entity_metrics.accuracy,
            'relation_accuracy': relation_metrics.accuracy
        }

    def _perform_statistical_analysis(self, complexity_results):
        statistical_results = {}
        for complexity, results in complexity_results.items():
            entity_f1_scores = self._bootstrap_metric(
                results['predictions']['entities'],
                results['ground_truth']['entities'],
                lambda p, g: self._calculate_entity_metrics(p, g, complexity).f1
            )
            relation_f1_scores = self._bootstrap_metric(
                results['predictions']['relations'],
                results['ground_truth']['relations'],
                lambda p, g: self._calculate_relation_metrics(p, g).f1
            )
            entity_ci = self._calculate_confidence_interval(entity_f1_scores)
            relation_ci = self._calculate_confidence_interval(relation_f1_scores)
            statistical_results[complexity] = {
                'entity_f1_ci': entity_ci,
                'relation_f1_ci': relation_ci,
                'entity_f1_std': float(np.std(entity_f1_scores)),
                'relation_f1_std': float(np.std(relation_f1_scores)),
                'entity_f1_samples': len(entity_f1_scores),
                'relation_f1_samples': len(relation_f1_scores)
            }
        return statistical_results

    def _bootstrap_metric(self, predictions, ground_truth, metric_func):
        bootstrap_scores = []
        for _ in range(self.bootstrap_samples):
            indices = np.random.choice(len(predictions), len(predictions), replace=True)
            sample_predictions = [predictions[i] for i in indices]
            sample_ground_truth = [ground_truth[i] for i in indices if i < len(ground_truth)]
            try:
                score = metric_func(sample_predictions, sample_ground_truth)
                bootstrap_scores.append(score)
            except Exception:
                continue
        return bootstrap_scores

    def _calculate_confidence_interval(self, scores):
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_bound = np.percentile(scores, lower_percentile)
        upper_bound = np.percentile(scores, upper_percentile)
        return (float(lower_bound), float(upper_bound))

    def _compare_with_baselines(self, complexity_results):
        comparison_results = {}
        for complexity, results in complexity_results.items():
            entity_f1 = results['entity_metrics'].f1
            relation_f1 = results['relation_metrics'].f1
            comparison_results[complexity] = {
                'our_results': {
                    'NER_F1': entity_f1,
                    'RE_F1': relation_f1
                },
                'baseline_comparison': {}
            }
            for baseline_name, baseline_results in self.baseline_results.items():
                if complexity in baseline_results:
                    baseline_ner_f1 = baseline_results[complexity]['NER_F1']
                    baseline_re_f1 = baseline_results[complexity]['RE_F1']
                    comparison_results[complexity]['baseline_comparison'][baseline_name] = {
                        'NER_F1': baseline_ner_f1,
                        'RE_F1': baseline_re_f1,
                        'NER_improvement': entity_f1 - baseline_ner_f1,
                        'RE_improvement': relation_f1 - baseline_re_f1,
                        'NER_relative_improvement': ((entity_f1 - baseline_ner_f1) / baseline_ner_f1) * 100,
                        'RE_relative_improvement': ((relation_f1 - baseline_re_f1) / baseline_re_f1) * 100
                    }
        return comparison_results

    def _perform_error_analysis(self, complexity_results):
        error_analysis = {}
        for complexity, results in complexity_results.items():
            entity_errors = self._analyze_entity_errors(
                results['predictions']['entities'],
                results['ground_truth']['entities'],
                complexity
            )
            relation_errors = self._analyze_relation_errors(
                results['predictions']['relations'],
                results['ground_truth']['relations']
            )
            error_analysis[complexity] = {
                'entity_errors': entity_errors,
                'relation_errors': relation_errors
            }
        return error_analysis

    def _analyze_entity_errors(self, predictions, ground_truth, complexity):
        pred_labels, true_labels = self._align_entity_predictions(predictions, ground_truth)
        misclassified = []
        for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
            if pred != true:
                misclassified.append({
                    'index': i,
                    'predicted': pred,
                    'true': true,
                    'confidence': predictions[i].get('confidence', 0.0) if i < len(predictions) else 0.0
                })
        error_patterns = defaultdict(int)
        for error in misclassified:
            pattern = f"{error['true']} -> {error['predicted']}"
            error_patterns[pattern] += 1
        return {
            'total_errors': len(misclassified),
            'error_rate': len(misclassified) / len(pred_labels) if pred_labels else 0.0,
            'most_common_errors': dict(Counter(error_patterns).most_common(10)),
            'low_confidence_errors': [e for e in misclassified if e['confidence'] < 0.5]
        }

    def _analyze_relation_errors(self, predictions, ground_truth):
        pred_labels, true_labels = self._align_relation_predictions(predictions, ground_truth)
        misclassified = []
        for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
            if pred != true:
                misclassified.append({
                    'index': i,
                    'predicted': pred,
                    'true': true,
                    'confidence': predictions[i].get('confidence', 0.0) if i < len(predictions) else 0.0
                })
        error_patterns = defaultdict(int)
        for error in misclassified:
            pattern = f"{error['true']} -> {error['predicted']}"
            error_patterns[pattern] += 1
        return {
            'total_errors': len(misclassified),
            'error_rate': len(misclassified) / len(pred_labels) if pred_labels else 0.0,
            'most_common_errors': dict(Counter(error_patterns).most_common(10)),
            'low_confidence_errors': [e for e in misclassified if e['confidence'] < 0.5]
        }

    def _compute_overall_metrics(self, complexity_results):
        all_entity_f1 = []
        all_relation_f1 = []
        all_combined_f1 = []
        for results in complexity_results.values():
            all_entity_f1.append(results['entity_metrics'].f1)
            all_relation_f1.append(results['relation_metrics'].f1)
            all_combined_f1.append(results['combined_metrics']['combined_f1'])
        return {
            'mean_entity_f1': float(np.mean(all_entity_f1)),
            'mean_relation_f1': float(np.mean(all_relation_f1)),
            'mean_combined_f1': float(np.mean(all_combined_f1)),
            'std_entity_f1': float(np.std(all_entity_f1)),
            'std_relation_f1': float(np.std(all_relation_f1)),
            'std_combined_f1': float(np.std(all_combined_f1)),
            'complexity_levels_evaluated': len(complexity_results)
        }

    def _analyze_dataset(self, dataset):
        total_samples = len(dataset)
        total_entities = 0
        total_relations = 0
        entity_types = Counter()
        relation_types = Counter()
        for sample in dataset:
            annotations = sample.get('annotations', {})
            if 'entities' in annotations:
                entities = annotations['entities']
                total_entities += len(entities)
                for entity in entities:
                    entity_types[entity.get('type', 'unknown')] += 1
            if 'relations' in annotations:
                relations = annotations['relations']
                total_relations += len(relations)
                for relation in relations:
                    relation_types[relation.get('type', 'unknown')] += 1
        return {
            'total_samples': total_samples,
            'total_entities': total_entities,
            'total_relations': total_relations,
            'avg_entities_per_sample': total_entities / total_samples if total_samples > 0 else 0,
            'avg_relations_per_sample': total_relations / total_samples if total_samples > 0 else 0,
            'entity_type_distribution': dict(entity_types.most_common(20)),
            'relation_type_distribution': dict(relation_types.most_common()),
            'unique_entity_types': len(entity_types),
            'unique_relation_types': len(relation_types)
        }

    def _filter_dataset_by_complexity(self, dataset, complexity):
        return dataset

    def _save_evaluation_results(self, results):
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f'evaluation_results_{timestamp}.json'
        JSONHandler.save(results, str(results_file))
        summary_data = []
        for complexity, complexity_results in results['complexity_results'].items():
            summary_data.append({
                'Complexity': complexity,
                'Entity_F1': complexity_results['entity_metrics'].f1,
                'Relation_F1': complexity_results['relation_metrics'].f1,
                'Combined_F1': complexity_results['combined_metrics']['combined_f1'],
                'Dataset_Size': complexity_results['dataset_size']
            })
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.tables_dir / f'results_summary_{timestamp}.csv'
        summary_df.to_csv(summary_file, index=False)
        self.logger.info(f"Saved evaluation results to {results_file}")

    def _generate_evaluation_report(self, results):
        self._plot_complexity_performance(results)
        self._plot_baseline_comparison(results)
        self._plot_confusion_matrices(results)
        self._generate_latex_tables(results)
        self.logger.info("Generated evaluation report with plots and tables")

    def _plot_complexity_performance(self, results):
        complexities = list(results['complexity_results'].keys())
        entity_f1s = [results['complexity_results'][c]['entity_metrics'].f1 for c in complexities]
        relation_f1s = [results['complexity_results'][c]['relation_metrics'].f1 for c in complexities]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(complexities))
        width = 0.35
        ax.bar(x - width/2, entity_f1s, width, label='Entity F1', alpha=0.8)
        ax.bar(x + width/2, relation_f1s, width, label='Relation F1', alpha=0.8)
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance Across Complexity Levels')
        ax.set_xticks(x)
        ax.set_xticklabels(complexities)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'complexity_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_baseline_comparison(self, results):
        comparison_data = []
        for complexity, comp_results in results['baseline_comparison'].items():
            our_ner = comp_results['our_results']['NER_F1']
            our_re = comp_results['our_results']['RE_F1']
            comparison_data.append({
                'Complexity': complexity,
                'Method': 'Our Method',
                'NER_F1': our_ner,
                'RE_F1': our_re
            })
            for baseline_name, baseline_data in comp_results['baseline_comparison'].items():
                comparison_data.append({
                    'Complexity': complexity,
                    'Method': baseline_name,
                    'NER_F1': baseline_data['NER_F1'],
                    'RE_F1': baseline_data['RE_F1']
                })
        df = pd.DataFrame(comparison_data)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            ax1.plot(method_data['Complexity'], method_data['NER_F1'], marker='o', label=method, linewidth=2)
        ax1.set_xlabel('Complexity Level')
        ax1.set_ylabel('NER F1 Score')
        ax1.set_title('Named Entity Recognition Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            ax2.plot(method_data['Complexity'], method_data['RE_F1'], marker='s', label=method, linewidth=2)
        ax2.set_xlabel('Complexity Level')
        ax2.set_ylabel('Relation Extraction F1 Score')
        ax2.set_title('Relation Extraction Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self, results):
        for complexity, complexity_results in results['complexity_results'].items():
            entity_cm = complexity_results['entity_metrics'].confusion_matrix
            if entity_cm is not None and entity_cm.size > 0:
                plt.figure(figsize=(8, 6))
                sns.heatmap(entity_cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Entity Confusion Matrix - {complexity}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(self.figures_dir / f'entity_confusion_matrix_{complexity}.png', dpi=300, bbox_inches='tight')
                plt.close()
            relation_cm = complexity_results['relation_metrics'].confusion_matrix
            if relation_cm is not None and relation_cm.size > 0:
                plt.figure(figsize=(8, 6))
                sns.heatmap(relation_cm, annot=True, fmt='d', cmap='Reds')
                plt.title(f'Relation Confusion Matrix - {complexity}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig(self.figures_dir / f'relation_confusion_matrix_{complexity}.png', dpi=300, bbox_inches='tight')
                plt.close()

    def _generate_latex_tables(self, results):
        latex_content = self._create_main_results_table(results)
        with open(self.tables_dir / 'main_results.tex', 'w') as f:
            f.write(latex_content)
        comparison_latex = self._create_baseline_comparison_table(results)
        with open(self.tables_dir / 'baseline_comparison.tex', 'w') as f:
            f.write(comparison_latex)

    def _create_main_results_table(self, results) -> str:
        table_rows = []
        for complexity, comp_results in results['complexity_results'].items():
            entity_f1 = comp_results['entity_metrics'].f1 * 100
            relation_f1 = comp_results['relation_metrics'].f1 * 100
            combined_f1 = comp_results['combined_metrics']['combined_f1'] * 100
            stat_analysis = results.get('statistical_analysis', {}).get(complexity, {})
            entity_ci = stat_analysis.get('entity_f1_ci', (0, 0))
            relation_ci = stat_analysis.get('relation_f1_ci', (0, 0))
            row = f"{complexity} & {entity_f1:.2f} $\\pm$ {(entity_ci[1] - entity_ci[0])/2:.2f} & " \
                  f"{relation_f1:.2f} $\\pm$ {(relation_ci[1] - relation_ci[0])/2:.2f} & " \
                  f"{combined_f1:.2f} \\\\" 
            table_rows.append(row)
        latex_table = """
\begin{table}[h]
\centering
\caption{MaintIE LLM-GNN Results Across Complexity Levels}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
Complexity & Entity F1 & Relation F1 & Combined F1 \\
\midrule
""" + "\n".join(table_rows) + """
\bottomrule
\end{tabular}
\end{table}
"""
        return latex_table

    def _create_baseline_comparison_table(self, results) -> str:
        table_rows = []
        for complexity in ['FG-0', 'FG-1', 'FG-2', 'FG-3']:
            if complexity in results['baseline_comparison']:
                comp_data = results['baseline_comparison'][complexity]
                our_ner = comp_data['our_results']['NER_F1']
                our_re = comp_data['our_results']['RE_F1']
                spert_data = comp_data['baseline_comparison'].get('SPERT', {})
                spert_ner = spert_data.get('NER_F1', 0)
                spert_re = spert_data.get('RE_F1', 0)
                rebel_data = comp_data['baseline_comparison'].get('REBEL', {})
                rebel_ner = rebel_data.get('NER_F1', 0)
                rebel_re = rebel_data.get('RE_F1', 0)
                row = f"{complexity} & {spert_ner:.2f} & {spert_re:.2f} & " \
                      f"{rebel_ner:.2f} & {rebel_re:.2f} & " \
                      f"\\textbf{{{our_ner:.2f}}} & \\textbf{{{our_re:.2f}}} \\\\" 
                table_rows.append(row)
        latex_table = """
\begin{table}[h]
\centering
\caption{Comparison with Published Baselines}
\label{tab:baseline_comparison}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{SPERT} & \multicolumn{2}{c}{REBEL} & \multicolumn{2}{c}{Our Method} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Complexity & NER F1 & RE F1 & NER F1 & RE F1 & NER F1 & RE F1 \\
\midrule
""" + "\n".join(table_rows) + """
\bottomrule
\end{tabular}
\end{table}
"""
        return latex_table

# Example usage
if __name__ == "__main__":
    config = {
        'evaluation': {
            'bootstrap_samples': 1000,
            'confidence_level': 0.95
        },
        'paths': {
            'evaluation_output': 'results/evaluation'
        }
    }
    evaluator = GoldStandardEvaluator(config)
    print("GoldStandardEvaluator initialized successfully")
