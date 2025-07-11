from typing import List, Dict, Optional, Tuple, Union, Any, Set
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix,
    classification_report, matthews_corrcoef, cohen_kappa_score
)
from sklearn.utils import resample
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
from pathlib import Path
import json

class EvaluationMode(Enum):
    STRICT = "strict"
    RELAXED = "relaxed"
    PARTIAL = "partial"
    TYPE_ONLY = "type_only"

class AveragingMethod(Enum):
    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    BINARY = "binary"

@dataclass
class EntityMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    support: int
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    classification_report: str

@dataclass
class RelationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    support: int
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: np.ndarray
    classification_report: str

@dataclass
class DetailedMetrics:
    entity_metrics: EntityMetrics
    relation_metrics: RelationMetrics
    combined_metrics: Dict[str, float]
    statistical_analysis: Dict[str, Any]
    error_analysis: Dict[str, Any]

class MaintenanceMetricsCalculator:
    def __init__(self, 
                 entity_labels: Optional[List[str]] = None,
                 relation_labels: Optional[List[str]] = None,
                 evaluation_mode: EvaluationMode = EvaluationMode.STRICT,
                 averaging_method: AveragingMethod = AveragingMethod.WEIGHTED):
        self.logger = logging.getLogger(__name__)
        self.entity_labels = entity_labels or self._get_default_entity_labels()
        self.relation_labels = relation_labels or self._get_default_relation_labels()
        self.evaluation_mode = evaluation_mode
        self.averaging_method = averaging_method
        self.entity_label_to_idx = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.relation_label_to_idx = {label: idx for idx, label in enumerate(self.relation_labels)}
        self.bootstrap_samples = 1000
        self.confidence_level = 0.95
        self.logger.info(f"Initialized MetricsCalculator with {len(self.entity_labels)} entity types "
                        f"and {len(self.relation_labels)} relation types")

    def _get_default_entity_labels(self) -> List[str]:
        return [
            'PhysicalObject', 'Activity', 'Process', 'State', 'Property',
            'SensingObject', 'DrivingObject', 'ProcessingObject',
            'MaintenanceActivity', 'InspectionActivity', 'RepairActivity',
            'OperationalState', 'FailureState', 'PhysicalProperty'
        ]

    def _get_default_relation_labels(self) -> List[str]:
        return [
            'located_at', 'part_of', 'connected_to', 
            'affects', 'caused_by', 'no_relation'
        ]

    def calculate_entity_metrics(self, 
                                predictions: Union[List[int], List[str], List[Dict[str, Any]]],
                                targets: Union[List[int], List[str], List[Dict[str, Any]]],
                                entity_spans: Optional[List[Tuple[int, int]]] = None) -> EntityMetrics:
        self.logger.debug(f"Calculating entity metrics for {len(predictions)} predictions")
        pred_labels, true_labels, pred_spans, true_spans = self._standardize_entity_inputs(
            predictions, targets, entity_spans
        )
        if self.evaluation_mode == EvaluationMode.STRICT and pred_spans and true_spans:
            aligned_preds, aligned_trues = self._strict_entity_matching(
                pred_labels, true_labels, pred_spans, true_spans
            )
        elif self.evaluation_mode == EvaluationMode.RELAXED and pred_spans and true_spans:
            aligned_preds, aligned_trues = self._relaxed_entity_matching(
                pred_labels, true_labels, pred_spans, true_spans
            )
        else:
            aligned_preds, aligned_trues = pred_labels, true_labels
        precision, recall, f1, support = precision_recall_fscore_support(
            aligned_trues, aligned_preds, 
            average=self.averaging_method.value,
            labels=list(range(len(self.entity_labels))),
            zero_division=0
        )
        accuracy = accuracy_score(aligned_trues, aligned_preds)
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(
                aligned_trues, aligned_preds, 
                average=None,
                labels=list(range(len(self.entity_labels))),
                zero_division=0
            )
        per_class_metrics = {}
        for i, label in enumerate(self.entity_labels):
            per_class_metrics[label] = {
                'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
                'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
                'f1': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
                'support': int(per_class_support[i]) if i < len(per_class_support) else 0
            }
        cm = confusion_matrix(
            aligned_trues, aligned_preds,
            labels=list(range(len(self.entity_labels)))
        )
        report = classification_report(
            aligned_trues, aligned_preds,
            target_names=self.entity_labels,
            zero_division=0
        )
        return EntityMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            support=int(np.sum(support)) if hasattr(support, '__len__') else int(support),
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm,
            classification_report=report
        )

    def calculate_relation_metrics(self, 
                                 predictions: Union[List[int], List[str], List[Dict[str, Any]]],
                                 targets: Union[List[int], List[str], List[Dict[str, Any]]]) -> RelationMetrics:
        self.logger.debug(f"Calculating relation metrics for {len(predictions)} predictions")
        pred_labels, true_labels = self._standardize_relation_inputs(predictions, targets)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels,
            average=self.averaging_method.value,
            labels=list(range(len(self.relation_labels))),
            zero_division=0
        )
        accuracy = accuracy_score(true_labels, pred_labels)
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(
                true_labels, pred_labels,
                average=None,
                labels=list(range(len(self.relation_labels))),
                zero_division=0
            )
        per_class_metrics = {}
        for i, label in enumerate(self.relation_labels):
            per_class_metrics[label] = {
                'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0.0,
                'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0.0,
                'f1': float(per_class_f1[i]) if i < len(per_class_f1) else 0.0,
                'support': int(per_class_support[i]) if i < len(per_class_support) else 0
            }
        cm = confusion_matrix(
            true_labels, pred_labels,
            labels=list(range(len(self.relation_labels)))
        )
        report = classification_report(
            true_labels, pred_labels,
            target_names=self.relation_labels,
            zero_division=0
        )
        return RelationMetrics(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            accuracy=float(accuracy),
            support=int(np.sum(support)) if hasattr(support, '__len__') else int(support),
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm,
            classification_report=report
        )

    def calculate_detailed_metrics(self,
                                 entity_predictions: Any,
                                 entity_targets: Any,
                                 relation_predictions: Any,
                                 relation_targets: Any,
                                 entity_spans: Optional[Any] = None) -> DetailedMetrics:
        self.logger.info("Calculating detailed metrics with statistical analysis")
        entity_metrics = self.calculate_entity_metrics(
            entity_predictions, entity_targets, entity_spans
        )
        relation_metrics = self.calculate_relation_metrics(
            relation_predictions, relation_targets
        )
        combined_metrics = self._calculate_combined_metrics(entity_metrics, relation_metrics)
        statistical_analysis = self._perform_statistical_analysis(
            entity_predictions, entity_targets, relation_predictions, relation_targets
        )
        error_analysis = self._perform_error_analysis(
            entity_predictions, entity_targets, relation_predictions, relation_targets
        )
        return DetailedMetrics(
            entity_metrics=entity_metrics,
            relation_metrics=relation_metrics,
            combined_metrics=combined_metrics,
            statistical_analysis=statistical_analysis,
            error_analysis=error_analysis
        )

    def _standardize_entity_inputs(self, 
                                 predictions: Any, 
                                 targets: Any,
                                 spans: Optional[Any] = None) -> Tuple[List[int], List[int], Optional[List], Optional[List]]:
        pred_labels = []
        true_labels = []
        pred_spans = None
        true_spans = None
        if isinstance(predictions[0], dict):
            pred_labels = [self._entity_to_label_idx(p.get('type', 'O')) for p in predictions]
            pred_spans = [p.get('span', [0, 0]) for p in predictions] if 'span' in predictions[0] else None
        elif isinstance(predictions[0], str):
            pred_labels = [self._entity_to_label_idx(p) for p in predictions]
        else:
            pred_labels = list(predictions)
        if isinstance(targets[0], dict):
            true_labels = [self._entity_to_label_idx(t.get('type', 'O')) for t in targets]
            true_spans = [t.get('span', [0, 0]) for t in targets] if 'span' in targets[0] else None
        elif isinstance(targets[0], str):
            true_labels = [self._entity_to_label_idx(t) for t in targets]
        else:
            true_labels = list(targets)
        if spans and not pred_spans:
            pred_spans = spans
            true_spans = spans
        return pred_labels, true_labels, pred_spans, true_spans

    def _standardize_relation_inputs(self, predictions: Any, targets: Any) -> Tuple[List[int], List[int]]:
        pred_labels = []
        true_labels = []
        if isinstance(predictions[0], dict):
            pred_labels = [self._relation_to_label_idx(p.get('type', 'no_relation')) for p in predictions]
        elif isinstance(predictions[0], str):
            pred_labels = [self._relation_to_label_idx(p) for p in predictions]
        else:
            pred_labels = list(predictions)
        if isinstance(targets[0], dict):
            true_labels = [self._relation_to_label_idx(t.get('type', 'no_relation')) for t in targets]
        elif isinstance(targets[0], str):
            true_labels = [self._relation_to_label_idx(t) for t in targets]
        else:
            true_labels = list(targets)
        return pred_labels, true_labels

    def _entity_to_label_idx(self, entity_label: str) -> int:
        if entity_label in self.entity_label_to_idx:
            return self.entity_label_to_idx[entity_label]
        for label, idx in self.entity_label_to_idx.items():
            if entity_label.startswith(label) or label in entity_label:
                return idx
        return 0

    def _relation_to_label_idx(self, relation_label: str) -> int:
        if relation_label in self.relation_label_to_idx:
            return self.relation_label_to_idx[relation_label]
        return self.relation_label_to_idx.get('no_relation', len(self.relation_labels) - 1)

    def _strict_entity_matching(self, 
                              pred_labels: List[int], 
                              true_labels: List[int],
                              pred_spans: List[Tuple[int, int]], 
                              true_spans: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
        aligned_preds = []
        aligned_trues = []
        pred_entities = {(span[0], span[1], label) for span, label in zip(pred_spans, pred_labels)}
        true_entities = {(span[0], span[1], label) for span, label in zip(true_spans, true_labels)}
        matched_entities = pred_entities & true_entities
        for entity in matched_entities:
            aligned_preds.append(entity[2])
            aligned_trues.append(entity[2])
        for entity in pred_entities - true_entities:
            aligned_preds.append(entity[2])
            aligned_trues.append(0)
        for entity in true_entities - pred_entities:
            aligned_preds.append(0)
            aligned_trues.append(entity[2])
        return aligned_preds, aligned_trues

    def _relaxed_entity_matching(self, 
                               pred_labels: List[int], 
                               true_labels: List[int],
                               pred_spans: List[Tuple[int, int]], 
                               true_spans: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
        aligned_preds = []
        aligned_trues = []
        matched_preds = set()
        matched_trues = set()
        for i, (pred_span, pred_label) in enumerate(zip(pred_spans, pred_labels)):
            for j, (true_span, true_label) in enumerate(zip(true_spans, true_labels)):
                if pred_label == true_label and self._spans_overlap(pred_span, true_span):
                    if i not in matched_preds and j not in matched_trues:
                        aligned_preds.append(pred_label)
                        aligned_trues.append(true_label)
                        matched_preds.add(i)
                        matched_trues.add(j)
                        break
        for i, pred_label in enumerate(pred_labels):
            if i not in matched_preds:
                aligned_preds.append(pred_label)
                aligned_trues.append(0)
        for j, true_label in enumerate(true_labels):
            if j not in matched_trues:
                aligned_preds.append(0)
                aligned_trues.append(true_label)
        return aligned_preds, aligned_trues

    def _spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        return not (span1[1] <= span2[0] or span2[1] <= span1[0])

    def _calculate_combined_metrics(self, 
                                  entity_metrics: EntityMetrics, 
                                  relation_metrics: RelationMetrics) -> Dict[str, float]:
        return {
            'combined_f1': (entity_metrics.f1 + relation_metrics.f1) / 2.0,
            'combined_precision': (entity_metrics.precision + relation_metrics.precision) / 2.0,
            'combined_recall': (entity_metrics.recall + relation_metrics.recall) / 2.0,
            'combined_accuracy': (entity_metrics.accuracy + relation_metrics.accuracy) / 2.0,
            'entity_f1': entity_metrics.f1,
            'relation_f1': relation_metrics.f1,
            'entity_precision': entity_metrics.precision,
            'relation_precision': relation_metrics.precision,
            'entity_recall': entity_metrics.recall,
            'relation_recall': relation_metrics.recall
        }

    def _perform_statistical_analysis(self, 
                                    entity_predictions: Any,
                                    entity_targets: Any,
                                    relation_predictions: Any,
                                    relation_targets: Any) -> Dict[str, Any]:
        statistical_results = {}
        entity_f1_scores = self._bootstrap_entity_f1(entity_predictions, entity_targets)
        entity_ci = self._calculate_confidence_interval(entity_f1_scores)
        relation_f1_scores = self._bootstrap_relation_f1(relation_predictions, relation_targets)
        relation_ci = self._calculate_confidence_interval(relation_f1_scores)
        statistical_results = {
            'entity_f1_confidence_interval': entity_ci,
            'relation_f1_confidence_interval': relation_ci,
            'entity_f1_std': float(np.std(entity_f1_scores)),
            'relation_f1_std': float(np.std(relation_f1_scores)),
            'entity_f1_bootstrap_samples': len(entity_f1_scores),
            'relation_f1_bootstrap_samples': len(relation_f1_scores),
            'bootstrap_iterations': self.bootstrap_samples
        }
        return statistical_results

    def _bootstrap_entity_f1(self, predictions: Any, targets: Any) -> List[float]:
        f1_scores = []
        pred_simple, true_simple, _, _ = self._standardize_entity_inputs(predictions, targets)
        for _ in range(self.bootstrap_samples):
            indices = resample(range(len(pred_simple)), random_state=None)
            bootstrap_preds = [pred_simple[i] for i in indices]
            bootstrap_trues = [true_simple[i] for i in indices]
            try:
                _, _, f1, _ = precision_recall_fscore_support(
                    bootstrap_trues, bootstrap_preds, 
                    average=self.averaging_method.value,
                    zero_division=0
                )
                f1_scores.append(float(f1))
            except Exception:
                continue
        return f1_scores

    def _bootstrap_relation_f1(self, predictions: Any, targets: Any) -> List[float]:
        f1_scores = []
        pred_simple, true_simple = self._standardize_relation_inputs(predictions, targets)
        for _ in range(self.bootstrap_samples):
            indices = resample(range(len(pred_simple)), random_state=None)
            bootstrap_preds = [pred_simple[i] for i in indices]
            bootstrap_trues = [true_simple[i] for i in indices]
            try:
                _, _, f1, _ = precision_recall_fscore_support(
                    bootstrap_trues, bootstrap_preds,
                    average=self.averaging_method.value,
                    zero_division=0
                )
                f1_scores.append(float(f1))
            except Exception:
                continue
        return f1_scores

    def _calculate_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        if not scores:
            return (0.0, 0.0)
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        lower_bound = np.percentile(scores, lower_percentile)
        upper_bound = np.percentile(scores, upper_percentile)
        return (float(lower_bound), float(upper_bound))

    def _perform_error_analysis(self, 
                              entity_predictions: Any,
                              entity_targets: Any,
                              relation_predictions: Any,
                              relation_targets: Any) -> Dict[str, Any]:
        error_analysis = {}
        entity_errors = self._analyze_entity_errors(entity_predictions, entity_targets)
        error_analysis['entity_errors'] = entity_errors
        relation_errors = self._analyze_relation_errors(relation_predictions, relation_targets)
        error_analysis['relation_errors'] = relation_errors
        error_analysis['common_patterns'] = self._identify_common_error_patterns(
            entity_errors, relation_errors
        )
        return error_analysis

    def _analyze_entity_errors(self, predictions: Any, targets: Any) -> Dict[str, Any]:
        pred_labels, true_labels, _, _ = self._standardize_entity_inputs(predictions, targets)
        misclassified = []
        for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
            if pred != true:
                misclassified.append({
                    'index': i,
                    'predicted': pred,
                    'true': true,
                    'predicted_label': self.entity_labels[pred] if pred < len(self.entity_labels) else 'unknown',
                    'true_label': self.entity_labels[true] if true < len(self.entity_labels) else 'unknown'
                })
        error_patterns = Counter()
        for error in misclassified:
            pattern = f"{error['true_label']} -> {error['predicted_label']}"
            error_patterns[pattern] += 1
        return {
            'total_errors': len(misclassified),
            'error_rate': len(misclassified) / len(pred_labels) if pred_labels else 0.0,
            'most_common_errors': dict(error_patterns.most_common(10)),
            'misclassified_samples': misclassified[:50]
        }

    def _analyze_relation_errors(self, predictions: Any, targets: Any) -> Dict[str, Any]:
        pred_labels, true_labels = self._standardize_relation_inputs(predictions, targets)
        misclassified = []
        for i, (pred, true) in enumerate(zip(pred_labels, true_labels)):
            if pred != true:
                misclassified.append({
                    'index': i,
                    'predicted': pred,
                    'true': true,
                    'predicted_label': self.relation_labels[pred] if pred < len(self.relation_labels) else 'unknown',
                    'true_label': self.relation_labels[true] if true < len(self.relation_labels) else 'unknown'
                })
        error_patterns = Counter()
        for error in misclassified:
            pattern = f"{error['true_label']} -> {error['predicted_label']}"
            error_patterns[pattern] += 1
        return {
            'total_errors': len(misclassified),
            'error_rate': len(misclassified) / len(pred_labels) if pred_labels else 0.0,
            'most_common_errors': dict(error_patterns.most_common(10)),
            'misclassified_samples': misclassified[:50]
        }

    def _identify_common_error_patterns(self, 
                                      entity_errors: Dict[str, Any],
                                      relation_errors: Dict[str, Any]) -> Dict[str, Any]:
        patterns = {
            'high_confusion_entity_pairs': [],
            'high_confusion_relation_pairs': [],
            'systematic_biases': {},
            'error_correlations': {}
        }
        entity_patterns = entity_errors.get('most_common_errors', {})
        for pattern, count in entity_patterns.items():
            if count > 5:
                patterns['high_confusion_entity_pairs'].append({
                    'pattern': pattern,
                    'count': count,
                    'error_rate': count / entity_errors.get('total_errors', 1)
                })
        relation_patterns = relation_errors.get('most_common_errors', {})
        for pattern, count in relation_patterns.items():
            if count > 3:
                patterns['high_confusion_relation_pairs'].append({
                    'pattern': pattern,
                    'count': count,
                    'error_rate': count / relation_errors.get('total_errors', 1)
                })
        return patterns

    def compare_with_baseline(self, 
                            our_entity_metrics: EntityMetrics,
                            our_relation_metrics: RelationMetrics,
                            baseline_entity_f1: float,
                            baseline_relation_f1: float) -> Dict[str, Any]:
        comparison = {
            'entity_comparison': {
                'our_f1': our_entity_metrics.f1,
                'baseline_f1': baseline_entity_f1,
                'improvement': our_entity_metrics.f1 - baseline_entity_f1,
                'relative_improvement': ((our_entity_metrics.f1 - baseline_entity_f1) / baseline_entity_f1) * 100 if baseline_entity_f1 > 0 else 0
            },
            'relation_comparison': {
                'our_f1': our_relation_metrics.f1,
                'baseline_f1': baseline_relation_f1,
                'improvement': our_relation_metrics.f1 - baseline_relation_f1,
                'relative_improvement': ((our_relation_metrics.f1 - baseline_relation_f1) / baseline_relation_f1) * 100 if baseline_relation_f1 > 0 else 0
            }
        }
        our_combined = (our_entity_metrics.f1 + our_relation_metrics.f1) / 2
        baseline_combined = (baseline_entity_f1 + baseline_relation_f1) / 2
        comparison['combined_comparison'] = {
            'our_combined_f1': our_combined,
            'baseline_combined_f1': baseline_combined,
            'improvement': our_combined - baseline_combined,
            'relative_improvement': ((our_combined - baseline_combined) / baseline_combined) * 100 if baseline_combined > 0 else 0
        }
        return comparison

    def generate_evaluation_report(self, 
                                 detailed_metrics: DetailedMetrics,
                                 output_path: Optional[str] = None) -> str:
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MAINTIE LLM-GNN EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("ENTITY RECOGNITION METRICS")
        report_lines.append("-" * 40)
        em = detailed_metrics.entity_metrics
        report_lines.append(f"Precision: {em.precision:.4f}")
        report_lines.append(f"Recall:    {em.recall:.4f}")
        report_lines.append(f"F1 Score:  {em.f1:.4f}")
        report_lines.append(f"Accuracy:  {em.accuracy:.4f}")
        report_lines.append(f"Support:   {em.support}")
        report_lines.append("")
        report_lines.append("RELATION EXTRACTION METRICS")
        report_lines.append("-" * 40)
        rm = detailed_metrics.relation_metrics
        report_lines.append(f"Precision: {rm.precision:.4f}")
        report_lines.append(f"Recall:    {rm.recall:.4f}")
        report_lines.append(f"F1 Score:  {rm.f1:.4f}")
        report_lines.append(f"Accuracy:  {rm.accuracy:.4f}")
        report_lines.append(f"Support:   {rm.support}")
        report_lines.append("")
        report_lines.append("COMBINED METRICS")
        report_lines.append("-" * 40)
        cm = detailed_metrics.combined_metrics
        report_lines.append(f"Combined F1:        {cm['combined_f1']:.4f}")
        report_lines.append(f"Combined Precision: {cm['combined_precision']:.4f}")
        report_lines.append(f"Combined Recall:    {cm['combined_recall']:.4f}")
        report_lines.append("")
        if detailed_metrics.statistical_analysis:
            report_lines.append("STATISTICAL ANALYSIS")
            report_lines.append("-" * 40)
            sa = detailed_metrics.statistical_analysis
            entity_ci = sa.get('entity_f1_confidence_interval', (0, 0))
            relation_ci = sa.get('relation_f1_confidence_interval', (0, 0))
            report_lines.append(f"Entity F1 95% CI:   [{entity_ci[0]:.4f}, {entity_ci[1]:.4f}]")
            report_lines.append(f"Relation F1 95% CI: [{relation_ci[0]:.4f}, {relation_ci[1]:.4f}]")
            report_lines.append(f"Entity F1 Std:      {sa.get('entity_f1_std', 0):.4f}")
            report_lines.append(f"Relation F1 Std:    {sa.get('relation_f1_std', 0):.4f}")
            report_lines.append("")
        if detailed_metrics.error_analysis:
            report_lines.append("ERROR ANALYSIS SUMMARY")
            report_lines.append("-" * 40)
            entity_errors = detailed_metrics.error_analysis.get('entity_errors', {})
            relation_errors = detailed_metrics.error_analysis.get('relation_errors', {})
            report_lines.append(f"Entity Error Rate:   {entity_errors.get('error_rate', 0):.4f}")
            report_lines.append(f"Relation Error Rate: {relation_errors.get('error_rate', 0):.4f}")
            report_lines.append("")
            entity_common = entity_errors.get('most_common_errors', {})
            if entity_common:
                report_lines.append("Most Common Entity Errors:")
                for pattern, count in list(entity_common.items())[:5]:
                    report_lines.append(f"  {pattern}: {count}")
                report_lines.append("")
            relation_common = relation_errors.get('most_common_errors', {})
            if relation_common:
                report_lines.append("Most Common Relation Errors:")
                for pattern, count in list(relation_common.items())[:5]:
                    report_lines.append(f"  {pattern}: {count}")
                report_lines.append("")
        report_text = "\n".join(report_lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Saved evaluation report to {output_path}")
        return report_text

def calculate_macro_f1(per_class_f1_scores: List[float]) -> float:
    return np.mean(per_class_f1_scores)

def calculate_micro_f1(true_positives: List[int], 
                      false_positives: List[int], 
                      false_negatives: List[int]) -> float:
    total_tp = sum(true_positives)
    total_fp = sum(false_positives)
    total_fn = sum(false_negatives)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

if __name__ == "__main__":
    entity_labels = ['O', 'PhysicalObject', 'Activity', 'State', 'Property']
    relation_labels = ['no_relation', 'located_at', 'part_of', 'affects']
    calculator = MaintenanceMetricsCalculator(
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        evaluation_mode=EvaluationMode.STRICT,
        averaging_method=AveragingMethod.WEIGHTED
    )
    entity_predictions = [1, 2, 1, 3, 0, 2, 1]
    entity_targets = [1, 2, 0, 3, 0, 2, 2]
    relation_predictions = [0, 1, 2, 0, 1]
    relation_targets = [0, 1, 1, 0, 2]
    entity_metrics = calculator.calculate_entity_metrics(entity_predictions, entity_targets)
    print(f"Entity F1: {entity_metrics.f1:.4f}")
    relation_metrics = calculator.calculate_relation_metrics(relation_predictions, relation_targets)
    print(f"Relation F1: {relation_metrics.f1:.4f}")
    detailed_metrics = calculator.calculate_detailed_metrics(
        entity_predictions, entity_targets,
        relation_predictions, relation_targets
    )
    report = calculator.generate_evaluation_report(detailed_metrics)
    print(report)
