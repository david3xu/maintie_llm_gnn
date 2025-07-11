from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class MetricsCalculator:
    """Calculate evaluation metrics"""

    def __init__(self):
        """Initialize metrics calculator"""
        pass

    def calculate_f1_score(self, predictions: List[int], targets: List[int], average='macro') -> float:
        """Calculate F1 score"""
        return f1_score(targets, predictions, average=average, zero_division=0)

    def calculate_precision_recall(self, predictions: List[int], targets: List[int], average='macro') -> Tuple[float, float]:
        """Calculate precision and recall"""
        precision = precision_score(targets, predictions, average=average, zero_division=0)
        recall = recall_score(targets, predictions, average=average, zero_division=0)
        return precision, recall

    def calculate_entity_metrics(self, entity_preds: List[int], entity_targets: List[int]) -> Dict[str, float]:
        """Calculate entity-specific metrics"""
        f1 = self.calculate_f1_score(entity_preds, entity_targets)
        precision, recall = self.calculate_precision_recall(entity_preds, entity_targets)
        return {'f1': f1, 'precision': precision, 'recall': recall}

    def calculate_relation_metrics(self, relation_preds: List[int], relation_targets: List[int]) -> Dict[str, float]:
        """Calculate relation-specific metrics"""
        f1 = self.calculate_f1_score(relation_preds, relation_targets)
        precision, recall = self.calculate_precision_recall(relation_preds, relation_targets)
        return {'f1': f1, 'precision': precision, 'recall': recall}

    def calculate_confusion_matrix(self, predictions: List[int], targets: List[int]) -> np.ndarray:
        """Calculate confusion matrix"""
        # Ensure labels are consistent between predictions and targets
        labels = sorted(list(set(targets) | set(predictions)))
        return confusion_matrix(targets, predictions, labels=labels)

    def calculate_per_class_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics"""
        labels = sorted(list(set(targets)))
        per_class_f1 = f1_score(targets, predictions, labels=labels, average=None, zero_division=0)
        per_class_precision = precision_score(targets, predictions, labels=labels, average=None, zero_division=0)
        per_class_recall = recall_score(targets, predictions, labels=labels, average=None, zero_division=0)

        metrics = {}
        for i, label in enumerate(labels):
            metrics[str(label)] = {
                'f1': per_class_f1[i],
                'precision': per_class_precision[i],
                'recall': per_class_recall[i]
            }
        return metrics
