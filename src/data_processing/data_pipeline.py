from typing import List, Dict, Optional, Tuple, Union, Any, Iterator, Set
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from pathlib import Path
import logging
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from dataclasses import dataclass
import hashlib
import time
import warnings
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Import custom modules
from ..utils.file_handlers import JSONHandler, PickleHandler
from ..core.config import ConfigManager
from .data_splitter import MaintIEDataSplitter
from .data_validator import DataValidator

@dataclass
class DatasetStatistics:
    """Container for dataset statistics."""
    total_samples: int
    total_entities: int
    total_relations: int
    avg_entities_per_sample: float
    avg_relations_per_sample: float
    entity_type_distribution: Dict[str, int]
    relation_type_distribution: Dict[str, int]
    text_length_stats: Dict[str, float]
    complexity_distribution: Dict[str, int]
    quality_score: float

@dataclass
class ProcessingResult:
    """Container for processing results."""
    processed_data: List[Dict[str, Any]]
    statistics: DatasetStatistics
    processing_time: float
    validation_errors: List[str]
    skipped_samples: int

class MaintIEDataProcessor:
    """
    Comprehensive data processing pipeline for MaintIE corpora.
    Handles silver and gold corpus processing, validation, curriculum learning
    preparation, and batch generation for training and evaluation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = ConfigManager(config)
        self.data_splitter = MaintIEDataSplitter(config)
        self.data_validator = DataValidator(config)
        self.max_sequence_length = self.config.get('data.max_sequence_length', 512)
        self.validation_enabled = self.config.get('data.validation_enabled', True)
        self.quality_threshold = self.config.get('data.quality_threshold', 0.8)
        self.complexity_mappings = self._load_complexity_mappings()
        self.processing_cache = {}
        self.processing_stats = {
            'total_processed': 0,
            'total_skipped': 0,
            'processing_time': 0.0
        }
        self.logger.info("Initialized MaintIEDataProcessor")

    def _load_complexity_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {
            'FG-0': {
                'entity_classes': 1,
                'max_entity_types': {'PhysicalObject'},
                'description': 'Single entity type classification'
            },
            'FG-1': {
                'entity_classes': 5,
                'max_entity_types': {
                    'PhysicalObject', 'Activity', 'Process', 'State', 'Property'
                },
                'description': 'Basic entity type classification'
            },
            'FG-2': {
                'entity_classes': 32,
                'max_entity_types': set(),
                'description': 'Intermediate entity type classification'
            },
            'FG-3': {
                'entity_classes': 224,
                'max_entity_types': set(),
                'description': 'Full hierarchy entity type classification'
            }
        }

    def process_silver_corpus(self, silver_corpus_path: str, output_dir: Optional[str] = None, chunk_size: int = 1000) -> ProcessingResult:
        self.logger.info(f"Processing silver corpus from {silver_corpus_path}")
        start_time = time.time()
        raw_data = self._load_corpus_data(silver_corpus_path)
        processed_data = []
        validation_errors = []
        skipped_count = 0
        for chunk_start in range(0, len(raw_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(raw_data))
            chunk_data = raw_data[chunk_start:chunk_end]
            chunk_result = self._process_data_chunk(chunk_data, corpus_type='silver', chunk_id=chunk_start // chunk_size)
            processed_data.extend(chunk_result.processed_data)
            validation_errors.extend(chunk_result.validation_errors)
            skipped_count += chunk_result.skipped_samples
            self.logger.debug(f"Processed chunk {chunk_start//chunk_size + 1}: {len(chunk_result.processed_data)} samples")
        statistics = self._calculate_dataset_statistics(processed_data)
        if output_dir:
            self._save_processed_data(processed_data, output_dir, 'silver_processed')
        processing_time = time.time() - start_time
        self.processing_stats['total_processed'] += len(processed_data)
        self.processing_stats['total_skipped'] += skipped_count
        self.processing_stats['processing_time'] += processing_time
        result = ProcessingResult(
            processed_data=processed_data,
            statistics=statistics,
            processing_time=processing_time,
            validation_errors=validation_errors,
            skipped_samples=skipped_count
        )
        self.logger.info(f"Silver corpus processing completed: {len(processed_data)} samples in {processing_time:.2f}s")
        return result

    def process_gold_corpus(self, gold_corpus_path: str, output_dir: Optional[str] = None, strict_validation: bool = True) -> ProcessingResult:
        self.logger.info(f"Processing gold corpus from {gold_corpus_path}")
        start_time = time.time()
        raw_data = self._load_corpus_data(gold_corpus_path)
        processed_data = []
        validation_errors = []
        skipped_count = 0
        for i, sample in enumerate(raw_data):
            try:
                processed_sample = self._process_single_sample(sample, corpus_type='gold', sample_id=i, strict_validation=strict_validation)
                if processed_sample:
                    processed_data.append(processed_sample)
                else:
                    skipped_count += 1
            except Exception as e:
                validation_errors.append(f"Sample {i}: {str(e)}")
                skipped_count += 1
                if strict_validation:
                    self.logger.warning(f"Skipping gold sample {i} due to validation error: {e}")
        statistics = self._calculate_dataset_statistics(processed_data)
        if statistics.quality_score < self.quality_threshold:
            self.logger.warning(f"Gold corpus quality score {statistics.quality_score:.3f} below threshold {self.quality_threshold}")
        if output_dir:
            self._save_processed_data(processed_data, output_dir, 'gold_processed')
        processing_time = time.time() - start_time
        result = ProcessingResult(
            processed_data=processed_data,
            statistics=statistics,
            processing_time=processing_time,
            validation_errors=validation_errors,
            skipped_samples=skipped_count
        )
        self.logger.info(f"Gold corpus processing completed: {len(processed_data)} samples in {processing_time:.2f}s")
        return result

    def _load_corpus_data(self, file_path: str) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {file_path}")
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.suffix.lower() == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        if isinstance(data, dict):
            if 'samples' in data:
                return data['samples']
            elif 'data' in data:
                return data['data']
            else:
                return [{'id': k, **v} for k, v in data.items()]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected data format in {file_path}")

    def _process_data_chunk(self, chunk_data: List[Dict[str, Any]], corpus_type: str, chunk_id: int) -> ProcessingResult:
        processed_samples = []
        validation_errors = []
        skipped_count = 0
        for i, sample in enumerate(chunk_data):
            try:
                processed_sample = self._process_single_sample(sample, corpus_type=corpus_type, sample_id=f"{chunk_id}_{i}")
                if processed_sample:
                    processed_samples.append(processed_sample)
                else:
                    skipped_count += 1
            except Exception as e:
                validation_errors.append(f"Chunk {chunk_id}, Sample {i}: {str(e)}")
                skipped_count += 1
        return ProcessingResult(
            processed_data=processed_samples,
            statistics=None,
            processing_time=0.0,
            validation_errors=validation_errors,
            skipped_samples=skipped_count
        )

    def _process_single_sample(self, sample: Dict[str, Any], corpus_type: str, sample_id: Any, strict_validation: bool = None) -> Optional[Dict[str, Any]]:
        if strict_validation is None:
            strict_validation = corpus_type == 'gold'
        if self.validation_enabled:
            validation_result = self.data_validator.validate_sample(sample, corpus_type)
            if not validation_result.is_valid:
                if strict_validation:
                    raise ValueError(f"Sample validation failed: {validation_result.errors}")
                else:
                    self.logger.debug(f"Sample {sample_id} has validation warnings: {validation_result.warnings}")
        texts = self._extract_texts(sample)
        if not texts:
            return None
        processed_texts = []
        for text in texts:
            processed_text = self._process_text(text)
            if processed_text:
                processed_texts.append(processed_text)
        if not processed_texts:
            return None
        annotations = self._extract_annotations(sample)
        processed_annotations = self._process_annotations(annotations, len(processed_texts))
        processed_sample = {
            'sample_id': sample_id,
            'corpus_type': corpus_type,
            'texts': processed_texts,
            'annotations': processed_annotations,
            'original_sample': sample,
            'processing_metadata': {
                'processed_at': datetime.now().isoformat(),
                'text_count': len(processed_texts),
                'entity_count': len(processed_annotations.get('entities', [])),
                'relation_count': len(processed_annotations.get('relations', []))
            }
        }
        return processed_sample

    def _extract_texts(self, sample: Dict[str, Any]) -> List[str]:
        texts = []
        text_fields = ['text', 'texts', 'content', 'document', 'sentence']
        for field in text_fields:
            if field in sample:
                text_value = sample[field]
                if isinstance(text_value, str):
                    texts.append(text_value)
                elif isinstance(text_value, list):
                    texts.extend([str(t) for t in text_value if t])
                break
        return texts

    def _process_text(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return None
        text = text.strip()
        text = ' '.join(text.split())
        if len(text) > self.max_sequence_length * 4:
            self.logger.debug(f"Text too long, truncating: {len(text)} chars")
            text = text[:self.max_sequence_length * 4]
        if len(text) < 10:
            return None
        return text

    def _extract_annotations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        annotations = {}
        if 'entities' in sample:
            annotations['entities'] = sample['entities']
        elif 'entity_annotations' in sample:
            annotations['entities'] = sample['entity_annotations']
        else:
            annotations['entities'] = []
        if 'relations' in sample:
            annotations['relations'] = sample['relations']
        elif 'relation_annotations' in sample:
            annotations['relations'] = sample['relation_annotations']
        else:
            annotations['relations'] = []
        if 'labels' in sample:
            annotations['labels'] = sample['labels']
        return annotations

    def _process_annotations(self, annotations: Dict[str, Any], text_count: int) -> Dict[str, Any]:
        processed_annotations = {
            'entities': [],
            'relations': [],
            'labels': []
        }
        for entity in annotations.get('entities', []):
            processed_entity = self._process_entity_annotation(entity, text_count)
            if processed_entity:
                processed_annotations['entities'].append(processed_entity)
        for relation in annotations.get('relations', []):
            processed_relation = self._process_relation_annotation(relation, text_count)
            if processed_relation:
                processed_annotations['relations'].append(processed_relation)
        processed_annotations['labels'] = annotations.get('labels', [])
        return processed_annotations

    def _process_entity_annotation(self, entity: Dict[str, Any], text_count: int) -> Optional[Dict[str, Any]]:
        required_fields = ['type']
        for field in required_fields:
            if field not in entity:
                return None
        processed_entity = {
            'type': entity['type'],
            'text_id': entity.get('text_id', 0),
            'span': entity.get('span', [0, 0]),
            'text': entity.get('text', ''),
            'confidence': entity.get('confidence', 1.0)
        }
        if processed_entity['text_id'] >= text_count:
            processed_entity['text_id'] = 0
        return processed_entity

    def _process_relation_annotation(self, relation: Dict[str, Any], text_count: int) -> Optional[Dict[str, Any]]:
        required_fields = ['type', 'source', 'target']
        for field in required_fields:
            if field not in relation:
                return None
        processed_relation = {
            'type': relation['type'],
            'source': relation['source'],
            'target': relation['target'],
            'confidence': relation.get('confidence', 1.0)
        }
        if (processed_relation['source'] >= text_count or processed_relation['target'] >= text_count):
            return None
        return processed_relation

    def prepare_curriculum_data(self, processed_data: List[Dict[str, Any]], complexity_level: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Preparing curriculum data for {complexity_level}")
        if complexity_level not in self.complexity_mappings:
            raise ValueError(f"Unknown complexity level: {complexity_level}")
        complexity_config = self.complexity_mappings[complexity_level]
        curriculum_data = []
        for sample in processed_data:
            adapted_sample = self._adapt_sample_for_complexity(sample, complexity_config)
            if adapted_sample:
                curriculum_data.append(adapted_sample)
        self.logger.info(f"Prepared {len(curriculum_data)} samples for {complexity_level}")
        return curriculum_data

    def _adapt_sample_for_complexity(self, sample: Dict[str, Any], complexity_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        adapted_sample = sample.copy()
        adapted_annotations = sample['annotations'].copy()
        if complexity_config['entity_classes'] == 1:
            adapted_entities = []
            for entity in adapted_annotations['entities']:
                adapted_entity = entity.copy()
                adapted_entity['type'] = 'Entity'
                adapted_entities.append(adapted_entity)
            adapted_annotations['entities'] = adapted_entities
        elif complexity_config['max_entity_types']:
            allowed_types = complexity_config['max_entity_types']
            adapted_entities = []
            for entity in adapted_annotations['entities']:
                entity_type = entity['type']
                simplified_type = self._map_to_complexity_type(entity_type, allowed_types)
                if simplified_type:
                    adapted_entity = entity.copy()
                    adapted_entity['type'] = simplified_type
                    adapted_entities.append(adapted_entity)
            adapted_annotations['entities'] = adapted_entities
        adapted_sample['annotations'] = adapted_annotations
        adapted_sample['complexity_level'] = complexity_config
        if not adapted_annotations['entities']:
            return None
        return adapted_sample

    def _map_to_complexity_type(self, entity_type: str, allowed_types: Set[str]) -> Optional[str]:
        if entity_type in allowed_types:
            return entity_type
        type_mappings = {
            'PhysicalObject/SensingObject': 'PhysicalObject',
            'PhysicalObject/DrivingObject': 'PhysicalObject',
            'PhysicalObject/ProcessingObject': 'PhysicalObject',
            'Activity/MaintenanceActivity': 'Activity',
            'Activity/InspectionActivity': 'Activity',
            'Process/MaintenanceProcess': 'Process',
            'State/OperationalState': 'State',
            'Property/PhysicalProperty': 'Property'
        }
        for source_type, target_type in type_mappings.items():
            if entity_type.startswith(source_type) and target_type in allowed_types:
                return target_type
        if 'PhysicalObject' in allowed_types:
            return 'PhysicalObject'
        return None

    def create_data_splits(self, processed_data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, stratify_by: Optional[str] = None, random_state: int = 42) -> Dict[str, List[Dict[str, Any]]]:
        self.logger.info(f"Creating data splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        splits = self.data_splitter.create_stratified_splits(
            data=processed_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_by=stratify_by,
            random_state=random_state
        )
        self.logger.info(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits

    def _calculate_dataset_statistics(self, data: List[Dict[str, Any]]) -> DatasetStatistics:
        if not data:
            return DatasetStatistics(
                total_samples=0, total_entities=0, total_relations=0,
                avg_entities_per_sample=0, avg_relations_per_sample=0,
                entity_type_distribution={}, relation_type_distribution={},
                text_length_stats={}, complexity_distribution={}, quality_score=0.0
            )
        total_samples = len(data)
        total_entities = 0
        total_relations = 0
        entity_types = Counter()
        relation_types = Counter()
        text_lengths = []
        complexity_levels = Counter()
        quality_issues = 0
        for sample in data:
            annotations = sample.get('annotations', {})
            texts = sample.get('texts', [])
            entities = annotations.get('entities', [])
            relations = annotations.get('relations', [])
            total_entities += len(entities)
            total_relations += len(relations)
            for entity in entities:
                entity_types[entity.get('type', 'unknown')] += 1
            for relation in relations:
                relation_types[relation.get('type', 'unknown')] += 1
            for text in texts:
                text_lengths.append(len(text.split()) if text else 0)
            complexity = sample.get('complexity_level', {}).get('entity_classes', 224)
            complexity_levels[f"FG-{complexity}"] += 1
            if not texts or not entities:
                quality_issues += 1
        avg_entities = total_entities / total_samples
        avg_relations = total_relations / total_samples
        text_length_stats = {
            'mean': np.mean(text_lengths) if text_lengths else 0,
            'std': np.std(text_lengths) if text_lengths else 0,
            'min': np.min(text_lengths) if text_lengths else 0,
            'max': np.max(text_lengths) if text_lengths else 0,
            'median': np.median(text_lengths) if text_lengths else 0
        }
        quality_score = 1.0 - (quality_issues / total_samples)
        return DatasetStatistics(
            total_samples=total_samples,
            total_entities=total_entities,
            total_relations=total_relations,
            avg_entities_per_sample=avg_entities,
            avg_relations_per_sample=avg_relations,
            entity_type_distribution=dict(entity_types.most_common(50)),
            relation_type_distribution=dict(relation_types.most_common()),
            text_length_stats=text_length_stats,
            complexity_distribution=dict(complexity_levels),
            quality_score=quality_score
        )

    def _save_processed_data(self, data: List[Dict[str, Any]], output_dir: str, filename_prefix: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = output_path / f"{filename_prefix}_{timestamp}.json"
        JSONHandler.save(data, str(json_file))
        pickle_file = output_path / f"{filename_prefix}_{timestamp}.pkl"
        PickleHandler.save(data, str(pickle_file))
        self.logger.info(f"Saved processed data to {output_path}")

    def batch_process_directory(self, input_dir: str, output_dir: str, file_pattern: str = "*.json") -> Dict[str, ProcessingResult]:
        input_path = Path(input_dir)
        results = {}
        for file_path in input_path.glob(file_pattern):
            self.logger.info(f"Processing file: {file_path}")
            try:
                if 'silver' in file_path.name.lower():
                    result = self.process_silver_corpus(str(file_path), output_dir)
                elif 'gold' in file_path.name.lower():
                    result = self.process_gold_corpus(str(file_path), output_dir)
                else:
                    result = self.process_silver_corpus(str(file_path), output_dir)
                results[file_path.name] = result
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                results[file_path.name] = ProcessingResult(
                    processed_data=[],
                    statistics=None,
                    processing_time=0.0,
                    validation_errors=[str(e)],
                    skipped_samples=0
                )
        return results

    def get_processing_summary(self) -> Dict[str, Any]:
        return {
            'total_processed': self.processing_stats['total_processed'],
            'total_skipped': self.processing_stats['total_skipped'],
            'total_processing_time': self.processing_stats['processing_time'],
            'average_processing_rate': (
                self.processing_stats['total_processed'] /
                self.processing_stats['processing_time']
                if self.processing_stats['processing_time'] > 0 else 0
            ),
            'cache_size': len(self.processing_cache)
        }

class MaintIEDataset(Dataset):
    """
    PyTorch Dataset class for MaintIE processed data.
    Provides efficient data loading and batching for training and evaluation.
    """
    def __init__(self, processed_data: List[Dict[str, Any]], transform: Optional[callable] = None):
        self.data = processed_data
        self.transform = transform
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def get_statistics(self) -> DatasetStatistics:
        processor = MaintIEDataProcessor({})
        return processor._calculate_dataset_statistics(self.data)

# Utility functions
def create_maintie_datasets(config: Dict[str, Any]) -> Dict[str, MaintIEDataset]:
    processor = MaintIEDataProcessor(config)
    silver_path = config.get('data.silver_corpus_path')
    if silver_path:
        silver_result = processor.process_silver_corpus(silver_path)
        splits = processor.create_data_splits(
            silver_result.processed_data,
            train_ratio=config.get('data.train_split', 0.8),
            val_ratio=config.get('data.val_split', 0.2),
            test_ratio=0.0
        )
        return {
            'train': MaintIEDataset(splits['train']),
            'val': MaintIEDataset(splits['val'])
        }
    return {}

def create_gold_validation_dataset(config: Dict[str, Any]) -> MaintIEDataset:
    processor = MaintIEDataProcessor(config)
    gold_path = config.get('data.gold_corpus_path')
    if not gold_path:
        raise ValueError("Gold corpus path not specified in configuration")
    gold_result = processor.process_gold_corpus(gold_path, strict_validation=True)
    return MaintIEDataset(gold_result.processed_data)

# Example usage
if __name__ == "__main__":
    config = {
        'data': {
            'max_sequence_length': 512,
            'validation_enabled': True,
            'quality_threshold': 0.8,
            'silver_corpus_path': 'data/raw/silver_release.json',
            'gold_corpus_path': 'data/raw/gold_release.json'
        }
    }
    processor = MaintIEDataProcessor(config)
    silver_result = processor.process_silver_corpus(
        'data/raw/silver_release.json',
        output_dir='data/processed'
    )
    print(f"Processed {silver_result.statistics.total_samples} silver samples")
    print(f"Processing time: {silver_result.processing_time:.2f}s")
    print(f"Quality score: {silver_result.statistics.quality_score:.3f}")
    curriculum_data = processor.prepare_curriculum_data(
        silver_result.processed_data, 
        'FG-1'
    )
    print(f"Prepared {len(curriculum_data)} samples for FG-1 complexity")
    splits = processor.create_data_splits(curriculum_data)
    print(f"Data splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
