from typing import List, Dict, Optional, Tuple, Any
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

@dataclass
class DatasetStats:
    """Statistics for dataset analysis"""
    total_samples: int
    total_entities: int
    total_relations: int
    entity_type_counts: Dict[str, int]
    relation_type_counts: Dict[str, int]
    avg_entities_per_sample: float
    avg_relations_per_sample: float
    avg_tokens_per_sample: float
    hierarchy_depth_distribution: Dict[int, int]

class MaintIEDataLoader:
    """
    FIXED: Simple, reliable data loader for MaintIE JSON files.
    Handles REAL data format with hierarchical entities.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.validation_enabled = self.config.get('validation_enabled', True)
        self.validation_errors = []
        self.load_times = []
        self.logger.info("Initialized FIXED MaintIEDataLoader")

    def load_gold_corpus(self, file_path: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Loading gold corpus from {file_path}")
        return self._load_json_file(file_path, corpus_type='gold')

    def load_silver_corpus(self, file_path: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Loading silver corpus from {file_path}")
        return self._load_json_file(file_path, corpus_type='silver')

    def load_ontology(self, file_path: str) -> Dict[str, Any]:
        self.logger.info(f"Loading ontology from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ontology = json.load(f)
            entity_count = len(ontology.get('entity', []))
            relation_count = len(ontology.get('relation', []))
            self.logger.info(f"Loaded ontology: {entity_count} entities, {relation_count} relations")
            return ontology
        except Exception as e:
            self.logger.error(f"Failed to load ontology from {file_path}: {e}")
            raise

    def _load_json_file(self, file_path: str, corpus_type: str) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'documents' in data:
                data = data['documents']
            elif not isinstance(data, list):
                raise ValueError(f"Expected list or dict with 'documents', got {type(data)}")
            if self.validation_enabled:
                validated_data = []
                for i, sample in enumerate(data):
                    if self._validate_sample_fixed(sample, i):
                        validated_data.append(sample)
                    else:
                        self.logger.warning(f"Skipped invalid sample {i}")
                data = validated_data
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            self.logger.info(f"✅ Loaded {len(data)} {corpus_type} samples in {load_time:.2f}s")
            return data
        except Exception as e:
            self.logger.error(f"❌ Failed to load {corpus_type} corpus: {e}")
            raise

    def _validate_sample_fixed(self, sample: Dict[str, Any], index: int) -> bool:
        required_fields = ['text', 'entities', 'relations']
        for field in required_fields:
            if field not in sample:
                error = f"Sample {index}: Missing required field '{field}'"
                self.validation_errors.append(error)
                return False
        if not isinstance(sample['text'], str) or not sample['text'].strip():
            self.validation_errors.append(f"Sample {index}: Invalid or empty text")
            return False
        for j, entity in enumerate(sample['entities']):
            required_entity_fields = ['start', 'end', 'type']
            for field in required_entity_fields:
                if field not in entity:
                    self.validation_errors.append(f"Sample {index}, entity {j}: Missing '{field}'")
                    return False
            entity_type = entity['type']
            if not isinstance(entity_type, str):
                self.validation_errors.append(f"Sample {index}, entity {j}: Invalid entity type")
                return False
            if entity['start'] >= entity['end']:
                self.validation_errors.append(f"Sample {index}, entity {j}: Invalid span")
                return False
        entity_count = len(sample['entities'])
        for j, relation in enumerate(sample['relations']):
            required_relation_fields = ['head', 'tail', 'type']
            for field in required_relation_fields:
                if field not in relation:
                    self.validation_errors.append(f"Sample {index}, relation {j}: Missing '{field}'")
                    return False
            head_idx = relation['head']
            tail_idx = relation['tail']
            if not (0 <= head_idx < entity_count and 0 <= tail_idx < entity_count):
                self.validation_errors.append(f"Sample {index}, relation {j}: Invalid entity indices")
                return False
        return True

class MaintIEEntityProcessor:
    """
    FIXED: Handles hierarchical entity types and complexity mapping
    """
    def __init__(self, ontology_path: str):
        self.logger = logging.getLogger(__name__)
        self.ontology_path = ontology_path
        loader = MaintIEDataLoader()
        self.ontology = loader.load_ontology(ontology_path)
        self.entity_hierarchy = self._build_entity_hierarchy_fixed()
        self.complexity_mappings = self._build_complexity_mappings_fixed()
        self.logger.info(f"✅ FIXED entity processor: {len(self.entity_hierarchy)} entity types")

    def extract_entity_hierarchy(self, entity_type: str) -> List[str]:
        return entity_type.split('/')

    def map_entity_to_complexity(self, entity_type: str, complexity_level: str) -> str:
        hierarchy = self.extract_entity_hierarchy(entity_type)
        depth_mapping = {
            'FG-0': 1,
            'FG-1': 2,
            'FG-2': 3,
            'FG-3': None
        }
        max_depth = depth_mapping.get(complexity_level)
        if max_depth is None:
            return entity_type
        truncated_parts = hierarchy[:min(max_depth, len(hierarchy))]
        return '/'.join(truncated_parts)

    def get_entity_classes_for_complexity(self, complexity_level: str) -> int:
        return self.complexity_mappings.get(complexity_level, {}).get('entity_count', 0)

    def _build_entity_hierarchy_fixed(self) -> Dict[str, Dict[str, Any]]:
        hierarchy = {}
        entities = self.ontology.get('entity', [])
        for entity in entities:
            if isinstance(entity, dict):
                entity_id = entity.get('id', entity.get('name', ''))
                entity_path = entity.get('path', entity.get('type', entity_id))
                hierarchy[entity_path] = {
                    'id': entity_id,
                    'depth': len(entity_path.split('/')),
                    'parent': '/'.join(entity_path.split('/')[:-1]) if '/' in entity_path else None
                }
        return hierarchy

    def _build_complexity_mappings_fixed(self) -> Dict[str, Dict[str, Any]]:
        mappings = {}
        entities_by_depth = defaultdict(set)
        for entity_path, entity_info in self.entity_hierarchy.items():
            depth = entity_info['depth']
            entities_by_depth[depth].add(entity_path)
        mappings['FG-0'] = {
            'entity_count': len(entities_by_depth[1]) if entities_by_depth[1] else 5,
            'entity_types': sorted(list(entities_by_depth[1])) if entities_by_depth[1] else
                           ['PhysicalObject', 'Activity', 'Process', 'State', 'Property'],
            'max_depth': 1,
            'description': 'Root level entities only'
        }
        mappings['FG-1'] = {
            'entity_count': len(entities_by_depth[1] | entities_by_depth[2]) if entities_by_depth[2] else 20,
            'entity_types': sorted(list(entities_by_depth[1] | entities_by_depth[2])),
            'max_depth': 2,
            'description': 'Two-level hierarchy'
        }
        mappings['FG-2'] = {
            'entity_count': sum(len(entities_by_depth[i]) for i in range(1, 4)) if entities_by_depth[3] else 80,
            'entity_types': sorted(list(entities_by_depth[1] | entities_by_depth[2] | entities_by_depth[3])),
            'max_depth': 3,
            'description': 'Three-level hierarchy'
        }
        mappings['FG-3'] = {
            'entity_count': len(self.entity_hierarchy),
            'entity_types': sorted(list(self.entity_hierarchy.keys())),
            'max_depth': None,
            'description': 'Full entity hierarchy'
        }
        for level, mapping in mappings.items():
            self.logger.info(f"{level}: {mapping['entity_count']} entity classes")
        return mappings

class MaintIEDataset(Dataset):
    """
    FIXED: PyTorch Dataset for MaintIE data with proper complexity handling
    """
    def __init__(self,
                 data: List[Dict[str, Any]],
                 entity_processor: MaintIEEntityProcessor,
                 complexity_level: str = 'FG-3',
                 transform: Optional[callable] = None):
        self.data = data
        self.entity_processor = entity_processor
        self.complexity_level = complexity_level
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"✅ FIXED dataset: {len(self.data)} samples at {complexity_level}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.data)}")
        sample = self.data[idx].copy()
        mapped_entities = []
        for entity in sample['entities']:
            mapped_type = self.entity_processor.map_entity_to_complexity(
                entity['type'], self.complexity_level
            )
            mapped_entity = entity.copy()
            mapped_entity['type'] = mapped_type
            mapped_entity['original_type'] = entity['type']
            mapped_entities.append(mapped_entity)
        sample['entities'] = mapped_entities
        sample['complexity_level'] = self.complexity_level
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_statistics(self) -> DatasetStats:
        if not self.data:
            return DatasetStats(0, 0, 0, {}, {}, 0.0, 0.0, 0.0, {})
        total_entities = 0
        total_relations = 0
        total_tokens = 0
        entity_type_counts = Counter()
        relation_type_counts = Counter()
        hierarchy_depths = Counter()
        for sample in self.data:
            entities = sample.get('entities', [])
            relations = sample.get('relations', [])
            tokens = sample.get('tokens', sample.get('text', '').split())
            total_entities += len(entities)
            total_relations += len(relations)
            total_tokens += len(tokens)
            for entity in entities:
                mapped_type = self.entity_processor.map_entity_to_complexity(
                    entity['type'], self.complexity_level
                )
                entity_type_counts[mapped_type] += 1
                depth = len(entity['type'].split('/'))
                hierarchy_depths[depth] += 1
            for relation in relations:
                relation_type_counts[relation['type']] += 1
        num_samples = len(self.data)
        avg_entities = total_entities / num_samples
        avg_relations = total_relations / num_samples
        avg_tokens = total_tokens / num_samples
        return DatasetStats(
            total_samples=num_samples,
            total_entities=total_entities,
            total_relations=total_relations,
            entity_type_counts=dict(entity_type_counts),
            relation_type_counts=dict(relation_type_counts),
            avg_entities_per_sample=avg_entities,
            avg_relations_per_sample=avg_relations,
            avg_tokens_per_sample=avg_tokens,
            hierarchy_depth_distribution=dict(hierarchy_depths)
        )
