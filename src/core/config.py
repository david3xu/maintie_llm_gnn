from typing import Any, Dict, List, Optional, Union, Type, Callable
import yaml
import json
import os
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import copy
from datetime import datetime
import hashlib
import warnings

class Environment(Enum):
    DEVELOPMENT = "development"
    TRAINING = "training"
    PRODUCTION = "production"
    TESTING = "testing"

class ConfigValidationError(Exception):
    pass

@dataclass
class ConfigSchema:
    name: str
    required: bool = True
    type_hint: Type = str
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    choices: Optional[List[Any]] = None
    range_min: Optional[Union[int, float]] = None
    range_max: Optional[Union[int, float]] = None

class ConfigManager:
    def __init__(self, config_data: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None, environment: Environment = Environment.DEVELOPMENT):
        self.logger = logging.getLogger(__name__)
        self.environment = environment
        self._config = {}
        self._schema = {}
        self._validation_enabled = True
        self._change_history = []
        if config_data:
            self._config = copy.deepcopy(config_data)
        elif config_file:
            self._load_from_file(config_file)
        else:
            self._load_default_config()
        self._apply_environment_config()
        self._initialize_schema()
        self._validate_config()
        self.logger.info(f"ConfigManager initialized for {environment.value} environment")

    def _load_from_file(self, config_file: str):
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        file_extension = config_path.suffix.lower()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if file_extension in ['.yaml', '.yml']:
                    self._config = yaml.safe_load(f)
                elif file_extension == '.json':
                    self._config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_extension}")
            self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            raise ConfigValidationError(f"Failed to load configuration from {config_file}: {e}")

    def _load_default_config(self):
        self._config = {
            'project': {
                'name': 'maintie_llm_gnn',
                'version': '1.0.0',
                'description': 'MaintIE LLM-GNN Hybrid Information Extraction'
            },
            'environment': self.environment.value,
            'data': {
                'raw_data_path': 'data/raw/',
                'processed_data_path': 'data/processed/',
                'gold_corpus_path': 'data/raw/gold_release.json',
                'silver_corpus_path': 'data/raw/silver_release.json',
                'ontology_path': 'data/raw/scheme.json',
                'train_split': 0.8,
                'val_split': 0.2,
                'max_sequence_length': 512
            },
            'model': {
                'llm_embedder': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'embedding_dim': 384,
                    'domain_features_dim': 32,
                    'combined_features_dim': 416,
                    'cache_embeddings': True,
                    'feature_config': {
                        'normalize_features': True,
                        'use_tfidf_features': True,
                        'tfidf_max_features': 100
                    }
                },
                'gnn': {
                    'type': 'GAT',
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'num_heads': 8,
                    'dropout': 0.2
                },
                'num_entity_classes': 224,
                'num_relation_classes': 6,
                'use_domain_constraints': True
            },
            'graph': {
                'similarity_threshold': 0.75,
                'k_neighbors': 15,
                'max_edges_per_node': 50,
                'edge_types': [
                    'semantic_similarity',
                    'entity_cooccurrence',
                    'equipment_hierarchy',
                    'procedure_similarity'
                ],
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
                }
            },
            'training': {
                'batch_size': 8,
                'num_epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'early_stopping_patience': 10,
                'gradient_clipping': 1.0,
                'checkpoint_frequency': 10,
                'log_frequency': 50,
                'num_workers': 0,
                'use_mixed_precision': True,
                'entity_loss_weight': 1.0,
                'relation_loss_weight': 1.0,
                'use_focal_loss': False,
                'optimizer': {
                    'type': 'AdamW',
                    'learning_rate': 0.001,
                    'weight_decay': 0.0001,
                    'betas': [0.9, 0.999]
                },
                'scheduler': {
                    'type': 'ReduceLROnPlateau',
                    'factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-6
                }
            },
            'evaluation': {
                'bootstrap_samples': 1000,
                'confidence_level': 0.95,
                'complexity_levels': ['FG-0', 'FG-1', 'FG-2', 'FG-3'],
                'metrics': ['precision', 'recall', 'f1_score', 'accuracy']
            },
            'curriculum_learning': {
                'enabled': False,
                'stages': [
                    {'complexity': 'FG-0', 'epochs': 10, 'entity_classes': 1},
                    {'complexity': 'FG-1', 'epochs': 20, 'entity_classes': 5},
                    {'complexity': 'FG-2', 'epochs': 30, 'entity_classes': 32},
                    {'complexity': 'FG-3', 'epochs': 40, 'entity_classes': 224}
                ]
            },
            'paths': {
                'data_dir': 'data/',
                'models_dir': 'models/',
                'results_dir': 'results/',
                'logs_dir': 'logs/',
                'output_dir': 'results/',
                'evaluation_output': 'results/evaluation',
                'checkpoints_dir': 'models/checkpoints'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_handler': True,
                'console_handler': True,
                'log_file': 'logs/maintie.log'
            },
            'azure': {
                'subscription_id': None,
                'resource_group': None,
                'workspace_name': None,
                'compute_target': None,
                'experiment_name': 'maintie-llm-gnn',
                'tags': {
                    'project': 'maintie',
                    'framework': 'pytorch',
                    'model_type': 'llm-gnn-hybrid'
                }
            },
            'deployment': {
                'model_name': 'maintie-llm-gnn',
                'version': '1.0.0',
                'description': 'MaintIE LLM-GNN model for maintenance information extraction',
                'inference_config': {
                    'batch_size': 1,
                    'max_batch_size': 32,
                    'timeout_ms': 30000
                },
                'scaling': {
                    'min_instances': 1,
                    'max_instances': 10,
                    'target_utilization': 70
                }
            }
        }

    def _apply_environment_config(self):
        env_overrides = self._get_environment_overrides()
        if env_overrides:
            self._deep_update(self._config, env_overrides)
            self.logger.info(f"Applied {self.environment.value} environment overrides")

    def _get_environment_overrides(self) -> Dict[str, Any]:
        overrides = {}
        if self.environment == Environment.DEVELOPMENT:
            overrides = {
                'training': {
                    'batch_size': 4,
                    'num_epochs': 10,
                    'checkpoint_frequency': 5,
                    'log_frequency': 10
                },
                'logging': {
                    'level': 'DEBUG'
                },
                'evaluation': {
                    'bootstrap_samples': 100
                }
            }
        elif self.environment == Environment.TRAINING:
            overrides = {
                'training': {
                    'batch_size': 16,
                    'num_epochs': 100,
                    'use_mixed_precision': True,
                    'num_workers': 4
                },
                'logging': {
                    'level': 'INFO'
                },
                'evaluation': {
                    'bootstrap_samples': 1000
                }
            }
        elif self.environment == Environment.PRODUCTION:
            overrides = {
                'training': {
                    'batch_size': 32,
                    'use_mixed_precision': True,
                    'num_workers': 8
                },
                'logging': {
                    'level': 'WARNING',
                    'console_handler': False
                },
                'deployment': {
                    'scaling': {
                        'min_instances': 2,
                        'max_instances': 20
                    }
                }
            }
        elif self.environment == Environment.TESTING:
            overrides = {
                'training': {
                    'batch_size': 2,
                    'num_epochs': 2,
                    'checkpoint_frequency': 1
                },
                'evaluation': {
                    'bootstrap_samples': 10
                },
                'logging': {
                    'level': 'ERROR'
                }
            }
        return overrides

    def _initialize_schema(self):
        self._schema = {
            'project.name': ConfigSchema(name='project.name', type_hint=str, description='Project name'),
            'project.version': ConfigSchema(name='project.version', type_hint=str, description='Project version'),
            'data.train_split': ConfigSchema(name='data.train_split', type_hint=float, range_min=0.0, range_max=1.0, description='Training data split ratio'),
            'data.val_split': ConfigSchema(name='data.val_split', type_hint=float, range_min=0.0, range_max=1.0, description='Validation data split ratio'),
            'model.llm_embedder.embedding_dim': ConfigSchema(name='model.llm_embedder.embedding_dim', type_hint=int, range_min=1, range_max=2048, description='LLM embedding dimension'),
            'model.gnn.hidden_dim': ConfigSchema(name='model.gnn.hidden_dim', type_hint=int, range_min=32, range_max=1024, description='GNN hidden dimension'),
            'model.gnn.type': ConfigSchema(name='model.gnn.type', type_hint=str, choices=['GAT', 'GCN', 'GraphSAGE'], description='GNN architecture type'),
            'model.gnn.num_layers': ConfigSchema(name='model.gnn.num_layers', type_hint=int, range_min=1, range_max=10, description='Number of GNN layers'),
            'model.gnn.dropout': ConfigSchema(name='model.gnn.dropout', type_hint=float, range_min=0.0, range_max=0.9, description='GNN dropout rate'),
            'training.batch_size': ConfigSchema(name='training.batch_size', type_hint=int, range_min=1, range_max=256, description='Training batch size'),
            'training.learning_rate': ConfigSchema(name='training.learning_rate', type_hint=float, range_min=1e-6, range_max=1.0, description='Learning rate'),
            'training.num_epochs': ConfigSchema(name='training.num_epochs', type_hint=int, range_min=1, range_max=1000, description='Number of training epochs'),
            'graph.similarity_threshold': ConfigSchema(name='graph.similarity_threshold', type_hint=float, range_min=0.0, range_max=1.0, description='Graph similarity threshold'),
            'graph.k_neighbors': ConfigSchema(name='graph.k_neighbors', type_hint=int, range_min=1, range_max=100, description='Number of nearest neighbors')
        }

    def _validate_config(self):
        if not self._validation_enabled:
            return
        errors = []
        for schema_key, schema in self._schema.items():
            try:
                value = self.get(schema_key)
                if schema.required and value is None:
                    errors.append(f"Required configuration '{schema_key}' is missing")
                    continue
                if value is None:
                    continue
                if not isinstance(value, schema.type_hint):
                    try:
                        converted_value = schema.type_hint(value)
                        self.set(schema_key, converted_value)
                        value = converted_value
                    except (ValueError, TypeError):
                        errors.append(f"Configuration '{schema_key}' has invalid type. "
                                    f"Expected {schema.type_hint.__name__}, got {type(value).__name__}")
                        continue
                if schema.range_min is not None and value < schema.range_min:
                    errors.append(f"Configuration '{schema_key}' value {value} is below minimum {schema.range_min}")
                if schema.range_max is not None and value > schema.range_max:
                    errors.append(f"Configuration '{schema_key}' value {value} is above maximum {schema.range_max}")
                if schema.choices is not None and value not in schema.choices:
                    errors.append(f"Configuration '{schema_key}' value '{value}' not in allowed choices: {schema.choices}")
                if schema.validator is not None and not schema.validator(value):
                    errors.append(f"Configuration '{schema_key}' failed custom validation")
            except Exception as e:
                errors.append(f"Error validating configuration '{schema_key}': {e}")
        if errors:
            raise ConfigValidationError(f"Configuration validation failed:\n" + "\n".join(errors))

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        current = self._config
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any, validate: bool = True) -> None:
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        old_value = current.get(keys[-1])
        current[keys[-1]] = value
        self._change_history.append({
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'old_value': old_value,
            'new_value': value
        })
        if validate and self._validation_enabled:
            try:
                self._validate_single_key(key, value)
            except ConfigValidationError as e:
                current[keys[-1]] = old_value
                self._change_history.pop()
                raise e

    def _validate_single_key(self, key: str, value: Any):
        if key in self._schema:
            schema = self._schema[key]
            if not isinstance(value, schema.type_hint):
                try:
                    value = schema.type_hint(value)
                except (ValueError, TypeError):
                    raise ConfigValidationError(f"Invalid type for '{key}': expected {schema.type_hint.__name__}")
            if schema.range_min is not None and value < schema.range_min:
                raise ConfigValidationError(f"Value for '{key}' is below minimum {schema.range_min}")
            if schema.range_max is not None and value > schema.range_max:
                raise ConfigValidationError(f"Value for '{key}' is above maximum {schema.range_max}")
            if schema.choices is not None and value not in schema.choices:
                raise ConfigValidationError(f"Value for '{key}' not in allowed choices: {schema.choices}")

    def get_section(self, section: str) -> Dict[str, Any]:
        return self.get(section, {})

    def update(self, updates: Dict[str, Any], validate: bool = True) -> None:
        for key, value in updates.items():
            self.set(key, value, validate=False)
        if validate and self._validation_enabled:
            self._validate_config()

    def merge(self, other_config: Dict[str, Any]) -> None:
        self._deep_update(self._config, other_config)
        if self._validation_enabled:
            self._validate_config()

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

    def to_yaml(self, file_path: Optional[str] = None) -> str:
        yaml_str = yaml.dump(self._config, default_flow_style=False, sort_keys=True)
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            self.logger.info(f"Saved configuration to {file_path}")
        return yaml_str

    def to_json(self, file_path: Optional[str] = None, indent: int = 2) -> str:
        json_str = json.dumps(self._config, indent=indent, sort_keys=True)
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            self.logger.info(f"Saved configuration to {file_path}")
        return json_str

    def get_hash(self) -> str:
        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_change_history(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._change_history)

    def clear_change_history(self) -> None:
        self._change_history = []

    def enable_validation(self) -> None:
        self._validation_enabled = True

    def disable_validation(self) -> None:
        self._validation_enabled = False

    def get_environment_info(self) -> Dict[str, Any]:
        return {
            'environment': self.environment.value,
            'config_hash': self.get_hash(),
            'python_version': os.sys.version,
            'working_directory': str(Path.cwd()),
            'environment_variables': {
                key: value for key, value in os.environ.items()
                if key.startswith(('AZURE_', 'CUDA_', 'TORCH_'))
            }
        }

    def create_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        return {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config_hash': self.get_hash(),
            'environment': self.environment.value,
            'config': self.to_dict(),
            'environment_info': self.get_environment_info()
        }

    def validate_paths(self) -> List[str]:
        errors = []
        path_configs = [
            'paths.data_dir',
            'paths.models_dir', 
            'paths.results_dir',
            'paths.logs_dir',
            'data.raw_data_path',
            'data.processed_data_path'
        ]
        for path_config in path_configs:
            path_str = self.get(path_config)
            if path_str:
                path = Path(path_str)
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create path '{path_config}': {path} - {e}")
        return errors

    def setup_azure_config(self, subscription_id: str, resource_group: str, workspace_name: str) -> None:
        azure_config = {
            'azure.subscription_id': subscription_id,
            'azure.resource_group': resource_group,
            'azure.workspace_name': workspace_name
        }
        self.update(azure_config)
        self.logger.info("Updated Azure ML configuration")

    def get_azure_config(self) -> Dict[str, Any]:
        return self.get_section('azure')

    def __str__(self) -> str:
        return f"ConfigManager(environment={self.environment.value}, hash={self.get_hash()[:8]})"

    def __repr__(self) -> str:
        return f"ConfigManager(environment={self.environment.value}, config_hash={self.get_hash()})"

class ConfigTemplates:
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        return {
            'training': {
                'batch_size': 2,
                'num_epochs': 5,
                'checkpoint_frequency': 2,
                'log_frequency': 5
            },
            'evaluation': {
                'bootstrap_samples': 10
            },
            'logging': {
                'level': 'DEBUG'
            }
        }
    @staticmethod
    def get_quick_test_config() -> Dict[str, Any]:
        return {
            'training': {
                'batch_size': 1,
                'num_epochs': 1,
                'checkpoint_frequency': 1
            },
            'model': {
                'gnn': {
                    'hidden_dim': 64,
                    'num_layers': 1
                }
            },
            'graph': {
                'k_neighbors': 5
            }
        }
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        return {
            'training': {
                'batch_size': 32,
                'use_mixed_precision': True,
                'num_workers': 8
            },
            'deployment': {
                'scaling': {
                    'min_instances': 3,
                    'max_instances': 20
                }
            },
            'logging': {
                'level': 'WARNING',
                'console_handler': False
            }
        }
    @staticmethod
    def get_hyperopt_config() -> Dict[str, Any]:
        return {
            'hyperopt': {
                'enabled': True,
                'n_trials': 100,
                'search_space': {
                    'learning_rate': {
                        'type': 'float',
                        'low': 1e-5,
                        'high': 1e-2,
                        'log': True
                    },
                    'hidden_dim': {
                        'type': 'int',
                        'low': 128,
                        'high': 512
                    },
                    'dropout': {
                        'type': 'float',
                        'low': 0.1,
                        'high': 0.5
                    },
                    'batch_size': {
                        'type': 'categorical',
                        'choices': [8, 16, 32]
                    }
                }
            }
        }

def load_config_from_env() -> ConfigManager:
    env = os.getenv('MAINTIE_ENV', 'development')
    config_file = os.getenv('MAINTIE_CONFIG_FILE')
    if config_file and Path(config_file).exists():
        return ConfigManager(config_file=config_file, environment=Environment(env))
    else:
        possible_configs = [
            f'config/{env}_config.yaml',
            f'config/{env}_config.yml', 
            'config/default_config.yaml',
            'config/default_config.yml'
        ]
        for config_path in possible_configs:
            if Path(config_path).exists():
                return ConfigManager(config_file=config_path, environment=Environment(env))
        return ConfigManager(environment=Environment(env))

def merge_configs(*configs: ConfigManager) -> ConfigManager:
    if not configs:
        raise ValueError("At least one configuration must be provided")
    base_config = configs[0]
    merged_dict = base_config.to_dict()
    for config in configs[1:]:
        base_config._deep_update(merged_dict, config.to_dict())
    return ConfigManager(config_data=merged_dict, environment=base_config.environment)

if __name__ == "__main__":
    config = ConfigManager(environment=Environment.DEVELOPMENT)
    print(f"Model type: {config.get('model.gnn.type')}")
    print(f"Learning rate: {config.get('training.learning_rate')}")
    config.set('training.batch_size', 16)
    config.set('model.gnn.hidden_dim', 512)
    training_config = config.get_section('training')
    print(f"Training config: {training_config}")
    config.to_yaml('config/current_config.yaml')
    print(f"Config hash: {config.get_hash()}")
    path_errors = config.validate_paths()
    if path_errors:
        print(f"Path validation errors: {path_errors}")
    print("Configuration management example completed successfully")
