from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import json
import logging
from pathlib import Path

class MaintenanceLLMEmbedder:
    """
    Advanced LLM embedder specialized for maintenance text processing.
    
    Combines sentence transformer embeddings with domain-specific features
    to create rich node representations for graph neural networks.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 feature_config: Optional[Dict] = None,
                 cache_embeddings: bool = True,
                 device: str = "auto"):
        """
        Initialize the maintenance LLM embedder.
        
        Args:
            model_name: Sentence transformer model identifier
            feature_config: Configuration for domain feature extraction
            cache_embeddings: Whether to cache computed embeddings
            device: Computing device (auto, cpu, cuda)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer model
        self.device = self._setup_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Setup feature extraction configuration
        self.feature_config = self._load_feature_config(feature_config)
        self.domain_features_dim = self.feature_config.get('domain_features_dim', 32)
        self.combined_dim = self.embedding_dim + self.domain_features_dim
        
        # Initialize maintenance vocabulary and patterns
        self.maintenance_vocab = self._load_maintenance_vocabulary()
        self.equipment_patterns = self._load_equipment_patterns()
        self.failure_patterns = self._load_failure_patterns()
        self.action_patterns = self._load_action_patterns()
        
        # Setup caching and preprocessing
        self.cache_embeddings = cache_embeddings
        self.embedding_cache = {}
        self.text_preprocessor = MaintenanceTextPreprocessor()
        
        # Initialize feature extractors
        self.tfidf_vectorizer = None
        self.feature_scaler = StandardScaler()
        
        self.logger.info(f"Initialized MaintenanceLLMEmbedder with {self.embedding_dim}D embeddings")
    
    def _setup_device(self, device: str) -> str:
        """Setup computing device for embeddings."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_feature_config(self, config: Optional[Dict]) -> Dict:
        """Load feature extraction configuration."""
        default_config = {
            'domain_features_dim': 32,
            'equipment_weight': 0.3,
            'failure_weight': 0.25,
            'action_weight': 0.25,
            'urgency_weight': 0.2,
            'normalize_features': True,
            'use_tfidf_features': True,
            'tfidf_max_features': 100
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _load_maintenance_vocabulary(self) -> Dict[str, List[str]]:
        """Load maintenance-specific vocabulary terms."""
        return {
            'equipment_types': [
                'pump', 'motor', 'valve', 'sensor', 'bearing', 'belt', 'filter',
                'compressor', 'turbine', 'generator', 'transformer', 'switch',
                'conveyor', 'cylinder', 'actuator', 'controller', 'relay'
            ],
            'failure_terms': [
                'leak', 'crack', 'wear', 'corrosion', 'vibration', 'noise',
                'overheat', 'fail', 'malfunction', 'break', 'damage', 'fault',
                'defect', 'deteriorate', 'degrade', 'clog', 'jam', 'stuck'
            ],
            'action_terms': [
                'replace', 'repair', 'inspect', 'clean', 'lubricate', 'adjust',
                'calibrate', 'test', 'service', 'maintain', 'fix', 'install',
                'remove', 'check', 'verify', 'monitor', 'examine'
            ],
            'urgency_terms': [
                'urgent', 'critical', 'emergency', 'immediate', 'asap', 'priority',
                'routine', 'scheduled', 'preventive', 'planned', 'minor', 'major'
            ]
        }
    
    def _load_equipment_patterns(self) -> List[re.Pattern]:
        """Load compiled regex patterns for equipment detection."""
        patterns = [
            r'\b\w*pump\w*\b',
            r'\b\w*motor\w*\b', 
            r'\b\w*valve\w*\b',
            r'\b\w*sensor\w*\b',
            r'\b\w*bearing\w*\b',
            r'\b\w*belt\w*\b',
            r'\b\w*filter\w*\b',
            r'\b\w*compressor\w*\b',
            r'\b[A-Z]{2,4}-?\d{1,4}\b',  # Equipment codes
            r'\b\w+\s*#\s*\d+\b'        # Equipment numbers
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_failure_patterns(self) -> List[re.Pattern]:
        """Load compiled regex patterns for failure detection."""
        patterns = [
            r'\b\w*leak\w*\b',
            r'\b\w*crack\w*\b',
            r'\b\w*vibrat\w*\b',
            r'\b\w*overheat\w*\b',
            r'\b\w*fail\w*\b',
            r'\b\w*fault\w*\b',
            r'\b\w*alarm\w*\b',
            r'\berror\s*code\b',
            r'\bout\s*of\s*spec\b'
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _load_action_patterns(self) -> List[re.Pattern]:
        """Load compiled regex patterns for action detection."""
        patterns = [
            r'\breplace\w*\b',
            r'\brepair\w*\b',
            r'\binspect\w*\b',
            r'\bclean\w*\b',
            r'\bservice\w*\b',
            r'\bcheck\w*\b',
            r'\btest\w*\b',
            r'\binstall\w*\b',
            r'\bremove\w*\b',
            r'\badjust\w*\b'
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def create_node_features(self, 
                           texts: List[str], 
                           batch_size: int = 32,
                           show_progress: bool = True) -> np.ndarray:
        """
        Generate rich node features combining LLM embeddings with domain features.
        
        Args:
            texts: List of maintenance text documents
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar
            
        Returns:
            Node feature matrix [N, combined_dim]
        """
        self.logger.info(f"Generating node features for {len(texts)} texts")
        
        # Step 1: Preprocess maintenance texts
        processed_texts = self.preprocess_maintenance_texts(texts)
        
        # Step 2: Generate base LLM embeddings
        llm_embeddings = self.generate_llm_embeddings(
            processed_texts, batch_size, show_progress
        )
        
        # Step 3: Extract maintenance-specific features
        domain_features = self.extract_maintenance_features(texts)
        
        # Step 4: Combine embeddings with domain features
        combined_features = self.combine_features(llm_embeddings, domain_features)
        
        self.logger.info(f"Generated features with shape: {combined_features.shape}")
        return combined_features
    
    def preprocess_maintenance_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess maintenance texts for optimal embedding generation.
        
        Args:
            texts: Raw maintenance text documents
            
        Returns:
            Preprocessed text documents
        """
        processed_texts = []
        
        for text in texts:
            # Clean and normalize text
            processed_text = self.text_preprocessor.clean_text(text)
            
            # Standardize equipment terminology
            processed_text = self.text_preprocessor.standardize_equipment_terms(processed_text)
            
            # Expand abbreviations
            processed_text = self.text_preprocessor.expand_abbreviations(processed_text)
            
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def generate_llm_embeddings(self, 
                               texts: List[str], 
                               batch_size: int = 32,
                               show_progress: bool = True) -> np.ndarray:
        """
        Generate sentence transformer embeddings for texts.
        
        Args:
            texts: Preprocessed text documents
            batch_size: Batch size for processing
            show_progress: Whether to show progress
            
        Returns:
            LLM embeddings matrix [N, embedding_dim]
        """
        # Check cache first
        if self.cache_embeddings:
            cache_key = hash(str(texts))
            if cache_key in self.embedding_cache:
                self.logger.info("Retrieved embeddings from cache")
                return self.embedding_cache[cache_key]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Cache if enabled
        if self.cache_embeddings:
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def extract_maintenance_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract domain-specific maintenance features from texts.
        
        Args:
            texts: Original maintenance text documents
            
        Returns:
            Domain features matrix [N, domain_features_dim]
        """
        features_list = []
        
        for text in texts:
            # Extract different types of maintenance features
            equipment_features = self._extract_equipment_features(text)
            failure_features = self._extract_failure_features(text)
            action_features = self._extract_action_features(text)
            urgency_features = self._extract_urgency_features(text)
            
            # Combine all features
            text_features = np.concatenate([
                equipment_features,
                failure_features, 
                action_features,
                urgency_features
            ])
            
            features_list.append(text_features)
        
        # Convert to matrix
        features_matrix = np.array(features_list)
        
        # Normalize if configured
        if self.feature_config['normalize_features']:
            features_matrix = self.feature_scaler.fit_transform(features_matrix)
        
        # Ensure correct dimensionality
        if features_matrix.shape[1] != self.domain_features_dim:
            # Pad or truncate to desired dimension
            if features_matrix.shape[1] < self.domain_features_dim:
                padding = np.zeros((len(texts), self.domain_features_dim - features_matrix.shape[1]))
                features_matrix = np.concatenate([features_matrix, padding], axis=1)
            else:
                features_matrix = features_matrix[:, :self.domain_features_dim]
        
        return features_matrix
    
    def _extract_equipment_features(self, text: str) -> np.ndarray:
        """Extract equipment-related features from text."""
        features = []
        
        # Count equipment type mentions
        equipment_counts = {}
        for eq_type in self.maintenance_vocab['equipment_types']:
            count = len(re.findall(rf'\b{eq_type}\w*\b', text, re.IGNORECASE))
            equipment_counts[eq_type] = count
        
        # Equipment diversity score
        unique_equipment = sum(1 for count in equipment_counts.values() if count > 0)
        features.append(unique_equipment / len(self.maintenance_vocab['equipment_types']))
        
        # Total equipment mentions
        total_mentions = sum(equipment_counts.values())
        features.append(min(total_mentions / 10.0, 1.0))  # Normalized
        
        # Equipment code patterns
        equipment_code_matches = 0
        for pattern in self.equipment_patterns:
            equipment_code_matches += len(pattern.findall(text))
        features.append(min(equipment_code_matches / 5.0, 1.0))
        
        # Most common equipment type indicator
        most_common_equipment = max(equipment_counts, key=equipment_counts.get)
        equipment_type_features = [0.0] * 5  # Top 5 equipment types
        common_types = ['pump', 'motor', 'valve', 'sensor', 'bearing']
        
        if most_common_equipment in common_types:
            idx = common_types.index(most_common_equipment)
            equipment_type_features[idx] = 1.0
        
        features.extend(equipment_type_features)
        
        return np.array(features)
    
    def _extract_failure_features(self, text: str) -> np.ndarray:
        """Extract failure-related features from text."""
        features = []
        
        # Count failure indicators
        failure_counts = {}
        for failure_term in self.maintenance_vocab['failure_terms']:
            count = len(re.findall(rf'\b{failure_term}\w*\b', text, re.IGNORECASE))
            failure_counts[failure_term] = count
        
        # Failure severity score
        severe_failures = ['crack', 'break', 'fail', 'emergency', 'critical']
        severe_count = sum(failure_counts.get(term, 0) for term in severe_failures)
        features.append(min(severe_count / 3.0, 1.0))
        
        # Failure type diversity
        unique_failures = sum(1 for count in failure_counts.values() if count > 0)
        features.append(unique_failures / len(self.maintenance_vocab['failure_terms']))
        
        # Pattern-based failure detection
        failure_pattern_matches = 0
        for pattern in self.failure_patterns:
            failure_pattern_matches += len(pattern.findall(text))
        features.append(min(failure_pattern_matches / 3.0, 1.0))
        
        # Specific failure type indicators
        common_failures = ['leak', 'vibration', 'overheat', 'wear', 'noise']
        failure_type_features = []
        for failure_type in common_failures:
            has_failure = 1.0 if failure_counts.get(failure_type, 0) > 0 else 0.0
            failure_type_features.append(has_failure)
        
        features.extend(failure_type_features)
        
        return np.array(features)
    
    def _extract_action_features(self, text: str) -> np.ndarray:
        """Extract maintenance action features from text."""
        features = []
        
        # Count action type mentions
        action_counts = {}
        for action_term in self.maintenance_vocab['action_terms']:
            count = len(re.findall(rf'\b{action_term}\w*\b', text, re.IGNORECASE))
            action_counts[action_term] = count
        
        # Action complexity score
        complex_actions = ['replace', 'repair', 'calibrate', 'install']
        simple_actions = ['clean', 'check', 'inspect', 'test']
        
        complex_count = sum(action_counts.get(action, 0) for action in complex_actions)
        simple_count = sum(action_counts.get(action, 0) for action in simple_actions)
        
        if complex_count + simple_count > 0:
            complexity_score = complex_count / (complex_count + simple_count)
        else:
            complexity_score = 0.0
        
        features.append(complexity_score)
        
        # Total action mentions
        total_actions = sum(action_counts.values())
        features.append(min(total_actions / 5.0, 1.0))
        
        # Action type indicators
        common_actions = ['replace', 'repair', 'inspect', 'clean', 'test']
        action_type_features = []
        for action_type in common_actions:
            has_action = 1.0 if action_counts.get(action_type, 0) > 0 else 0.0
            action_type_features.append(has_action)
        
        features.extend(action_type_features)
        
        return np.array(features)
    
    def _extract_urgency_features(self, text: str) -> np.ndarray:
        """Extract urgency and priority features from text."""
        features = []
        
        # Count urgency indicators
        urgency_counts = {}
        for urgency_term in self.maintenance_vocab['urgency_terms']:
            count = len(re.findall(rf'\b{urgency_term}\w*\b', text, re.IGNORECASE))
            urgency_counts[urgency_term] = count
        
        # Urgency level score
        high_urgency = ['urgent', 'critical', 'emergency', 'immediate']
        low_urgency = ['routine', 'scheduled', 'preventive', 'planned']
        
        high_count = sum(urgency_counts.get(term, 0) for term in high_urgency)
        low_count = sum(urgency_counts.get(term, 0) for term in low_urgency)
        
        if high_count + low_count > 0:
            urgency_score = high_count / (high_count + low_count)
        else:
            urgency_score = 0.5  # Neutral urgency
        
        features.append(urgency_score)
        
        # Time indicators
        time_patterns = [
            r'\basap\b', r'\bimmediately\b', r'\btoday\b', r'\btomorrow\b',
            r'\bnext\s+week\b', r'\bscheduled\b', r'\bplanned\b'
        ]
        
        time_urgency = 0.0
        for pattern in time_patterns[:3]:  # High urgency patterns
            if re.search(pattern, text, re.IGNORECASE):
                time_urgency = 1.0
                break
        
        features.append(time_urgency)
        
        return np.array(features)
    
    def combine_features(self, 
                        llm_embeddings: np.ndarray, 
                        domain_features: np.ndarray) -> np.ndarray:
        """
        Combine LLM embeddings with domain-specific features.
        
        Args:
            llm_embeddings: Sentence transformer embeddings [N, embedding_dim]
            domain_features: Maintenance domain features [N, domain_features_dim]
            
        Returns:
            Combined feature matrix [N, combined_dim]
        """
        # Ensure compatible shapes
        assert llm_embeddings.shape[0] == domain_features.shape[0], \
            "Embedding and feature batch sizes must match"
        
        # Concatenate features
        combined_features = np.concatenate([llm_embeddings, domain_features], axis=1)
        
        # Optional: Apply feature fusion techniques
        if self.feature_config.get('use_feature_fusion', False):
            combined_features = self._apply_feature_fusion(llm_embeddings, domain_features)
        
        return combined_features
    
    def _apply_feature_fusion(self, 
                            llm_embeddings: np.ndarray, 
                            domain_features: np.ndarray) -> np.ndarray:
        """Apply advanced feature fusion techniques."""
        # Weighted concatenation
        llm_weight = self.feature_config.get('llm_weight', 0.8)
        domain_weight = self.feature_config.get('domain_weight', 0.2)
        
        weighted_llm = llm_embeddings * llm_weight
        weighted_domain = domain_features * domain_weight
        
        # Concatenate weighted features
        fused_features = np.concatenate([weighted_llm, weighted_domain], axis=1)
        
        return fused_features
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """Save embeddings to disk."""
        np.save(filepath, embeddings)
        self.logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(filepath)
        self.logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the combined embedding dimension."""
        return self.combined_dim


class MaintenanceTextPreprocessor:
    """Specialized text preprocessor for maintenance documents."""
    
    def __init__(self):
        self.abbreviation_map = self._load_abbreviation_map()
        self.equipment_normalizer = self._load_equipment_normalizer()
    
    def _load_abbreviation_map(self) -> Dict[str, str]:
        """Load common maintenance abbreviations and expansions."""
        return {
            'temp': 'temperature',
            'psi': 'pressure per square inch',
            'rpm': 'revolutions per minute',
            'hp': 'horsepower',
            'kw': 'kilowatt',
            'ac': 'alternating current',
            'dc': 'direct current',
            'hmi': 'human machine interface',
            'plc': 'programmable logic controller',
            'vfd': 'variable frequency drive',
            'mcc': 'motor control center',
            'o&m': 'operation and maintenance',
            'pm': 'preventive maintenance',
            'cm': 'corrective maintenance'
        }
    
    def _load_equipment_normalizer(self) -> Dict[str, str]:
        """Load equipment term normalizations."""
        return {
            'xfmr': 'transformer',
            'xformer': 'transformer',
            'genset': 'generator set',
            'gearbox': 'gear box',
            'switchgear': 'switch gear'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize maintenance text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep equipment codes
        text = re.sub(r'[^\w\s\-#]', ' ', text)
        
        # Normalize equipment codes
        text = re.sub(r'(\w+)\s*-\s*(\d+)', r'\1-\2', text)
        text = re.sub(r'(\w+)\s*#\s*(\d+)', r'\1 #\2', text)
        
        return text.strip()
    
    def standardize_equipment_terms(self, text: str) -> str:
        """Standardize equipment terminology."""
        for abbrev, full_term in self.equipment_normalizer.items():
            text = re.sub(rf'\b{abbrev}\b', full_term, text, flags=re.IGNORECASE)
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common maintenance abbreviations."""
        for abbrev, expansion in self.abbreviation_map.items():
            text = re.sub(rf'\b{abbrev}\b', expansion, text, flags=re.IGNORECASE)
        
        return text


# Example usage and testing
if __name__ == "__main__":
    # Initialize embedder
    embedder = MaintenanceLLMEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        feature_config={
            'domain_features_dim': 32,
            'normalize_features': True
        }
    )
    
    # Sample maintenance texts
    sample_texts = [
        "Replace faulty pressure sensor in cooling system pump #3",
        "Inspect vibration levels in main turbine bearing assembly",
        "Clean and lubricate conveyor belt drive motor",
        "Emergency repair of hydraulic leak in valve actuator"
    ]
    
    # Generate embeddings
    embeddings = embedder.create_node_features(sample_texts)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
