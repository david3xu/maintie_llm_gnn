# 🚀 MaintIE LLM-GNN Project Template & Implementation Guide

**Project**: MaintIE LLM-Enhanced Graph Neural Network Information Extraction  
**Template Version**: 1.0  
**Target Environment**: Azure ML / Local Development  
**Expected Timeline**: 6-8 weeks implementation

---

## 📋 **Quick Start Commands**

### **Project Initialization**
```bash
# 1. Create project directory and navigate
mkdir maintie_llm_gnn && cd maintie_llm_gnn

# 2. Initialize git repository
git init
git remote add origin <your-repository-url>

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Create project structure
make init-structure

# 5. Install dependencies
make install-deps

# 6. Download MaintIE dataset
make download-data

# 7. Setup configuration
make setup-config

# 8. Verify installation
make verify-setup

# 9. Run initial data processing
make process-data

# 10. Start training pipeline
make train-baseline
```

---

## 📁 **Complete Directory Structure Template**

```
maintie_llm_gnn/
├── 📊 data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── README.md
├── 🧠 src/
│   ├── core/
│   ├── data_processing/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
├── ⚙️ config/
├── 🚀 scripts/
├── 📋 models/
├── 📊 results/
├── 🧪 tests/
├── 📚 docs/
├── 🚢 deployment/
├── 🔮 FUTURE_EXTENSIONS/
├── Makefile
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 🛠️ **Core Implementation Templates**

### **1. Main Hybrid Architecture Class**

**File**: `src/models/llm_gnn_hybrid.py`

```python
class MaintIELLMGNNHybrid:
    """
    Main hybrid architecture combining LLM embeddings with GNN processing
    for maintenance information extraction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # TODO: Initialize LLM embedder component
        # TODO: Initialize graph builder component  
        # TODO: Initialize GNN processor component
        # TODO: Initialize domain validator component
        # TODO: Load maintenance ontology and constraints
        pass
    
    def extract_maintenance_info(self, texts: List[str]) -> Dict[str, Any]:
        """
        End-to-end extraction pipeline.
        
        Args:
            texts: List of maintenance text documents
            
        Returns:
            Dictionary containing extracted entities and relations
        """
        # TODO: Generate LLM embeddings for input texts
        # TODO: Build maintenance-aware graph structure
        # TODO: Process through GNN for entity/relation prediction
        # TODO: Apply domain constraints and validation
        # TODO: Format and return structured results
        pass
    
    def train(self, train_data: Dataset, val_data: Dataset) -> None:
        """Train the hybrid model on silver corpus data."""
        # TODO: Implement training loop with multi-task loss
        # TODO: Add early stopping and checkpoint saving
        # TODO: Track training metrics and validation performance
        pass
    
    def evaluate(self, test_data: Dataset) -> Dict[str, float]:
        """Evaluate model on gold standard test data."""
        # TODO: Run inference on test data
        # TODO: Calculate NER and RE F1 scores
        # TODO: Generate detailed performance report
        pass
```

### **2. LLM Embedding Component**

**File**: `src/models/embedders/maintenance_llm_embedder.py`

```python
class MaintenanceLLMEmbedder:
    """
    Specialized LLM embedder for maintenance text processing.
    """
    
    def __init__(self, model_name: str, feature_config: Dict):
        # TODO: Initialize sentence transformer model
        # TODO: Load maintenance vocabulary and domain features
        # TODO: Setup preprocessing pipeline
        pass
    
    def create_node_features(self, texts: List[str]) -> np.ndarray:
        """
        Generate rich node features combining LLM embeddings with domain features.
        
        Args:
            texts: Maintenance text documents
            
        Returns:
            Node feature matrix [N, feature_dim]
        """
        # TODO: Preprocess maintenance texts (clean, normalize)
        # TODO: Generate base LLM embeddings
        # TODO: Extract maintenance-specific features
        # TODO: Combine embeddings with domain features
        # TODO: Return feature matrix
        pass
    
    def extract_maintenance_features(self, texts: List[str]) -> np.ndarray:
        """Extract domain-specific maintenance features."""
        # TODO: Count equipment type mentions
        # TODO: Detect failure pattern indicators
        # TODO: Identify maintenance action types
        # TODO: Assess urgency and severity signals
        pass
```

### **3. Graph Construction Component**

**File**: `src/data_processing/graph_builder.py`

```python
class MaintenanceGraphBuilder:
    """
    Builds domain-aware graphs from maintenance text embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # TODO: Load equipment ontology
        # TODO: Initialize similarity thresholds
        # TODO: Setup edge type configurations
        pass
    
    def build_maintenance_graph(self, node_features: np.ndarray, 
                              texts: List[str], 
                              annotations: Optional[List[Dict]] = None) -> Data:
        """
        Build PyTorch Geometric Data object with multiple edge types.
        
        Args:
            node_features: LLM embeddings + domain features
            texts: Original text documents
            annotations: Optional ground truth annotations
            
        Returns:
            PyTorch Geometric Data object
        """
        # TODO: Build semantic similarity edges
        # TODO: Add entity co-occurrence edges
        # TODO: Include equipment hierarchy edges
        # TODO: Create maintenance procedure similarity edges
        # TODO: Combine into PyTorch Geometric format
        pass
    
    def build_semantic_edges(self, embeddings: np.ndarray, threshold: float) -> List[List[int]]:
        """Build edges based on semantic similarity of embeddings."""
        # TODO: Compute cosine similarity matrix
        # TODO: Apply threshold filtering
        # TODO: Create bidirectional edge list
        pass
    
    def build_equipment_hierarchy_edges(self, texts: List[str]) -> List[List[int]]:
        """Build edges based on equipment ontology relationships."""
        # TODO: Extract equipment mentions from texts
        # TODO: Map to ontology hierarchy
        # TODO: Create edges between related equipment types
        pass
```

### **4. GNN Processing Component**

**File**: `src/models/graph_networks/maintenance_gnn.py`

```python
class MaintenanceGNN(torch.nn.Module):
    """
    Graph neural network optimized for maintenance information extraction.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_entity_classes: int, num_relation_classes: int):
        super().__init__()
        # TODO: Initialize input projection layer
        # TODO: Setup GNN layers (GAT or GCN)
        # TODO: Create entity classification head
        # TODO: Create relation classification head
        # TODO: Initialize domain constraint validator
        pass
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GNN for entity and relation prediction.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Graph connectivity [2, E]
            batch: Batch indices for multiple graphs
            
        Returns:
            Tuple of (entity_logits, relation_logits)
        """
        # TODO: Project input features to hidden dimension
        # TODO: Apply GNN layers with attention/convolution
        # TODO: Generate entity predictions
        # TODO: Generate relation predictions for edge pairs
        # TODO: Apply domain constraints validation
        pass
    
    def predict_relations(self, node_embeddings: torch.Tensor, 
                         edge_index: torch.Tensor) -> torch.Tensor:
        """Predict relations between entity pairs."""
        # TODO: Extract node pairs based on edges
        # TODO: Concatenate or combine node representations
        # TODO: Apply relation classifier
        pass
```

### **5. Training Pipeline**

**File**: `src/training/trainer.py`

```python
class MaintIETrainer:
    """
    Training pipeline for MaintIE LLM-GNN hybrid model.
    """
    
    def __init__(self, model: MaintIELLMGNNHybrid, config: Dict[str, Any]):
        # TODO: Initialize model and optimizer
        # TODO: Setup loss functions (entity + relation)
        # TODO: Configure learning rate scheduler
        # TODO: Setup logging and checkpoint saving
        pass
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        # TODO: Iterate through training batches
        # TODO: Compute forward pass and losses
        # TODO: Perform backward pass and optimization
        # TODO: Track training metrics
        pass
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model performance."""
        # TODO: Run validation without gradient computation
        # TODO: Calculate validation metrics
        # TODO: Return performance dictionary
        pass
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int) -> None:
        """Full training loop with validation."""
        # TODO: Main training loop
        # TODO: Early stopping logic
        # TODO: Checkpoint saving
        # TODO: Final model export
        pass
```

### **6. Evaluation Framework**

**File**: `src/evaluation/gold_evaluator.py`

```python
class GoldStandardEvaluator:
    """
    Evaluation framework for MaintIE gold standard validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # TODO: Initialize evaluation metrics
        # TODO: Setup gold standard data loader
        # TODO: Configure output formatting
        pass
    
    def evaluate_model(self, model: MaintIELLMGNNHybrid, 
                      gold_data: Dataset) -> Dict[str, Any]:
        """
        Comprehensive evaluation on gold standard data.
        
        Args:
            model: Trained MaintIE model
            gold_data: Gold standard dataset
            
        Returns:
            Detailed evaluation results
        """
        # TODO: Run model inference on gold data
        # TODO: Calculate entity extraction metrics (NER F1)
        # TODO: Calculate relation extraction metrics (RE F1)
        # TODO: Generate confusion matrices and error analysis
        # TODO: Compare against SPERT baseline results
        pass
    
    def calculate_ner_metrics(self, predictions: List[Dict], 
                            ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate Named Entity Recognition metrics."""
        # TODO: Align predicted and true entities
        # TODO: Calculate precision, recall, F1 per entity type
        # TODO: Calculate macro and micro averages
        pass
    
    def calculate_re_metrics(self, predictions: List[Dict], 
                           ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate Relation Extraction metrics."""
        # TODO: Align predicted and true relations
        # TODO: Calculate strict and relaxed matching scores
        # TODO: Calculate precision, recall, F1 per relation type
        pass
```

---

## ⚙️ **Configuration Templates**

### **Main Configuration**

**File**: `config/default_config.yaml`

```yaml
# MaintIE LLM-GNN Configuration Template

project:
  name: "maintie_llm_gnn"
  version: "1.0.0"
  description: "MaintIE LLM-GNN Hybrid Information Extraction"

data:
  # TODO: Configure data paths
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  gold_corpus_path: "data/raw/gold_release.json"
  silver_corpus_path: "data/raw/silver_release.json"
  ontology_path: "data/raw/scheme.json"
  
  # TODO: Configure data processing
  train_split: 0.8
  val_split: 0.2
  max_sequence_length: 512

model:
  # TODO: Configure LLM embedder
  llm_embedder:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: 384
    domain_features_dim: 32
    combined_features_dim: 416
  
  # TODO: Configure GNN architecture
  gnn:
    type: "GAT"  # Options: GAT, GCN, GraphSAGE
    hidden_dim: 256
    num_layers: 2
    num_heads: 8  # For GAT
    dropout: 0.2
  
  # TODO: Configure output dimensions
  num_entity_classes: 224  # FG-3 complexity
  num_relation_classes: 6

graph:
  # TODO: Configure graph construction
  similarity_threshold: 0.75
  k_neighbors: 15
  edge_types:
    - "semantic_similarity"
    - "entity_cooccurrence" 
    - "equipment_hierarchy"
    - "procedure_similarity"

training:
  # TODO: Configure training parameters
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10
  
  # TODO: Configure loss weights
  entity_loss_weight: 1.0
  relation_loss_weight: 1.0

evaluation:
  # TODO: Configure evaluation settings
  metrics:
    - "precision"
    - "recall" 
    - "f1_score"
  complexity_levels:
    - "FG-0"
    - "FG-1"
    - "FG-2" 
    - "FG-3"
```

### **Training Configuration**

**File**: `config/training_config.yaml`

```yaml
# Training-Specific Configuration

experiment:
  name: "silver_baseline_v1"
  # TODO: Configure experiment tracking
  track_metrics: true
  save_checkpoints: true
  checkpoint_frequency: 5

optimizer:
  # TODO: Configure optimizer settings
  type: "AdamW"
  learning_rate: 0.001
  weight_decay: 0.0001
  betas: [0.9, 0.999]

scheduler:
  # TODO: Configure learning rate scheduler
  type: "ReduceLROnPlateau"
  factor: 0.5
  patience: 5
  min_lr: 0.00001

# TODO: Configure data augmentation
augmentation:
  enabled: false
  techniques: []

# TODO: Configure regularization
regularization:
  dropout: 0.2
  layer_norm: true
  gradient_clipping: 1.0
```

---

## 🚀 **Setup Scripts Templates**

### **Main Setup Script**

**File**: `scripts/setup/setup_project.py`

```python
#!/usr/bin/env python3
"""
MaintIE LLM-GNN Project Setup Script
"""

import os
import json
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create complete project directory structure."""
    # TODO: Create all necessary directories
    # TODO: Add README files to each directory
    # TODO: Create .gitkeep files for empty directories
    pass

def download_maintie_data():
    """Download and setup MaintIE dataset."""
    # TODO: Download gold_release.json
    # TODO: Download silver_release.json (if available)
    # TODO: Download scheme.json ontology
    # TODO: Validate data integrity
    pass

def install_dependencies():
    """Install required Python packages."""
    # TODO: Install PyTorch and PyTorch Geometric
    # TODO: Install sentence-transformers
    # TODO: Install evaluation libraries
    # TODO: Install development tools
    pass

def setup_configuration():
    """Initialize configuration files."""
    # TODO: Copy template configurations
    # TODO: Set up environment variables
    # TODO: Create local config overrides
    pass

def verify_installation():
    """Verify that everything is set up correctly."""
    # TODO: Test imports
    # TODO: Verify data file access
    # TODO: Test model loading
    # TODO: Run simple pipeline test
    pass

if __name__ == "__main__":
    print("🚀 Setting up MaintIE LLM-GNN Project...")
    
    # TODO: Execute setup steps
    create_directory_structure()
    download_maintie_data()
    install_dependencies()
    setup_configuration()
    verify_installation()
    
    print("✅ Setup complete! Run 'make train-baseline' to start training.")
```

### **Data Processing Script**

**File**: `scripts/data_preparation/process_silver_corpus.py`

```python
#!/usr/bin/env python3
"""
Silver Corpus Data Processing Pipeline
"""

def load_silver_corpus(file_path: str) -> List[Dict]:
    """Load and validate silver corpus data."""
    # TODO: Load JSON data file
    # TODO: Validate data format and structure
    # TODO: Report data statistics
    pass

def preprocess_texts(texts: List[str]) -> List[str]:
    """Preprocess maintenance texts for embedding generation."""
    # TODO: Text cleaning and normalization
    # TODO: Handle special maintenance terminology
    # TODO: Remove or standardize abbreviations
    pass

def generate_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Generate LLM embeddings for text corpus."""
    # TODO: Initialize sentence transformer model
    # TODO: Batch process texts to embeddings
    # TODO: Save embeddings to disk
    pass

def extract_domain_features(texts: List[str]) -> np.ndarray:
    """Extract maintenance-specific features."""
    # TODO: Equipment type counting
    # TODO: Failure pattern detection
    # TODO: Action type identification
    # TODO: Urgency level assessment
    pass

def create_train_val_split(data: List[Dict], split_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    """Create training and validation splits."""
    # TODO: Stratified splitting by entity types
    # TODO: Ensure balanced representation
    # TODO: Save splits to separate files
    pass

if __name__ == "__main__":
    # TODO: Main processing pipeline
    pass
```

---

## 🧪 **Test Templates**

### **Unit Test Template**

**File**: `tests/unit/test_models.py`

```python
import unittest
import torch
import numpy as np
from src.models.llm_gnn_hybrid import MaintIELLMGNNHybrid

class TestMaintIEModels(unittest.TestCase):
    """Unit tests for MaintIE model components."""
    
    def setUp(self):
        # TODO: Setup test configuration
        # TODO: Create mock data
        # TODO: Initialize test models
        pass
    
    def test_llm_embedder_initialization(self):
        """Test LLM embedder component initialization."""
        # TODO: Test successful initialization
        # TODO: Test invalid configuration handling
        pass
    
    def test_graph_builder_creation(self):
        """Test graph construction from embeddings."""
        # TODO: Test with sample embeddings
        # TODO: Verify graph structure properties
        # TODO: Test edge type creation
        pass
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass computation."""
        # TODO: Create sample graph data
        # TODO: Test forward pass execution
        # TODO: Verify output dimensions
        pass
    
    def test_hybrid_model_integration(self):
        """Test full hybrid model pipeline."""
        # TODO: Test end-to-end processing
        # TODO: Verify output format
        # TODO: Test with different input sizes
        pass

if __name__ == "__main__":
    unittest.main()
```

---

## 📋 **Makefile Template**

**File**: `Makefile`

```makefile
# MaintIE LLM-GNN Project Makefile

.PHONY: help init-structure install-deps download-data setup-config verify-setup process-data train-baseline evaluate clean

help:  ## Show this help message
	@echo "MaintIE LLM-GNN Project Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

init-structure:  ## Create project directory structure
	@echo "📁 Creating project structure..."
	# TODO: Create all directories
	# TODO: Add template files
	# TODO: Initialize git repository

install-deps:  ## Install Python dependencies
	@echo "📦 Installing dependencies..."
	# TODO: Install PyTorch ecosystem
	# TODO: Install NLP libraries
	# TODO: Install development tools

download-data:  ## Download MaintIE dataset
	@echo "📊 Downloading MaintIE data..."
	# TODO: Download gold standard data
	# TODO: Download silver corpus
	# TODO: Download ontology files

setup-config:  ## Setup configuration files
	@echo "⚙️ Setting up configuration..."
	# TODO: Copy template configs
	# TODO: Set environment variables
	# TODO: Create local overrides

verify-setup:  ## Verify installation
	@echo "✅ Verifying setup..."
	# TODO: Test imports
	# TODO: Verify data access
	# TODO: Test model loading

process-data:  ## Process data for training
	@echo "🔧 Processing data..."
	# TODO: Generate embeddings
	# TODO: Build graphs
	# TODO: Create train/val splits

train-baseline:  ## Train baseline model
	@echo "🚀 Training baseline model..."
	# TODO: Run silver corpus training
	# TODO: Save model checkpoints
	# TODO: Log training progress

evaluate:  ## Evaluate on gold standard
	@echo "📊 Evaluating model..."
	# TODO: Run gold standard evaluation
	# TODO: Generate performance report
	# TODO: Compare with SPERT baseline

clean:  ## Clean generated files
	@echo "🧹 Cleaning up..."
	# TODO: Remove temporary files
	# TODO: Clean cache directories
	# TODO: Reset processed data
```

---

## 📚 **Documentation Templates**

### **Main README**

**File**: `README.md`

```markdown
# 🚀 MaintIE LLM-GNN: Hybrid Information Extraction

**Status**: In Development  
**Performance Target**: 90-95% Entity F1, 85-90% Relation F1  
**Baseline Improvement**: 15-20% over SPERT  

## 📋 Quick Start

```bash
# Setup project
git clone <repository> && cd maintie_llm_gnn
make install-deps && make download-data && make setup-config

# Train model
make process-data && make train-baseline

# Evaluate results
make evaluate
```

## 🎯 Project Overview

TODO: Add project description
TODO: Add architecture overview
TODO: Add performance comparisons

## 🏗️ Architecture

TODO: Add architecture diagram
TODO: Describe LLM+GNN hybrid approach
TODO: Explain training pipeline

## 📊 Results

TODO: Add benchmark results
TODO: Compare with SPERT baseline
TODO: Show performance across complexity levels

## 🚀 Deployment

TODO: Add deployment instructions
TODO: Docker configuration
TODO: Azure ML setup

## 📖 Citation

TODO: Add citation information
TODO: Reference MaintIE benchmark
TODO: Acknowledge contributions
```

---

## 🎯 **Implementation Priority Checklist**

### **Week 1: Foundation**
- [ ] Setup project structure with `make init-structure`
- [ ] Implement `MaintenanceLLMEmbedder` class
- [ ] Implement `MaintenanceGraphBuilder` class
- [ ] Create data processing pipeline
- [ ] Setup unit tests for core components

### **Week 2: Core Model**
- [ ] Implement `MaintenanceGNN` architecture
- [ ] Implement `MaintIELLMGNNHybrid` main class
- [ ] Create training pipeline with `MaintIETrainer`
- [ ] Setup configuration management
- [ ] Implement basic evaluation metrics

### **Week 3: Training & Validation**
- [ ] Process silver corpus with embedding generation
- [ ] Train baseline model on silver data
- [ ] Implement gold standard evaluation
- [ ] Setup experiment tracking and logging
- [ ] Create comparison with SPERT baseline

### **Week 4: Optimization & Analysis**
- [ ] Hyperparameter tuning and optimization
- [ ] Error analysis and failure mode investigation
- [ ] Performance benchmarking and profiling
- [ ] Documentation and result reporting
- [ ] Prepare for production deployment

---

## 🔮 **Future Extensions Framework**

### **Phase 2: Self-Training (FUTURE_EXTENSIONS/)**
- [ ] Implement `PseudoLabelingPipeline` class
- [ ] Create `ConfidenceEstimator` component
- [ ] Build `IterativeTrainer` for continuous improvement
- [ ] Add unlabeled data processing pipeline
- [ ] Implement convergence detection

### **Phase 3: Advanced Features**
- [ ] Multi-modal integration (text + images)
- [ ] Cross-domain adaptation
- [ ] Real-time inference optimization
- [ ] Production monitoring and alerting
- [ ] Advanced visualization and explainability

---

**This template provides the complete framework to implement your MaintIE LLM-GNN hybrid approach with clear TODOs and implementation priorities. Follow the weekly schedule and priority checklist for systematic development.**