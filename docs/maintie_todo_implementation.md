# MaintIE LLM-GNN: Complete Implementation TODO List

**Code Priority**: Working functionality first, optimization later  
**Start Simple**: Basic classes and methods, professional structure  
**Professional**: Clean architecture with proper separation of concerns  
**Good Lifecycle**: Development → Testing → Production workflow

---

## 📋 **Implementation Priority Levels**

- **🔥 P0**: Critical - Must work for basic pipeline
- **⚡ P1**: Important - Needed for professional functionality  
- **📋 P2**: Enhancement - Future improvement features

---

## 🗂️ **Module 1: Data Processing (`src/data_processing/`)**

### **File: `src/data_processing/data_loader.py`**
**Purpose**: Simple, reliable JSON loading for MaintIE data

```python
class MaintIEDataLoader:
    """Simple data loader for MaintIE JSON files"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with basic configuration"""
        # - Setup logging
        # - Store config parameters
        # - Initialize validation flags
    
    def load_gold_corpus(self, file_path: str) -> List[Dict[str, Any]]:
        """Load gold_release.json - must handle real format"""
        # - Open and parse JSON file
        # - Validate basic structure
        # - Return list of samples
        # - Handle errors gracefully
    
    def load_silver_corpus(self, file_path: str) -> List[Dict[str, Any]]:
        """Load silver_release.json - must handle real format"""
        # - Open and parse JSON file  
        # - Validate basic structure
        # - Return list of samples
        # - Handle errors gracefully
    
    def load_ontology(self, file_path: str) -> Dict[str, Any]:
        """Load scheme.json ontology"""
        # - Parse ontology structure
        # - Extract entity hierarchy
        # - Extract relation types
        # - Return structured ontology
    
    # ⚡ P1 - Professional Methods
    def validate_sample_format(self, sample: Dict[str, Any]) -> bool:
        """Validate single sample against expected format"""
        # - Check required fields: text, entities, relations
        # - Validate entity structure
        # - Validate relation structure
        # - Return validation result
    
    def get_corpus_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate corpus statistics"""
        # - Count total samples
        # - Count entities per sample
        # - Count relations per sample
        # - Calculate averages

# 🔥 P0 - Utility Functions
def create_train_val_split(data: List[Dict], ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Simple train/validation split"""
    # - Shuffle data
    # - Split by ratio
    # - Return train and val sets

def load_maintie_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main data loading function"""
    # - Initialize loader
    # - Load all required files
    # - Create train/val splits
    # - Return complete data dictionary
```

### **File: `src/data_processing/embedding_generator.py`**
**Purpose**: Generate LLM embeddings for text data

```python
class EmbeddingGenerator:
    """Simple LLM embedding generation"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with lightweight model"""
        # - Load sentence transformer model
        # - Setup device (CPU/GPU)
        # - Initialize configuration
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text list"""
        # - Process texts in batches
        # - Generate embeddings
        # - Return numpy array [N, embedding_dim]
        # - Handle empty/invalid texts
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], output_path: str) -> None:
        """Save embeddings to disk"""
        # - Create embeddings dictionary
        # - Save as pickle file
        # - Include metadata
    
    def load_embeddings(self, input_path: str) -> Tuple[np.ndarray, List[str]]:
        """Load embeddings from disk"""
        # - Load pickle file
        # - Extract embeddings and texts
        # - Return tuple
    
    # ⚡ P1 - Professional Methods
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        # - Return model embedding dimension
    
    def batch_process(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Process large text lists in batches"""
        # - Split into batches
        # - Process each batch
        # - Concatenate results

# 🔥 P0 - CLI Integration
def main():
    """CLI entry point for make generate-embeddings"""
    # - Parse command line arguments
    # - Load data from data_loader
    # - Generate embeddings
    # - Save to processed directory
```

### **File: `src/data_processing/graph_builder.py`**
**Purpose**: Build PyTorch Geometric graphs from embeddings

```python
class SimpleGraphBuilder:
    """Simple graph construction with semantic similarity"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, config: Dict[str, Any]):
        """Initialize graph builder"""
        # - Setup similarity threshold
        # - Configure k-neighbors
        # - Initialize logging
    
    def build_graph(self, embeddings: np.ndarray, texts: List[str]) -> Data:
        """Build PyTorch Geometric graph"""
        # - Calculate semantic similarity edges
        # - Create node features from embeddings
        # - Build PyG Data object
        # - Add self-loops if configured
        # - Return complete graph
    
    def _build_similarity_edges(self, embeddings: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build edges based on cosine similarity"""
        # - Use k-nearest neighbors for efficiency
        # - Calculate cosine similarity
        # - Filter by threshold
        # - Return edge_index and edge_weights
    
    def save_graph(self, graph: Data, output_path: str) -> None:
        """Save graph to disk"""
        # - Save PyG Data object
        # - Include metadata
    
    def load_graph(self, input_path: str) -> Data:
        """Load graph from disk"""
        # - Load PyG Data object
        # - Restore metadata
    
    # ⚡ P1 - Professional Methods
    def get_graph_statistics(self, graph: Data) -> Dict[str, Any]:
        """Calculate graph statistics"""
        # - Count nodes and edges
        # - Calculate average degree
        # - Compute connectivity metrics

# 🔥 P0 - CLI Integration
def main():
    """CLI entry point for make build-graphs"""
    # - Parse command line arguments
    # - Load embeddings
    # - Build graphs
    # - Save to processed directory
```

---

## 🧠 **Module 2: Models (`src/models/`)**

### **File: `src/models/llm_gnn_hybrid.py`**
**Purpose**: Main hybrid LLM+GNN model architecture

```python
class MaintIELLMGNNHybrid(torch.nn.Module):
    """Simple LLM+GNN hybrid model"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, config: Dict[str, Any]):
        """Initialize hybrid model"""
        # - Setup embedding dimension
        # - Initialize GNN layers
        # - Create output classification heads
        # - Configure complexity level
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through hybrid model"""
        # - Process node features through GNN
        # - Apply classification heads
        # - Return entity and relation predictions
    
    def _create_output_heads(self) -> Dict[str, torch.nn.Module]:
        """Create entity and relation classification heads"""
        # - Entity classification head
        # - Relation classification head
        # - Dynamic output dimensions based on complexity
    
    # ⚡ P1 - Professional Methods
    def get_complexity_classes(self, complexity_level: str) -> Tuple[int, int]:
        """Get number of entity and relation classes"""
        # - Map complexity level to class counts
        # - Return (num_entity_classes, num_relation_classes)
    
    def save_model(self, path: str, optimizer_state: Optional[Dict] = None) -> None:
        """Save model checkpoint"""
        # - Save model state dict
        # - Save configuration
        # - Save optimizer state if provided

# 🔥 P0 - Utility Functions
def create_model(config: Dict[str, Any]) -> MaintIELLMGNNHybrid:
    """Factory function to create model"""
    # - Read configuration
    # - Initialize model
    # - Return configured model

def load_model(checkpoint_path: str) -> MaintIELLMGNNHybrid:
    """Load model from checkpoint"""
    # - Load checkpoint
    # - Recreate model
    # - Load state dict
```

### **File: `src/models/simple_gnn.py`**
**Purpose**: Basic GNN implementation for graph processing

```python
class SimpleGNN(torch.nn.Module):
    """Simple Graph Neural Network"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """Initialize simple GNN"""
        # - Create GCN layers
        # - Setup dropout
        # - Configure activation functions
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GNN layers"""
        # - Apply GCN layers sequentially
        # - Apply activation and dropout
        # - Return node representations
    
    # ⚡ P1 - Professional Methods
    def get_node_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings without final classification"""
        # - Forward pass without output layer
        # - Return intermediate node representations
```

---

## 🚀 **Module 3: Training (`src/training/`)**

### **File: `src/training/simple_trainer.py`**
**Purpose**: Basic training loop implementation

```python
class SimpleTrainer:
    """Simple training implementation"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer"""
        # - Setup device (CPU/GPU)
        # - Configure learning rate, batch size
        # - Initialize logging
        # - Setup loss functions
    
    def train(self, model: torch.nn.Module, train_data: List[Data], val_data: List[Data]) -> Dict[str, Any]:
        """Main training loop"""
        # - Setup optimizer
        # - Create data loaders
        # - Training epoch loop
        # - Validation after each epoch
        # - Save best model
        # - Return training results
    
    def train_epoch(self, model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Single training epoch"""
        # - Set model to train mode
        # - Iterate through batches
        # - Forward pass and loss computation
        # - Backward pass and optimization
        # - Return average loss
    
    def validate(self, model: torch.nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        # - Set model to eval mode
        # - Iterate through validation batches
        # - Compute predictions
        # - Calculate metrics
        # - Return validation metrics
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss"""
        # - Entity classification loss
        # - Relation classification loss
        # - Combine losses
        # - Return total loss
    
    # ⚡ P1 - Professional Methods
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, path: str) -> None:
        """Save training checkpoint"""
        # - Save model state
        # - Save optimizer state
        # - Save training metadata
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        # - Load checkpoint file
        # - Return model and optimizer states

### **File: `src/training/utils.py`**
**Purpose**: Training utilities and helpers

```python
# 🔥 P0 - Utility Functions
def prepare_batch(samples: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare batch for training"""
    # - Extract entity and relation labels
    # - Convert to tensors
    # - Move to device
    # - Return batch dictionary

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy"""
    # - Compute predicted classes
    # - Compare with targets
    # - Return accuracy percentage

def setup_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Setup optimizer from configuration"""
    # - Read optimizer config
    # - Create optimizer (Adam by default)
    # - Return configured optimizer
```

---

## 📊 **Module 4: Evaluation (`src/evaluation/`)**

### **File: `src/evaluation/evaluator.py`**
**Purpose**: Model evaluation on gold standard data

```python
class GoldStandardEvaluator:
    """Simple gold standard evaluation"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator"""
        # - Setup evaluation configuration
        # - Initialize metrics calculator
        # - Configure logging
    
    def evaluate(self, model: torch.nn.Module, gold_data: List[Data]) -> Dict[str, float]:
        """Evaluate model on gold standard"""
        # - Set model to eval mode
        # - Iterate through gold data
        # - Generate predictions
        # - Calculate metrics
        # - Return evaluation results
    
    def _extract_predictions(self, model_output: Dict[str, torch.Tensor]) -> Tuple[List[int], List[int]]:
        """Extract entity and relation predictions"""
        # - Convert logits to predictions
        # - Extract entity predictions
        # - Extract relation predictions
        # - Return prediction lists
    
    def _extract_targets(self, sample: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """Extract ground truth labels"""
        # - Extract entity labels
        # - Extract relation labels
        # - Return target lists
    
    # ⚡ P1 - Professional Methods
    def generate_report(self, results: Dict[str, float]) -> str:
        """Generate evaluation report"""
        # - Format results
        # - Create summary
        # - Return formatted report
    
    def save_results(self, results: Dict[str, float], output_path: str) -> None:
        """Save evaluation results"""
        # - Save as JSON
        # - Include timestamp
        # - Add metadata

# 🔥 P0 - CLI Integration
def main():
    """CLI entry point for make evaluate"""
    # - Parse command line arguments
    # - Load trained model
    # - Load gold data
    # - Run evaluation
    # - Save results
```

### **File: `src/evaluation/metrics.py`**
**Purpose**: Performance metrics calculation

```python
class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    # 🔥 P0 - Critical Methods
    def __init__(self):
        """Initialize metrics calculator"""
        # - Setup metric configurations
    
    def calculate_f1_score(self, predictions: List[int], targets: List[int]) -> float:
        """Calculate F1 score"""
        # - Use sklearn F1 score
        # - Handle edge cases
        # - Return F1 score
    
    def calculate_precision_recall(self, predictions: List[int], targets: List[int]) -> Tuple[float, float]:
        """Calculate precision and recall"""
        # - Use sklearn precision/recall
        # - Handle edge cases
        # - Return precision and recall
    
    def calculate_entity_metrics(self, entity_preds: List[int], entity_targets: List[int]) -> Dict[str, float]:
        """Calculate entity-specific metrics"""
        # - Entity F1 score
        # - Entity precision
        # - Entity recall
        # - Return metrics dictionary
    
    def calculate_relation_metrics(self, relation_preds: List[int], relation_targets: List[int]) -> Dict[str, float]:
        """Calculate relation-specific metrics"""
        # - Relation F1 score
        # - Relation precision
        # - Relation recall
        # - Return metrics dictionary
    
    # ⚡ P1 - Professional Methods
    def calculate_confusion_matrix(self, predictions: List[int], targets: List[int]) -> np.ndarray:
        """Calculate confusion matrix"""
        # - Use sklearn confusion matrix
        # - Return matrix
    
    def calculate_per_class_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics"""
        # - Per-class F1, precision, recall
        # - Return detailed breakdown
```

---

## ⚙️ **Module 5: Configuration (`config/`)**

### **File: `config/config.yaml`**
**Purpose**: Main configuration file

```yaml
# 🔥 P0 - Critical Configuration
model:
  embedding_dim: 384                    # LLM embedding dimension
  hidden_dim: 256                      # GNN hidden dimension
  num_gnn_layers: 2                    # Number of GNN layers
  dropout: 0.1                         # Dropout rate

# Complexity level mapping
complexity_levels:
  FG-0:
    entity_classes: 5                  # Root level entities
    relation_classes: 6
  FG-1:
    entity_classes: 34                 # Second level
    relation_classes: 6
  FG-2:
    entity_classes: 128                # Third level  
    relation_classes: 6
  FG-3:
    entity_classes: 224                # Full hierarchy
    relation_classes: 6

# Training configuration
training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 50
  patience: 5                          # Early stopping

# Data configuration
data:
  similarity_threshold: 0.7            # Graph edge threshold
  k_neighbors: 10                      # k-NN for graph building
  train_split: 0.8                     # Train/val split ratio

# File paths
paths:
  gold_corpus: "data/raw/gold_release.json"
  silver_corpus: "data/raw/silver_release.json"
  ontology: "data/raw/scheme.json"
  embeddings: "data/processed/embeddings.pkl"
  graphs: "data/processed/graphs.pkl"
  models: "results/models/"
```

---

## 🚀 **Module 6: Automation Scripts (`scripts/`)**

### **File: `scripts/train.py`**
**Purpose**: CLI training script

```python
# 🔥 P0 - Main Function
def main():
    """Main training script"""
    # - Parse command line arguments
    # - Load configuration
    # - Setup data pipeline
    # - Load/create embeddings and graphs
    # - Initialize model and trainer
    # - Run training
    # - Save results

# 🔥 P0 - Supporting Functions
def setup_data_pipeline(config: Dict[str, Any]) -> Tuple[List[Data], List[Data]]:
    """Setup complete data pipeline"""
    # - Load data with data_loader
    # - Generate embeddings if needed
    # - Build graphs if needed
    # - Create train/val splits
    # - Return train and val data

def initialize_training(config: Dict[str, Any]) -> Tuple[MaintIELLMGNNHybrid, SimpleTrainer]:
    """Initialize model and trainer"""
    # - Create model from config
    # - Create trainer from config
    # - Return model and trainer

if __name__ == "__main__":
    main()
```

### **File: `scripts/evaluate.py`**
**Purpose**: CLI evaluation script

```python
# 🔥 P0 - Main Function
def main():
    """Main evaluation script"""
    # - Parse command line arguments
    # - Load configuration
    # - Load trained model
    # - Load gold standard data
    # - Run evaluation
    # - Display and save results

# 🔥 P0 - Supporting Functions
def load_trained_model(model_path: str, config: Dict[str, Any]) -> MaintIELLMGNNHybrid:
    """Load trained model from checkpoint"""
    # - Load model architecture
    # - Load trained weights
    # - Return loaded model

def setup_gold_data(config: Dict[str, Any]) -> List[Data]:
    """Setup gold standard evaluation data"""
    # - Load gold corpus
    # - Generate embeddings
    # - Build graphs
    # - Return evaluation data

if __name__ == "__main__":
    main()
```

---

## 🧪 **Module 7: Testing (`tests/`)**

### **File: `tests/test_data_loading.py`**
```python
class TestDataLoading(unittest.TestCase):
    # 🔥 P0 - Critical Tests
    def test_load_gold_corpus(self):
        """Test gold corpus loading"""
    
    def test_load_silver_corpus(self):
        """Test silver corpus loading"""
    
    def test_data_validation(self):
        """Test data format validation"""
```

### **File: `tests/test_model.py`**
```python
class TestModel(unittest.TestCase):
    # 🔥 P0 - Critical Tests
    def test_model_initialization(self):
        """Test model creates correctly"""
    
    def test_forward_pass(self):
        """Test model forward pass"""
    
    def test_output_dimensions(self):
        """Test output shapes are correct"""
```

### **File: `tests/test_training.py`**
```python
class TestTraining(unittest.TestCase):
    # 🔥 P0 - Critical Tests
    def test_training_loop(self):
        """Test training executes without errors"""
    
    def test_loss_computation(self):
        """Test loss calculation"""
    
    def test_model_saving(self):
        """Test model checkpoint saving"""
```

---

## 📋 **Makefile Implementation**

### **File: `Makefile`**
```makefile
# 🔥 P0 - Critical Commands
setup:                ## Initialize project environment
	python -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

load-data:            ## Load MaintIE data files
	python -c "from src.data_processing.data_loader import main; main()"

generate-embeddings:  ## Generate LLM embeddings
	python -c "from src.data_processing.embedding_generator import main; main()"

build-graphs:         ## Build semantic similarity graphs
	python -c "from src.data_processing.graph_builder import main; main()"

train:                ## Run training pipeline
	python scripts/train.py

evaluate:             ## Run evaluation
	python scripts/evaluate.py

test:                 ## Run unit tests
	python -m pytest tests/

# ⚡ P1 - Professional Commands
status:               ## Show project status
clean:                ## Clean generated files
pipeline:             ## Run complete pipeline (load-data → train → evaluate)
```

---

## ✅ **Implementation Success Criteria**

### **Working Pipeline Test:**
```bash
make setup
make load-data
make generate-embeddings
make build-graphs
make train
make evaluate
# Expected Output: Entity F1 > 0.6, Relation F1 > 0.4
```

### **File Count Verification:**
- **10 core implementation files** ✅
- **3 test files** ✅  
- **2 script files** ✅
- **1 config file** ✅
- **1 Makefile** ✅
- **Total: ~17 files** ✅

### **Code Quality Gates:**
- **Every class has clear purpose** ✅
- **Every method has specific functionality** ✅
- **Professional error handling** ✅
- **Clean separation of concerns** ✅
- **Testable components** ✅

**This implementation provides the simplest working MaintIE system while maintaining professional architecture standards for future enhancement.**