# MaintIE LLM-GNN: Simple Working Directory Structure

**Code Priority**: Working end-to-end pipeline first  
**Start Simple**: Minimal but complete implementation  
**Professional**: Clean architecture with proper separation  
**Good Lifecycle**: Development → Testing → Production workflow

---

## 📁 **Complete Working Structure (Week 1)**

```
maintie_llm_gnn/                               # Root project
│
├── 📊 data/                                   # Data storage
│   ├── raw/                                   # Original MaintIE files
│   │   ├── gold_release.json                 # ~1000 expert samples
│   │   ├── silver_release.json               # ~7000 auto samples  
│   │   └── scheme.json                       # Entity/relation ontology
│   ├── processed/                             # Generated artifacts
│   │   ├── embeddings.pkl                    # LLM embeddings
│   │   ├── graphs.pkl                        # Built graphs
│   │   └── splits.pkl                        # Train/val splits
│   └── README.md                              # Data documentation
│
├── 🧠 src/                                    # Core implementation
│   ├── __init__.py
│   ├── data_processing/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── data_loader.py                     # 🔥 Simple JSON loading
│   │   ├── embedding_generator.py             # 🔥 LLM embeddings  
│   │   └── graph_builder.py                  # 🔥 Basic graph construction
│   ├── models/                                # Model architecture
│   │   ├── __init__.py
│   │   ├── llm_gnn_hybrid.py                 # 🔥 Main hybrid model
│   │   └── simple_gnn.py                     # 🔥 Basic GNN implementation
│   ├── training/                              # Training system
│   │   ├── __init__.py
│   │   ├── simple_trainer.py                 # 🔥 Basic training loop
│   │   └── utils.py                          # Training utilities
│   └── evaluation/                            # Evaluation system
│       ├── __init__.py
│       ├── evaluator.py                      # 🔥 Basic evaluation
│       └── metrics.py                        # Performance metrics
│
├── ⚙️ config/                                 # Configuration
│   ├── config.yaml                           # 🔥 Main configuration
│   └── complexity_levels.yaml                # Entity complexity mappings
│
├── 🧪 tests/                                 # Testing framework
│   ├── __init__.py
│   ├── test_data_loading.py                  # Data pipeline tests
│   ├── test_model.py                         # Model tests
│   └── test_training.py                      # Training tests
│
├── 📊 results/                               # Experiment results
│   ├── training/                             # Training outputs
│   ├── evaluation/                           # Evaluation results
│   └── models/                               # Saved model checkpoints
│
├── 🚀 scripts/                               # Automation scripts
│   ├── train.py                              # 🔥 Training script
│   ├── evaluate.py                           # 🔥 Evaluation script
│   └── setup.py                              # Environment setup
│
├── Makefile                                   # 🔥 Automation commands
├── requirements.txt                           # Python dependencies
├── .env.example                              # Environment template
├── .gitignore                                # Git ignore rules
└── README.md                                 # Project documentation
```

---

## 🔥 **Core Working Files (Minimum Viable Product)**

### **Essential Implementation Files:**
1. **`src/data_processing/data_loader.py`** - Simple JSON loading
2. **`src/data_processing/embedding_generator.py`** - Basic LLM embeddings
3. **`src/data_processing/graph_builder.py`** - Simple semantic similarity graphs
4. **`src/models/llm_gnn_hybrid.py`** - Working hybrid model
5. **`src/training/simple_trainer.py`** - Basic training loop
6. **`src/evaluation/evaluator.py`** - Simple F1 evaluation

### **Essential Configuration:**
7. **`config/config.yaml`** - Working configuration
8. **`Makefile`** - Automation commands

### **Essential Scripts:**
9. **`scripts/train.py`** - CLI training
10. **`scripts/evaluate.py`** - CLI evaluation

---

## 🚀 **Complete Working Workflow**

### **Step 1: Setup & Data Loading**
```bash
# Environment setup
make setup
make install-deps

# Data pipeline
make load-data          # src.data_processing.data_loader
make generate-embeddings # src.data_processing.embedding_generator  
make build-graphs       # src.data_processing.graph_builder
```

### **Step 2: Training**
```bash
# Simple training
make train              # scripts/train.py → src.training.simple_trainer
```

### **Step 3: Evaluation**
```bash
# Basic evaluation
make evaluate           # scripts/evaluate.py → src.evaluation.evaluator
```

### **Step 4: Testing**
```bash
# Validation
make test               # Run tests/ directory
```

---

## 📋 **Makefile Commands (Complete Workflow)**

```makefile
# Setup commands
setup:          ## Initialize project environment
install-deps:   ## Install Python dependencies
load-data:      ## Load MaintIE data files

# Data processing commands  
generate-embeddings:  ## Create LLM embeddings
build-graphs:         ## Build semantic similarity graphs

# Training commands
train:          ## Run basic training
train-simple:   ## Alias for train

# Evaluation commands
evaluate:       ## Run basic evaluation
evaluate-gold:  ## Evaluate on gold standard

# Testing commands
test:           ## Run unit tests
test-pipeline:  ## Test end-to-end pipeline

# Status commands
status:         ## Show project status
clean:          ## Clean generated files
```

---

## 🎯 **Professional Architecture Principles**

### **Code Priority** ✅
- **Working first**: Each file has simple, functional implementation
- **No templates**: Real code that loads data, trains models, produces results
- **Testable**: Every core component has corresponding test

### **Start Simple** ✅
- **Single-purpose files**: Each file does one thing well
- **Minimal dependencies**: Core workflow with essential libraries only
- **Basic algorithms**: Simple similarity, basic GNN, straightforward training

### **Professional** ✅
- **Clean separation**: Data, models, training, evaluation clearly separated
- **Proper imports**: `__init__.py` files for clean module structure
- **Configuration management**: YAML configs, not hardcoded values
- **Documentation**: README files explaining each component

### **Good Lifecycle** ✅
- **Development**: `src/` for implementation, `config/` for settings
- **Testing**: `tests/` directory with unit and integration tests
- **Production**: `scripts/` for CLI, `results/` for outputs
- **Automation**: `Makefile` for reproducible workflows

---

## ✅ **Success Criteria**

### **End-to-End Working Pipeline:**
```bash
# This complete workflow must work:
make setup
make load-data
make generate-embeddings  
make build-graphs
make train
make evaluate
# Output: Entity F1 score, Relation F1 score, trained model saved
```

### **File Count: ~20 files total**
- **10 core implementation files** (data, models, training, evaluation)
- **3 configuration files** (config, requirements, Makefile)
- **4 test files** (data, model, training, evaluation tests)
- **3 automation files** (train, evaluate, setup scripts)

### **Complexity: Minimal but complete**
- **No advanced features** (curriculum learning, caching, monitoring)
- **Basic algorithms** (cosine similarity, simple GNN, Adam optimizer)
- **Essential functionality** (load → embed → graph → train → evaluate)

**This structure provides the simplest possible implementation that works end-to-end while maintaining professional architecture for future enhancement.**