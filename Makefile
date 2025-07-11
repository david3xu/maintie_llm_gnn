# Makefile for MaintIE LLM-GNN Project

.PHONY: help setup install-deps load-data generate-embeddings build-graphs train evaluate test status clean pipeline

# Define the Python interpreter from the virtual environment.
# This makes the Makefile more robust.
PYTHON = venv/bin/python
export PYTHONPATH=$(shell pwd)

# Default target
help:
	@echo "Available commands:"
	@echo "  setup                - Initialize project environment (creates venv)"
	@echo "  install-deps         - Install Python dependencies from requirements.txt"
	@echo "  load-data            - (Placeholder) Run data loading script"
	@echo "  generate-embeddings  - Generate LLM embeddings from raw data"
	@echo "  build-graphs         - Build semantic similarity graphs from embeddings"
	@echo "  train                - Run the training pipeline"
	@echo "  evaluate             - Run the evaluation pipeline"
	@echo "  test                 - Run all unit tests"
	@echo "  pipeline             - Run the full pipeline (data -> train -> evaluate)"
	@echo "  status               - (Placeholder) Show project status"
	@echo "  clean                - (Placeholder) Clean generated files"

# Setup commands
setup:
	@echo "Initializing Python virtual environment..."
	python3 -m venv venv
	@echo "Virtual environment 'venv' created."
	@echo "Activate it by running: source venv/bin/activate"

install-deps:
	@echo "Installing dependencies from requirements.txt..."
	@if [ -f "venv/bin/pip" ]; then \
		$(PYTHON) -m pip install -r requirements.txt; \
	else \
		echo "Virtual environment not found. Please run 'make setup' first."; \
	fi

# Data processing commands
load-data:
	@echo "Running data loading script..."
	$(PYTHON) -c "from src.data_processing.data_loader import main; main()"

generate-embeddings:
	@echo "Generating LLM embeddings..."
	$(PYTHON) -c "from src.data_processing.embedding_generator import main; main()"

build-graphs:
	@echo "Building semantic similarity graphs..."
	$(PYTHON) -c "from src.data_processing.graph_builder import main; main()"

# Training and Evaluation commands
train:
	@echo "Running training pipeline..."
	$(PYTHON) scripts/train.py

evaluate:
	@echo "Running evaluation pipeline..."
	$(PYTHON) scripts/evaluate.py

# Testing command
test:
	@echo "Running unit tests..."
	$(PYTHON) -m unittest discover -s tests

# Full pipeline
pipeline: load-data generate-embeddings build-graphs train evaluate

# Status and Clean commands (placeholders)
status:
	@echo "Project status: (to be implemented)"

clean:
	@echo "Cleaning generated files: (to be implemented)"
	rm -f data/processed/*.pkl
	rm -f results/models/*.pt
	rm -f results/evaluation/*.json
	find . -type d -name "__pycache__" -exec rm -r {} +
	@echo "Cleaned processed data, models, results, and __pycache__."
