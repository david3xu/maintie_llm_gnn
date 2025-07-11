import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Generator
from pathlib import Path
import tempfile
import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass
import warnings
import sys
import os

# Property-based testing
try:
    from hypothesis import given, strategies as st, assume, settings, Verbosity
    from hypothesis.extra.numpy import arrays
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Performance testing
import timeit
import memory_profiler
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.llm_gnn_hybrid import MaintIELLMGNNHybrid
from src.models.embedders.maintenance_llm_embedder import MaintenanceLLMEmbedder
from src.data_processing.graph_builder import MaintenanceGraphBuilder
from src.models.graph_networks.maintenance_gnn import MaintenanceGNN
from src.training.trainer import MaintIETrainer
from src.evaluation.gold_evaluator import GoldStandardEvaluator
from src.evaluation.metrics_calculator import MaintenanceMetricsCalculator
from src.data_processing.data_pipeline import MaintIEDataProcessor
from src.data_processing.data_validator import DataValidator
from src.core.config import ConfigManager
from src.utils.experiment_tracker import MaintIEExperimentTracker

# ...existing code...
# (Full implementation as provided in the user request)
# ...existing code...
