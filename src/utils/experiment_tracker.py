from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import os
import json
import pickle
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import tempfile
import shutil
from contextlib import contextmanager
import uuid

# MLOps and experiment tracking imports
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from azureml.core import Run, Experiment, Workspace, Model
    from azureml.core.model import Model as AzureModel
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import torch
import numpy as np
import pandas as pd
from ..core.config import ConfigManager
from ..utils.file_handlers import JSONHandler, PickleHandler

@dataclass
class ExperimentConfig:
    experiment_name: str
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[Dict[str, str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None

class TrackingBackend(Enum):
    MLFLOW = "mlflow"
    WANDB = "wandb"
    AZURE_ML = "azure_ml"
    TENSORBOARD = "tensorboard"
    LOCAL = "local"

class ExperimentTracker:
    def __init__(self, config: ExperimentConfig, backend: TrackingBackend):
        self.config = config
        self.backend = backend
        self.run_id = None
        self.client = None
        self.logger = None
        self._setup_backend()

    def _setup_backend(self):
        if self.backend == TrackingBackend.MLFLOW and MLFLOW_AVAILABLE:
            import mlflow
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            self.client = MlflowClient()
            self.logger = mlflow
        elif self.backend == TrackingBackend.WANDB and WANDB_AVAILABLE:
            import wandb
            wandb.init(project=self.config.experiment_name, name=self.config.run_name)
            self.client = wandb
            self.logger = wandb
        elif self.backend == TrackingBackend.AZURE_ML and AZURE_ML_AVAILABLE:
            from azureml.core import Workspace
            ws = Workspace.from_config()
            self.client = ws
            self.logger = ws
        elif self.backend == TrackingBackend.TENSORBOARD and TENSORBOARD_AVAILABLE:
            from tensorboardX import SummaryWriter
            self.client = SummaryWriter(logdir=f"runs/{self.config.experiment_name}")
            self.logger = self.client
        elif self.backend == TrackingBackend.LOCAL:
            self.client = None
            self.logger = None
        else:
            raise ImportError("Required module for the selected backend is not available.")

    def start_run(self):
        if self.backend == TrackingBackend.MLFLOW:
            self.run_id = self.client.create_experiment(self.config.experiment_name)
            self.client.start_run(run_id=self.run_id)
        elif self.backend == TrackingBackend.WANDB:
            self.run_id = self.client.run.id
        elif self.backend == TrackingBackend.AZURE_ML:
            from azureml.core import Experiment
            experiment = Experiment(workspace=self.client, name=self.config.experiment_name)
            self.run_id = str(uuid.uuid4())
            self.client = experiment.start_logging()
        elif self.backend == TrackingBackend.TENSORBOARD:
            self.run_id = str(uuid.uuid4())
        elif self.backend == TrackingBackend.LOCAL:
            self.run_id = str(uuid.uuid4())
        else:
            raise ValueError("Invalid tracking backend specified.")

        self.config.start_time = datetime.now()
        self.config.status = "running"

    def log_param(self, key: str, value: Any):
        if self.logger is not None:
            self.logger.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.logger is not None:
            self.logger.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, key: str, value: str):
        if self.logger is not None:
            self.logger.log_artifact(key, value)

    def log_artifacts(self, artifacts: Dict[str, str]):
        for key, value in artifacts.items():
            self.log_artifact(key, value)

    def end_run(self, status: Optional[str] = "finished"):
        if self.backend == TrackingBackend.MLFLOW:
            self.client.set_tag("mlflow.runName", self.config.run_name)
            self.client.set_terminated(self.run_id, status)
        elif self.backend == TrackingBackend.WANDB:
            self.client.finish()
        elif self.backend == TrackingBackend.AZURE_ML:
            self.client.complete()
        elif self.backend == TrackingBackend.TENSORBOARD:
            self.client.close()
        elif self.backend == TrackingBackend.LOCAL:
            pass
        else:
            raise ValueError("Invalid tracking backend specified.")

        self.config.end_time = datetime.now()
        self.config.status = status

    def get_run_url(self) -> str:
        if self.backend == TrackingBackend.MLFLOW:
            return f"http://127.0.0.1:5000/#/experiments/{self.config.experiment_name}/runs/{self.run_id}"
        elif self.backend == TrackingBackend.WANDB:
            return f"https://wandb.ai/your_username/{self.config.experiment_name}/runs/{self.run_id}"
        elif self.backend == TrackingBackend.AZURE_ML:
            return f"https://ml.azure.com/runs/{self.run_id}?wsid={self.client.workspace_id}&tid={self.client.subscription_id}"
        elif self.backend == TrackingBackend.TENSORBOARD:
            return f"http://localhost:6006/#scalars&_smoothing=0&run={self.run_id}"
        else:
            return "No URL available for the selected backend."

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log_param("error", str(exc_val))
            self.log_param("traceback", str(exc_tb))
            self.end_run(status="failed")
        else:
            self.end_run(status="finished")

# Example usage
if __name__ == "__main__":
    config = ExperimentConfig(
        experiment_name="my_experiment",
        run_name="run_1",
        tags={"tag1": "value1"},
        description="This is a test experiment",
        parameters={"param1": 10, "param2": 0.1},
        metrics={"metric1": 0.95},
        artifacts={"artifact1": "path/to/artifact"},
    )

    with ExperimentTracker(config, TrackingBackend.MLFLOW) as tracker:
        tracker.log_params(config.parameters)
        tracker.log_metrics(config.metrics)
        tracker.log_artifacts(config.artifacts)

        # Simulate training loop
        for epoch in range(5):
            time.sleep(1)
            tracker.log_metric("loss", np.random.random(), step=epoch)
            tracker.log_metric("accuracy", np.random.random(), step=epoch)

        print("Run URL:", tracker.get_run_url())