from typing import List, Dict, Optional, Tuple, Union, Any, AsyncIterator
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os
import pickle
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import uuid

# FastAPI and web components
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Azure ML components
try:
    from azureml.core import Workspace, Model, Environment
    from azureml.core.webservice import AciWebservice, AksWebservice
    from azureml.core.model import InferenceConfig
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

# Model and processing imports
import torch
import numpy as np
from ..models.llm_gnn_hybrid import MaintIELLMGNNHybrid
from ..data_processing.data_pipeline import MaintIEDataProcessor
from ..evaluation.metrics_calculator import MaintenanceMetricsCalculator
from ..core.config import ConfigManager
from ..utils.file_handlers import JSONHandler, PickleHandler

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()

# CORS and GZip middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency for Azure ML workspace
def get_azure_ml_workspace():
    try:
        workspace = Workspace.from_config()
        return workspace
    except Exception as e:
        logger.error(f"Error loading Azure ML workspace: {e}")
        raise HTTPException(status_code=500, detail="Azure ML workspace not available")


# Pydantic models for request and response
class ModelRequest(BaseModel):
    data: List[float]


class ModelResponse(BaseModel):
    predictions: List[float]


# FastAPI route for health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# FastAPI route for model inference
@app.post(
    "/predict",
    response_model=ModelResponse,
    summary="Run model inference",
    description="This endpoint accepts input data for the model and returns the predictions.",
)
async def predict(
    request: ModelRequest,
    background_tasks: BackgroundTasks,
    workspace: Workspace = Depends(get_azure_ml_workspace),
):
    # Log request data
    logger.info(f"Received request: {request.json()}")
    
    # Background task example: log to a file
    background_tasks.add_task(log_request_data, request.json())

    # Model inference logic
    try:
        model = Model.get_model_path("your_model_name", workspace=workspace)
        # Add your model inference code here
        predictions = [0.0]  # Dummy prediction
        response = ModelResponse(predictions=predictions)
        return response
    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        raise HTTPException(status_code=500, detail="Model inference error")


# Background task example: log request data to a file
def log_request_data(data: str):
    file_path = "request_logs.txt"
    with open(file_path, "a") as file:
        file.write(f"{datetime.now()}: {data}\n")


# Azure ML deployment example
@app.post("/deploy")
async def deploy_model(workspace: Workspace = Depends(get_azure_ml_workspace)):
    try:
        # Load model
        model = Model.register(
            workspace=workspace,
            model_path="path/to/your/model",
            model_name="your_model_name",
            tags={"key": "value"},
            description="Model description",
        )

        # Create inference configuration
        inference_config = InferenceConfig(
            entry_script="score.py",
            runtime="python",
            conda_file="env.yml",
        )

        # Deploy model to ACI
        aci_config = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            auth_enabled=True,
        )
        service = Model.deploy(
            workspace=workspace,
            name="your-service-name",
            models=[model],
            inference_config=inference_config,
            deployment_config=aci_config,
            overwrite=True,
        )
        service.wait_for_deployment(show_output=True)
        return {"status": "Model deployed", "scoring_uri": service.scoring_uri}
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail="Model deployment error")


# Azure ML inference example
@app.post("/azureml-inference")
async def azureml_inference(
    request: ModelRequest,
    workspace: Workspace = Depends(get_azure_ml_workspace),
):
    try:
        # Load model
        model = Model.get_model_path("your_model_name", workspace=workspace)

        # Prepare input data
        input_data = request.data

        # Run inference
        # Add your Azure ML inference code here
        predictions = [0.0]  # Dummy prediction
        response = ModelResponse(predictions=predictions)
        return response
    except Exception as e:
        logger.error(f"Error during Azure ML inference: {e}")
        raise HTTPException(status_code=500, detail="Azure ML inference error")


# Docker integration example
@app.post("/docker-integration")
async def docker_integration():
    try:
        # Docker build and run commands
        os.system("docker build -t your_image_name .")
        os.system("docker run -d -p 8000:80 your_image_name")
        return {"status": "Docker container deployed"}
    except Exception as e:
        logger.error(f"Error with Docker integration: {e}")
        raise HTTPException(status_code=500, detail="Docker integration error")


# Example usage
@app.get("/example-usage")
async def example_usage():
    return {
        "message": "This is an example usage endpoint.",
        "available_endpoints": ["/health", "/predict", "/deploy", "/azureml-inference", "/docker-integration"],
    }


# Entry point for uvicorn
if __name__ == "__main__":
    import sys

    # Check for --reload flag
    reload = "--reload" in sys.argv

    # Run the app with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=reload,
    )