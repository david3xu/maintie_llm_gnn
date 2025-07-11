from typing import List, Dict, Optional, Tuple, Union, Any, Set
import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime
import statistics
import unicodedata

try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    from pydantic import BaseModel, validator, ValidationError as PydanticValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from ..core.config import ConfigManager
from ..utils.file_handlers import JSONHandler

logger = logging.getLogger(__name__)

# Define custom exception for data validation
class DataValidationError(Exception):
    pass

# Base class for all data validators
class DataValidator:
    def __init__(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]):
        self.data = self._convert_to_dataframe(data)

    def _convert_to_dataframe(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, list) and all(isinstance(i, dict) for i in data):
            return pd.DataFrame(data)
        else:
            raise ValueError("Data must be a pandas DataFrame or a list of dictionaries.")

    def validate(self) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

# Example subclass for validating JSON data
class JSONDataValidator(DataValidator):
    def __init__(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], schema: Dict[str, Any]):
        super().__init__(data)
        self.schema = schema

    def validate(self) -> None:
        if not JSONSCHEMA_AVAILABLE:
            raise RuntimeError("jsonschema package is not available.")
        try:
            validate(instance=self.data.to_dict(orient="records"), schema=self.schema)
        except ValidationError as e:
            raise DataValidationError(f"JSON Schema validation error: {e.message}")

# Example subclass for validating CSV data
class CSVDataValidator(DataValidator):
    def __init__(self, data: Union[pd.DataFrame, List[Dict[str, Any]]], required_columns: List[str]):
        super().__init__(data)
        self.required_columns = required_columns

    def validate(self) -> None:
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

class MaintIEValidator:
    """MaintIE format validator for corpus samples"""
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        required_fields = ['text', 'entities', 'relations']
        for field in required_fields:
            if field not in sample:
                self.logger.warning(f"Missing required field: {field}")
                return False
        if not isinstance(sample['text'], str) or not sample['text'].strip():
            self.logger.warning("Invalid or empty text")
            return False
        for entity in sample['entities']:
            if not isinstance(entity.get('type', None), str):
                self.logger.warning("Invalid entity type")
                return False
            if entity.get('start', 0) >= entity.get('end', 0):
                self.logger.warning("Invalid entity span")
                return False
        entity_count = len(sample['entities'])
        for relation in sample['relations']:
            if not (0 <= relation.get('head', -1) < entity_count and 0 <= relation.get('tail', -1) < entity_count):
                self.logger.warning("Invalid relation entity indices")
                return False
        return True

    def validate_corpus(self, corpus: List[Dict[str, Any]]) -> List[int]:
        """Return indices of invalid samples"""
        invalid_indices = []
        for i, sample in enumerate(corpus):
            if not self.validate_sample(sample):
                invalid_indices.append(i)
        return invalid_indices

# Example usage
if __name__ == "__main__":
    # Sample JSON schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age", "email"]
    }

    # Sample data
    data = [
        {"name": "John Doe", "age": 30, "email": "john.doe@example.com"},
        {"name": "Jane Doe", "age": 25, "email": "jane.doe@example.com"}
    ]

    # Validate JSON data
    json_validator = JSONDataValidator(data, schema)
    try:
        json_validator.validate()
        print("JSON data is valid.")
    except DataValidationError as e:
        print(f"JSON data validation error: {e}")

    # Validate CSV data
    csv_data = pd.DataFrame(data)
    csv_validator = CSVDataValidator(csv_data, required_columns=["name", "age"])
    try:
        csv_validator.validate()
        print("CSV data is valid.")
    except DataValidationError as e:
        print(f"CSV data validation error: {e}")