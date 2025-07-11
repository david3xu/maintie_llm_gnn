from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

class DataSplitter:
    """Train/val/test split utility for MaintIE corpus"""
    def __init__(self, train_ratio: float = 0.8, val_ratio: float = 0.1, random_state: int = 42):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

    def split(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        train_data, temp_data = train_test_split(data, train_size=self.train_ratio, random_state=self.random_state)
        val_size = self.val_ratio / (1 - self.train_ratio)
        val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=self.random_state)
        return train_data, val_data, test_data
