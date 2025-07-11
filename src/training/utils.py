from typing import Any, Dict, List
import torch
import torch.optim as optim

def prepare_batch(samples: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare batch for training by converting samples to tensors."""
    # This is a placeholder. In a real scenario, you would extract entity and
    # relation labels from the samples, convert them to tensors, and move
    # them to the specified device. The structure of the samples and how
    # labels are represented will determine the implementation details.

    # Example placeholder structure:
    # B = len(samples)
    # N = max(len(s['entities']) for s in samples)
    # entity_labels = torch.zeros((B, N), dtype=torch.long)

    # For now, returning an empty dictionary as the structure is not yet defined.
    return {}

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate classification accuracy."""
    if targets.numel() == 0:
        return 0.0
    predicted_classes = torch.argmax(predictions, dim=1)
    correct_predictions = (predicted_classes == targets).sum().item()
    accuracy = correct_predictions / targets.numel()
    return accuracy

def setup_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Setup optimizer from configuration."""
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 1e-3)

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
