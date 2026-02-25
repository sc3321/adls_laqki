from .models import CalibrationConfig
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np
from transformers import PreTrainedTokenizer

class CalibrationDataManager:
    def __init__(self, config: CalibrationConfig, tokenizer: PreTrainedTokenizer): ...
    
    def get_dataloader(self, batch_size: int = 4) -> DataLoader:
        """Training split - used by SensitivityScorer."""
        ...
    
    def get_validation_dataloader(self, batch_size: int = 4) -> DataLoader:
        """Held-out split - used by ValidationEngine for perplexity screening."""
        ...
    
    def compute_importance_matrix(self, model: nn.Module) -> dict[str, np.ndarray]:
        """
        Sum-of-squared activations per weight group (llama.cpp imatrix approach)
        Used by ConfigurationExporter for importance-weighted quantization
        """
        ...