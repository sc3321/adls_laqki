from pydantic import BaseModel, ConfigDict
from typing import List
import numpy as np 

class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    num_samples: int = 256
    sequence_length: int = 2048
    datasets: List[str] = ["c4", "pile"]
    dataset_weights: List[float] = [0.5, 0.5]
    domain_augmentation: List[str] = []     # [code, math, multilingual]
    seed: int = 42
    cache_dir: str = ".qpe_cache/calibration"
    validation_split: float = 0.2           # Held-out fraction for ValidationEngine


class ImportanceMatrix(BaseModel):
    """
    Per-weight-group importance scores computed via sum of squared activations
    
    Follows llama.cpp imatrix methodology: for each weight group g in each layer, 
    I_g = Sum_x ||a_g(x)||^2
    Weight groups with high I_g carry more information and should be quantized more carefully
    
    Used by ConfigurationExporter when applying AWQ/GPTQ - passes importance scores 
    to quantization algorithm so salient weight groups receive proportionally more 
    precision budget within their assigned bit-width
    
    NOT used by solver - solver operates at layer granularity
    Importance matrix operates at sub-layer (weight-group) granularity and affects 
    how quantization is applied within a layer, not which precision the layer receives
    """
    model_config = ConfigDict(frozen=True)
    
    model_id: str
    num_calibration_samples: int
    sequence_length: int
    
    # {layer_name: np.ndarray of shape (num_weight_groups,)}
    # Stored as lists for Pydantic serialization; convert to numpy on use.
    scores: dict[str, list[float]]
    
    # Metadata per layer for interpreting the scores
    group_size: int = 128                   # Weight group size (matches AWQ/GPTQ default)
    
    def to_numpy(self) -> dict[str, np.ndarray]:
        """Convert to numpy arrays for quantization backends"""
        return {name: np.array(vals) for name, vals in self.scores.items()}
    
    def save(self, path: str) -> None:
        """Save as npz file (numpy compressed archive)"""
        np.savez_compressed(path, **self.to_numpy())
    
    @classmethod
    def load(cls, path: str, model_id: str, num_samples: int, seq_len: int) -> "ImportanceMatrix":
        """Load from npz file"""
        data = dict(np.load(path))
        return cls(
            model_id=model_id,
            num_calibration_samples=num_samples,
            sequence_length=seq_len,
            scores={k: v.tolist() for k, v in data.items()},
        )