from pydantic import BaseModel, ConfigDict
from typing import List

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

