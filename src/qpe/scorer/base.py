from typing import Protocol
import torch.nn as nn 
from solver.models import LayerDescriptor
from torch.utils.data.dataloader import DataLoader

class SensitivityScorer(Protocol):
    def score(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: list[str],
    ) -> list[LayerDescriptor]:
        """
        Populate sensitivity fields in LayerDescriptor for each layer.
        Resource fields (memory_bytes, latency_us) are left empty the HardwareProfiler fills those separately.
        
        Returns partial LayerDescriptors with sensitivity fields populated.
        """
        ...
