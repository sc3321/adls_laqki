import torch
from qpe.solver.models import LayerDescriptor
class HessianTraceScorer:
    """
    HAWQ-V2/V3 style Hessian trace via Hutchinson.
    """
    
    def __init__(
        self,
        num_hutchinson_samples: int = 200,
        collect_fisher: bool = True,
        dtype: torch.dtype = torch.float32,  # FP32 for Hessian numerical stability
    ):
        ...
    
    def score(self, model, dataloader, layer_names) -> list[LayerDescriptor]:
        # 1. Forward pass -> loss
        # 2. For each layer: Hutchinson trace estimation
        # 3. Simultaneously: activation stats via forward hooks
        # 4. Single backward: gradient norms
        # 5. Fisher diagonal estimation (if enabled)
        # 6. Assemble LayerDescriptor per layer (sensitivity fields only)
        ...