import torch
from transformers import PreTrainedModel
import torch.nn as nn 

def load_model(
    model_id: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    revision: str | None = None,
) -> PreTrainedModel:
    """
    Load a HuggingFace model with appropriate device mapping
    
    For sensitivity scoring (Hessian computation), caller should override dtype=torch.float32
    Hessian-vector products accumulate catastrophic rounding error in FP16
    For all other uses, FP16 is sufficient and halves memory requirements
    
    Args:
        model_id: HuggingFace model ID or local path to model directory
        dtype: Torch dtype for model weights
        device_map: Accelerate device map strategy (auto for multi-GPU, cuda:0 for single GPU, cpu for CPU-only)
        revision: Model revision (git commit hash or branch name)
    
    Returns:
        Loaded model in eval mode with gradients enabled for Hessian computation
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
        revision=revision,
    )
    model.eval()
    # Note: the scorer needs gradients enabled for Hessian-vector product computation.
    return model



QUANTIZABLE_MODULE_TYPES = (torch.nn.Linear,)

EXCLUDED_LAYER_PATTERNS = [
    "embed_tokens",
    "lm_head",
    "norm",
    "rotary_emb",
]


def get_quantizable_layers(model: nn.Module) -> list[str]:
    """
    Enumerate all quantizable layers in a model by name
    
    Returns list of fully qualified layer names that are candidates for mixed-precision quantization
    
    A layer is quantizable if:
    1. It is an instance of a quantizable module type (nn.Linear)
    2. Its name does not match any excluded pattern (embeddings, norms)
    3. It has > 0 parameters (skip empty/placeholder modules)
    
    Returned list is deterministic and ordered by model traversal order 
    (matches sequential execution order for standard transformer blocks)
    """
    quantizable = []
    for name, module in model.named_modules():
        if not isinstance(module, QUANTIZABLE_MODULE_TYPES):
            continue
        if any(pattern in name for pattern in EXCLUDED_LAYER_PATTERNS):
            continue
        if sum(p.numel() for p in module.parameters()) == 0:
            continue
        quantizable.append(name)
    return quantizable