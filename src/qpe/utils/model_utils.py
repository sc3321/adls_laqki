import torch
from transformers import PreTrainedModel
import torch.nn as nn 
from typing import List, Dict, Tuple 
import copy 

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

def get_layer_names(
    model : nn.Module 
) -> List[str] : 
    return [name for name, _ in model.named_modules() if name]

def _quantize_layer(layer: nn.Module, precision, gpu_spec) -> nn.Module:
    from .types import Precision 

    if precision == Precision.FP16:
        return layer

    if precision == Precision.W8A8_FP8 and not gpu_spec.supports_fp8:
        return layer

    # torchao int4_weight_only kernels require sm_80+ (Ampere); T4 is sm_75
    if precision == Precision.W4A16 and gpu_spec.compute_capability < (8, 0):
        return layer

    try:
        from torchao.quantization import (
            float8_dynamic_activation_float8_weight,
            int4_weight_only,
            int8_dynamic_activation_int8_weight,
            quantize_,
        )
    except ImportError:
        return layer

    try:
        q = copy.deepcopy(layer)
        if precision == Precision.W8A8_FP8:
            quantize_(q, float8_dynamic_activation_float8_weight())
        elif precision == Precision.W8A8_INT8:
            quantize_(q, int8_dynamic_activation_int8_weight())
        elif precision == Precision.W4A16:
            quantize_(q, int4_weight_only(group_size=128))
        return q
    except Exception as e:
        return layer


def _resolve_layers(model: nn.Module, layer_names: List[str]) -> Dict[str, nn.Module]:
    """
    Extract named modules form model
    """
    modules = {}
    for name in layer_names:
        try:
            modules[name] = model.get_submodule(name)
        except AttributeError:
            raise KeyError(
                f"Model has no layer {name} \n -> Available : {[n for n, _ in model.named_children()]}"
            )
    return modules


def _get_layer_shape(module: nn.Module) -> Tuple[int, ...]:
    if isinstance(module, nn.Linear):
        return (module.out_features, module.in_features)
    # TODO : extend for other module types
    return tuple(next(module.parameters()).shape)


def _get_layer_dtype(layer: nn.Module) -> str:
    # Check parameters first, then buffers
    tensors = list(layer.parameters()) + list(layer.buffers())
    if not tensors:
        return "unknown"

    return str(tensors[0].dtype)