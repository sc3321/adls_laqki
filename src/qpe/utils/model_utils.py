import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    revision: str | None = None,
) -> PreTrainedModel:
    """
    Load a HuggingFace model with appropriate device mapping.

    For sensitivity scoring (Hessian computation), caller should override
    dtype=torch.float32 because Hessian-vector products can accumulate severe
    rounding error in FP16.

    For most other stages, FP16 is sufficient and reduces memory footprint.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map=device_map,
        revision=revision,
    )
    model.eval()
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
    Enumerate all quantizable layers in a model by name.

    A layer is quantizable if:
      1. it is an instance of a quantizable module type
      2. its name does not match any excluded pattern
      3. it has nonzero parameters
    """
    quantizable: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, QUANTIZABLE_MODULE_TYPES):
            continue
        if any(pattern in name for pattern in EXCLUDED_LAYER_PATTERNS):
            continue
        if sum(p.numel() for p in module.parameters()) == 0:
            continue
        quantizable.append(name)
    return quantizable


def get_layer_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_modules() if name]


def _quantize_layer(
    layer: nn.Module,
    precision,
    gpu_spec,
) -> tuple[nn.Module, bool]:
    """
    Quantize a layer to the requested precision.

    Returns:
        (quantized_layer, fallback_occurred)

    fallback_occurred is True when the requested precision could not actually
    be realized and the original layer is returned instead.

    Current behavior:
      - FP16: returns the layer unchanged, fallback=False
      - FP8 requested on unsupported GPU: fallback=True
      - torchao unavailable: fallback=True for non-FP16 requests
      - quantization failure: fallback=True
    """
    from .types import Precision

    if precision == Precision.FP16:
        return layer, False

    if precision == Precision.W8A8_FP8 and not gpu_spec.supports_fp8:
        return layer, True

    try:
        from torchao.quantization import (
            float8_weight_only,
            int4_weight_only,
            int8_dynamic_activation_int8_weight,
            quantize_,
        )
    except ImportError:
        return layer, True

    try:
        q = copy.deepcopy(layer)

        if precision == Precision.W8A8_FP8:
            quantize_(q, float8_weight_only())
        elif precision == Precision.W8A8_INT8:
            quantize_(q, int8_dynamic_activation_int8_weight())
        elif precision == Precision.W4A16:
            quantize_(q, int4_weight_only(group_size=128))
        else:
            return layer, True

        return q, False

    except Exception:
        return layer, True


def _resolve_layers(model: nn.Module, layer_names: List[str]) -> Dict[str, nn.Module]:
    """
    Extract named modules from model.
    """
    modules: dict[str, nn.Module] = {}
    for name in layer_names:
        try:
            modules[name] = model.get_submodule(name)
        except AttributeError:
            raise KeyError(
                f"Model has no layer {name}\n"
                f"-> Available: {[n for n, _ in model.named_children()]}"
            )
    return modules


def _get_layer_shape(module: nn.Module) -> Tuple[int, ...]:
    if isinstance(module, nn.Linear):
        return (module.out_features, module.in_features)
    return tuple(next(module.parameters()).shape)


def _get_layer_dtype(layer: nn.Module) -> str:
    """
    Return dtype string from the first available parameter or buffer.
    """
    tensors = list(layer.parameters()) + list(layer.buffers())
    if not tensors:
        return "unknown"
    return str(tensors[0].dtype)
