from __future__ import annotations
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from qpe.profiler.gpu_specs import GPUSpec
from qpe.utils.types import Precision

# Bytes per weight parameter for each precision.
_BYTES_PER_PARAM: Dict[str, float] = {
    Precision.FP16.value: 2.0,
    Precision.W8A8_FP8.value: 1.0,
    Precision.W8A8_INT8.value: 1.0,
    Precision.W4A16.value: 0.5,
}

# Roofline threshold (FLOP/byte): layers below this are memory-bandwidth limited.
_MEMORY_BOUND_THRESHOLD = 100.0


def _get_gpu_mem_usage() -> float : 
    import pynvml
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 3)
    except Exception:
        return 0.0



def _time_layer(
    layer: nn.Module,
    inputs: torch.Tensor,
    num_warmup: int,
    num_measurements: int,
    device: torch.device,
) -> float:
    """
    Returns median latency in microseconds.
    Uses torch.cuda.Event on GPU; falls back to time.perf_counter on CPU.
    """
    layer = layer.to(device)
    inputs = inputs.to(device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(num_warmup):
            layer(inputs)
        if use_cuda:
            torch.cuda.synchronize()

        timings: List[float] = []
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(num_measurements):
                start.record()
                layer(inputs)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end) * 1_000.0)  # ms → µs
        else:
            for _ in range(num_measurements):
                t0 = time.perf_counter()
                layer(inputs)
                timings.append((time.perf_counter() - t0) * 1e6)

    timings.sort()
    return timings[len(timings) // 2]


def _measure_peak_memory(layer: nn.Module, inputs: torch.Tensor, device: torch.device) -> int:
    """
    Returns peak memory in bytes during a single forward pass.
    On CPU, returns a static estimate (weights + activations).
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return sum(p.nbytes for p in layer.parameters()) + inputs.nbytes * 2

    layer = layer.to(device)
    inputs = inputs.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    with torch.no_grad():
        layer(inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)

def _get_weight_memory_bytes(module: nn.Module, precision: Precision | None = None) -> int:
    """
    Get total size in bytes of weights at the target precision.
    Uses _BYTES_PER_PARAM when precision is given; falls back to actual element_size().
    """
    bytes_per_param = _BYTES_PER_PARAM.get(precision.value, None) if precision is not None else None
    if bytes_per_param is not None:
        return int(
            sum(param.numel() for param in module.parameters()) * bytes_per_param
            + sum(buf.numel() * buf.element_size() for buf in module.buffers())
        )
    return (
        sum(param.numel() * param.element_size() for param in module.parameters())
        + sum(buf.numel() * buf.element_size() for buf in module.buffers())
    )


def _is_memory_bound(
    module: nn.Module,
    input_tensor: torch.Tensor,
    precision: Precision,
    gpu_spec: GPUSpec,
    batch_size: int,
    sequence_length: int,
) -> bool:
    """
    Roofline heuristic: arithmetic intensity (FLOP/byte) < threshold → memory bound.
    Non-Linear layers always return False (no meaningful roofline for activations etc.).
    """

    if isinstance(module, nn.Linear):
        input_size = batch_size * sequence_length
        n_feat_out = module.out_features
        n_feat_in = module.in_features
    else:
        # estimate from parameter count
        input_size = batch_size * sequence_length
        params = sum(p.numel() for p in module.parameters())
        n_feat_in = n_feat_out = int(params**0.5)

    flops = 2 * input_size * n_feat_in * n_feat_out

    weight_bytes = _get_weight_memory_bytes(module, precision)
    input_bytes = input_tensor.numel() * input_tensor.element_size()
    output_bytes = input_size * n_feat_out * 2

    total_bytes = input_bytes + weight_bytes + output_bytes
    intensity = flops / max(total_bytes, 1)

    return intensity < _get_ops_per_byte(precision, gpu_spec)


def _get_ops_per_byte(
    precision: Precision,
    spec: GPUSpec,
) -> float:
    # GPU ops:byte ratio at the precision's peak throughput
    bw_bytes_per_sec = spec.memory_bandwidth_tb_s * 1e12

    if precision in (Precision.W8A8_FP8,) and spec.peak_fp8_tflops:
        peak_flops = spec.peak_fp8_tflops * 1e12
    elif precision in (Precision.W8A8_INT8,):
        peak_flops = spec.peak_int8_tops * 1e12
    else:
        peak_flops = spec.peak_fp16_tflops * 1e12

    return peak_flops / bw_bytes_per_sec