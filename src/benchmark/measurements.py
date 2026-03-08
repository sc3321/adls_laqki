from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.qpe.profiler.gpu_specs import GPUSpec
from src.qpe.utils.types import Precision


def _get_gpu_mem_usage() -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024**3)
    except Exception:
        return 0.0


def _time_layer(
    layer: nn.Module,
    inputs: torch.Tensor,
    num_warmup: int,
    num_measurements: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    Measure layer forward latency.

    Returns a structured timing summary with:
      - latency_us: canonical solver-facing latency (equal to p50_us)
      - p50_us
      - p99_us
      - mean_us
      - std_us
      - unstable
      - total_measurement_time_s

    CUDA path:
      - warm up
      - record one start/end event pair per iteration
      - synchronize once after all measurements
      - compute elapsed times from recorded events

    CPU path:
      - use time.perf_counter for each iteration
    """
    layer = layer.to(device)
    inputs = inputs.to(device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()

    with torch.no_grad():
        for _ in range(num_warmup):
            layer(inputs)

        if use_cuda:
            torch.cuda.synchronize(device)

        wall_t0 = time.perf_counter()
        timings_us: list[float] = []

        if use_cuda:
            starts = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_measurements)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_measurements)
            ]

            for i in range(num_measurements):
                starts[i].record()
                layer(inputs)
                ends[i].record()

            torch.cuda.synchronize(device)

            timings_us = [
                starts[i].elapsed_time(ends[i]) * 1_000.0
                for i in range(num_measurements)
            ]
        else:
            for _ in range(num_measurements):
                t0 = time.perf_counter()
                layer(inputs)
                timings_us.append((time.perf_counter() - t0) * 1e6)

        total_measurement_time_s = time.perf_counter() - wall_t0

    arr = np.asarray(timings_us, dtype=np.float64)
    if arr.size == 0:
        return {
            "latency_us": 0.0,
            "p50_us": 0.0,
            "p99_us": 0.0,
            "mean_us": 0.0,
            "std_us": 0.0,
            "unstable": True,
            "total_measurement_time_s": total_measurement_time_s,
        }

    p50_us = float(np.percentile(arr, 50))
    p99_us = float(np.percentile(arr, 99))
    mean_us = float(arr.mean())
    std_us = float(arr.std())
    unstable = bool((std_us / mean_us) > 0.05) if mean_us > 0.0 else True

    return {
        "latency_us": p50_us,
        "p50_us": p50_us,
        "p99_us": p99_us,
        "mean_us": mean_us,
        "std_us": std_us,
        "unstable": unstable,
        "total_measurement_time_s": total_measurement_time_s,
    }


def _measure_peak_memory(
    layer: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> int:
    """
    Return peak memory in bytes during a single forward pass.

    On CPU, returns a simple estimate:
      weights/buffers + 2x input bytes
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return (
            sum(p.numel() * p.element_size() for p in layer.parameters())
            + sum(b.numel() * b.element_size() for b in layer.buffers())
            + inputs.numel() * inputs.element_size() * 2
        )

    layer = layer.to(device)
    inputs = inputs.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    with torch.no_grad():
        layer(inputs)

    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def _get_weight_memory_bytes(module: nn.Module) -> int:
    """
    Return total storage in bytes for parameters + buffers of the given module.

    Important:
      This should be called on the quantized module (q_layer), not the original layer,
      so that INT8 / INT4 / FP8 storage differences are reflected correctly.
    """
    return int(
        sum(param.numel() * param.element_size() for param in module.parameters())
        + sum(buf.numel() * buf.element_size() for buf in module.buffers())
    )


def _is_memory_bound(
    module: nn.Module,
    input_tensor: torch.Tensor,
    weight_bytes: int,
    precision: Precision,
    gpu_spec: GPUSpec,
    batch_size: int,
    sequence_length: int,
) -> bool:
    """
    Roofline-style heuristic:
      arithmetic_intensity = FLOPs / bytes_moved
      memory-bound iff arithmetic_intensity < GPU peak ops/byte at this precision

    Notes:
      - weight_bytes must be passed explicitly from the profiled module so the
        heuristic uses the quantized representation's true storage cost.
      - output bytes currently assume FP16-like output activations (2 bytes each).
        This is a first-pass heuristic, not an exact traffic model.
    """
    input_size = batch_size * sequence_length

    if isinstance(module, nn.Linear):
        n_feat_in = int(module.in_features)
        n_feat_out = int(module.out_features)
    else:
        params = sum(p.numel() for p in module.parameters())
        n_feat_in = int(params**0.5)
        n_feat_out = int(params**0.5)

    flops = 2.0 * input_size * n_feat_in * n_feat_out

    input_bytes = input_tensor.numel() * input_tensor.element_size()
    output_bytes = input_size * n_feat_out * 2  # heuristic: FP16-sized outputs

    total_bytes = float(input_bytes + weight_bytes + output_bytes)
    intensity = flops / max(total_bytes, 1.0)

    return bool(intensity < _get_ops_per_byte(precision, gpu_spec))


def _get_ops_per_byte(
    precision: Precision,
    spec: GPUSpec,
) -> float:
    """
    Return GPU peak arithmetic intensity threshold (ops/byte) for the given precision.
    """
    bw_bytes_per_sec = spec.memory_bandwidth_tb_s * 1e12

    if precision == Precision.W8A8_FP8 and getattr(spec, "peak_fp8_tflops", 0):
        peak_ops_per_sec = spec.peak_fp8_tflops * 1e12
    elif precision == Precision.W8A8_INT8:
        peak_ops_per_sec = spec.peak_int8_tops * 1e12
    else:
        peak_ops_per_sec = spec.peak_fp16_tflops * 1e12

    return peak_ops_per_sec / bw_bytes_per_sec
