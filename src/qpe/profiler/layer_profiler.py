from __future__ import annotations

import copy
import logging
import subprocess
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from qpe.profiler.gpu_specs import GPUSpec
from qpe.solver.types import Precision
from .cache import ProfileCache

log = logging.getLogger(__name__)

# Bytes per weight parameter for each precision.
_BYTES_PER_PARAM: Dict[str, float] = {
    Precision.FP16.value: 2.0,
    Precision.W8A8_FP8.value: 1.0,
    Precision.W8A8_INT8.value: 1.0,
    Precision.W4A16.value: 0.5,
}

# Roofline threshold (FLOP/byte): layers below this are memory-bandwidth limited.
_MEMORY_BOUND_THRESHOLD = 100.0


def _get_weight_memory_bytes(module: nn.Module) -> int:
    """
    Get total size in bytes of weights
    """
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

    weight_bytes = _get_weight_memory_bytes(module)
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


def _quantize_layer(layer: nn.Module, precision: Precision, gpu_spec: GPUSpec) -> nn.Module:
    if precision == Precision.FP16:
        return layer

    if precision == Precision.W8A8_FP8 and not gpu_spec.supports_fp8:
        log.debug("GPU does not support FP8; using FP16 for latency of W8A8_FP8.")
        return layer

    try:
        from torchao.quantization import (
            float8_weight_only,
            int4_weight_only,
            int8_dynamic_activation_int8_weight,
            quantize_,
        )
    except ImportError:
        log.warning("torchao not installed; latency for %s will be measured as FP16.", precision.value)
        return layer

    try:
        q = copy.deepcopy(layer)
        if precision == Precision.W8A8_FP8:
            quantize_(q, float8_weight_only())
        elif precision == Precision.W8A8_INT8:
            quantize_(q, int8_dynamic_activation_int8_weight())
        elif precision == Precision.W4A16:
            quantize_(q, int4_weight_only(group_size=128))
        return q
    except Exception as e:
        log.warning("quantize_ failed for %s (%s); using FP16 layer for latency.", precision.value, e)
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

def _make_benchmark_input(
    batch_size: int,
    sequence_length: int,
    module: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Benchmark input for a layer.

    We keep inputs floating-point. For CUDA profiling we use fp16 inputs
    to match layer.half() profiling and avoid dtype mismatch.
    """
    if isinstance(module, nn.Linear):
        in_features = module.in_features
    else:
        in_features = next(module.parameters()).shape[-1]

    # Always use fp16 on CUDA for profiling (matches layer.half()).
    if device.type == "cuda":
        inp_dtype = torch.float16
    else:
        # CPU fallback: match module dtype if possible
        try:
            inp_dtype = next(module.parameters()).dtype
        except StopIteration:
            inp_dtype = torch.float32

    return torch.randn(
        batch_size,
        sequence_length,
        in_features,
        dtype=inp_dtype,
        device=device,
    )

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


class LayerProfiler:
    """
    Profiles individual model layers at multiple precisions.

    Profiling protocol (designed for measurement stability):
    1. Lock GPU clocks to base frequency (prevents thermal throttling).
    2. Allocate each layer in isolation on GPU.
    3. Generate representative inputs of shape (batch_size, seq_len, in_features).
    4. Warmup: num_warmup iterations - fills kernel caches, reaches thermal steady-state.
    5. Measure: num_measurements iterations with torch.cuda.Event timing (GPU-side only).
    6. Record median latency (robust to outlier CUDA preemption events).
    7. Repeat for every supported precision x target_batch_size combination.
    8. Cache results to disk - subsequent runs for the same model/GPU are instant.
    """

    def __init__(
        self,
        gpu_spec: GPUSpec,
        batch_sizes: List[int] = None,
        num_warmup: int = 50,
        num_measurements: int = 200,
        seq_len: int = 1,
        cache_dir: str = ".qpe_cache/profiles",
        qpe_version: str = "2.0",
    ):
        self.gpu_spec = gpu_spec
        self.batch_sizes = batch_sizes or [1, 4, 16, 64]
        self.num_warmup = num_warmup
        self.num_measurements = num_measurements
        self.seq_len = seq_len
        self.qpe_version = qpe_version
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._available_precisions = self._filter_precisions()
        self.profile_cache = ProfileCache(
            root_dir=cache_dir,
            qpe_version=self.qpe_version,
            gpu_name=self.gpu_spec.name,
            supported_precisions=[p.value for p in self._available_precisions],
        )

    def profile_all_layers(
        self,
        model: nn.Module,
        layer_names: List[str],
        target_batch_size: int = 1,
        model_id: str = "unknown",
    ) -> Dict[str, Dict]:
        """
        Profile all named layers across every supported precision.

        Returns:
            {
                layer_name: {
                    <precision>: {
                        "memory_bytes":       int,
                        "latency_us":         float,
                        "peak_memory_bytes":  int,
                        "kernel_name":        str,
                        "is_memory_bound":    bool,
                    },
                    ...
                },
                "__layer_metas__": {
                    layer_name: {
                        "layer_type":  str,
                        "shape":       list[int],
                        "dtype":       str,
                        "param_count": int,
                    },
                    ...
                }
            }

        Only nn.Linear layers are profiled; others are skipped.
        Results are cached to disk and reloaded on subsequent calls.
        """
        layer_metas: Dict[str, Dict] = {}

        # Cache-hit path: return cached profiles, but still attach metas from live model.
        if self.profile_cache.is_model_complete(model_id):
            cached = self._load_cached_profiles(
                model=model,
                layer_names=layer_names,
                target_batch_size=target_batch_size,
                model_id=model_id,
            )
            if cached is not None:
                named = dict(model.named_modules())
                for name in layer_names:
                    layer = named.get(name)
                    if layer is None or not isinstance(layer, nn.Linear):
                        continue
                    layer_metas[name] = {
                        "layer_type": type(layer).__name__,
                        "shape": list(_get_layer_shape(layer)),
                        "dtype": _get_layer_dtype(layer),
                        "param_count": int(sum(p.numel() for p in layer.parameters())),
                    }
                cached["__layer_metas__"] = layer_metas
                return cached

        self._lock_gpu_clocks()
        results: Dict[str, Dict] = {}
        named = dict(model.named_modules())

        try:
            for idx, name in enumerate(layer_names):
                if name not in named:
                    log.warning("Layer '%s' not in model - skipped.", name)
                    continue

                layer = named[name]
                if not isinstance(layer, nn.Linear):
                    log.debug("Skipping non-Linear layer '%s' (%s).", name, type(layer).__name__)
                    continue

                layer_metas[name] = {
                    "layer_type": type(layer).__name__,
                    "shape": list(_get_layer_shape(layer)),
                    "dtype": _get_layer_dtype(layer),
                    "param_count": int(sum(p.numel() for p in layer.parameters())),
                }

                log.info("[%d/%d] Profiling %s ...", idx + 1, len(layer_names), name)
                results[name] = self._profile_single_layer(
                    model_id=model_id,
                    layer_name=name,
                    layer=layer,
                    batch_size=target_batch_size,
                )
        finally:
            self._unlock_gpu_clocks()

        results["__layer_metas__"] = layer_metas
        return results

    def _profile_single_layer(
        self,
        model_id: str,
        layer_name: str,
        layer: nn.Module,
        batch_size: int,
    ) -> Dict:
        # Hold precision to layer profile at that quantization
        result: Dict[str, Dict] = {}

        inputs = _make_benchmark_input(batch_size, self.seq_len, layer, self._device)

        layer_meta = {
            "shape": list(_get_layer_shape(layer)),
            "dtype": _get_layer_dtype(layer),
            "param_count": int(sum(p.numel() for p in layer.parameters())),
        }

        for prec in self._available_precisions:
            pv = prec.value
            if layer_profile := self.profile_cache.get(model_id, layer_name, batch_size, pv):
                result[pv] = layer_profile
                continue

            q_layer = _quantize_layer(layer.half(), prec, self.gpu_spec)

            layer_profile = {
                "latency_us": _time_layer(
                    layer=q_layer,
                    inputs=inputs,
                    num_warmup=self.num_warmup,
                    num_measurements=self.num_measurements,
                    device=self._device,
                ),
                "memory_bytes": _get_weight_memory_bytes(layer),
                "peak_memory_bytes": _measure_peak_memory(q_layer, inputs, self._device),
                "is_memory_bound": _is_memory_bound(
                    module=layer,
                    input_tensor=inputs,
                    precision=prec,
                    gpu_spec=self.gpu_spec,
                    batch_size=batch_size,
                    sequence_length=self.seq_len,
                ),
                "kernel_name": self._kernel_name(prec),
            }
            result[pv] = layer_profile
            self.profile_cache.put(
                model_id=model_id,
                layer_name=layer_name,
                batch_size=batch_size,
                precision=pv,
                data=layer_profile,
                layer_meta=layer_meta,
            )

            del q_layer
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        return result

    def _kernel_name(self, precision: Precision) -> str:
        kernels = self.gpu_spec.available_kernels.get(precision.value, [])
        return kernels[0] if kernels else "generic"

    def _load_cached_profiles(
        self,
        model: nn.Module,
        layer_names: List[str],
        target_batch_size: int,
        model_id: str,
    ) -> Dict[str, Dict] | None:
        """Load cached profiles for all requested layers and precisions."""
        named = dict(model.named_modules())
        cached: Dict[str, Dict] = {}
        for name in layer_names:
            if name not in named:
                continue

            layer = named[name]
            if not isinstance(layer, nn.Linear):
                continue

            layer_profiles: Dict[str, Dict] = {}
            for precision in self._available_precisions:
                profile = self.profile_cache.get(
                    model_id=model_id,
                    layer_name=name,
                    batch_size=target_batch_size,
                    precision=precision.value,
                )
                if profile is None:
                    return None
                layer_profiles[precision.value] = profile
            cached[name] = layer_profiles

        log.info(
            "Profiler cache hit: model=%s gpu=%s batch=%s",
            model_id,
            self.gpu_spec.name,
            target_batch_size,
        )
        return cached

    def _lock_gpu_clocks(self) -> None:
        """Lock GPU clocks to base frequency for stable latency measurements."""
        if self._device.type != "cuda":
            return
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.applications.graphics",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            clock = r.stdout.strip().split("\n")[0].strip()
            if clock.isdigit():
                subprocess.run(
                    ["nvidia-smi", "-lgc", f"{clock},{clock}"],
                    capture_output=True,
                    timeout=5,
                )
                log.debug("GPU clocks locked to %s MHz.", clock)
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            log.debug("Could not lock GPU clocks (nvidia-smi unavailable or no permission).")

    def _unlock_gpu_clocks(self) -> None:
        """Restore dynamic GPU clock management."""
        if self._device.type != "cuda":
            return
        try:
            subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, timeout=5)
            log.debug("GPU clocks restored.")
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            pass

    def _filter_precisions(self) -> List[Precision]:
        """Return precisions supported by the target GPU spec."""
        supported = [Precision.FP16]
        if self.gpu_spec.supports_int8_tensor_core:
            supported += [Precision.W8A8_INT8, Precision.W4A16]
        if self.gpu_spec.supports_fp8:
            supported.append(Precision.W8A8_FP8)
        return supported
