from __future__ import annotations

import copy
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from qpe.profiler.gpu_specs import GPUSpec
from qpe.solver.types import Precision

log = logging.getLogger(__name__)

# Bytes per weight parameter for each precision.
_BYTES_PER_PARAM: Dict[str, float] = {
    Precision.FP16.value:      2.0,
    Precision.W8A8_FP8.value:  1.0,
    Precision.W8A8_INT8.value: 1.0,
    Precision.W4A16.value:     0.5,
}

# Roofline threshold (FLOP/byte): layers below this are memory-bandwidth limited.
_MEMORY_BOUND_THRESHOLD = 100.0


# ── Pure helper functions (importable for unit tests) ─────────────────────────

def _count_params(layer: nn.Module) -> int:
    return sum(p.numel() for p in layer.parameters())


def _weight_bytes(layer: nn.Module, precision: str) -> int:
    return int(_count_params(layer) * _BYTES_PER_PARAM[precision])


def _is_memory_bound(
    layer: nn.Module, precision: str, batch_size: int, seq_len: int
) -> bool:
    """
    Roofline heuristic: arithmetic intensity (FLOP/byte) < threshold → memory bound.
    Non-Linear layers always return False (no meaningful roofline for activations etc.).
    """
    if not isinstance(layer, nn.Linear):
        return False
    flops = 2.0 * batch_size * seq_len * layer.in_features * layer.out_features
    weight_b = _weight_bytes(layer, precision)
    act_b    = batch_size * seq_len * layer.in_features  * 2  # fp16 activations in
    out_b    = batch_size * seq_len * layer.out_features * 2  # fp16 activations out
    intensity = flops / max(weight_b + act_b + out_b, 1)
    return intensity < _MEMORY_BOUND_THRESHOLD


def _quantize_layer(layer: nn.Module, precision: Precision, gpu_spec: GPUSpec) -> nn.Module:
    """
    Return a quantized copy of *layer* suitable for latency benchmarking.
    FP16 → original layer (no copy needed).
    Other precisions → deep copy quantized via torchao when available,
    otherwise returns the FP16 layer as an approximation (latency is FP16-equivalent
    but memory numbers from _weight_bytes remain correct).
    """
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
        log.warning(
            "torchao not installed; latency for %s will be measured as FP16.", precision.value
        )
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
        log.warning(
            "quantize_ failed for %s (%s); using FP16 layer for latency.", precision.value, e
        )
        return layer


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
    layer  = layer.to(device)
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
            end   = torch.cuda.Event(enable_timing=True)
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


def _measure_peak_memory(
    layer: nn.Module, inputs: torch.Tensor, device: torch.device
) -> int:
    """
    Returns peak memory in bytes during a single forward pass.
    On CPU, returns a static estimate (weights + activations).
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return sum(p.nbytes for p in layer.parameters()) + inputs.nbytes * 2

    layer  = layer.to(device)
    inputs = inputs.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    with torch.no_grad():
        layer(inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)


# ── Main class ────────────────────────────────────────────────────────────────

class LayerProfiler:
    """
    Profiles individual model layers at multiple precisions.

    Profiling protocol (designed for measurement stability):
    1. Lock GPU clocks to base frequency (prevents thermal throttling).
    2. Allocate each layer in isolation on GPU.
    3. Generate representative inputs of shape (batch_size, seq_len, in_features).
    4. Warmup: num_warmup iterations — fills kernel caches, reaches thermal steady-state.
    5. Measure: num_measurements iterations with torch.cuda.Event timing (GPU-side only).
    6. Record median latency (robust to outlier CUDA preemption events).
    7. Repeat for every supported precision × target_batch_size combination.
    8. Cache results to disk — subsequent runs for the same model/GPU are instant.
    """

    def __init__(
        self,
        gpu_spec: GPUSpec,
        batch_sizes: List[int] = None,
        num_warmup: int = 50,
        num_measurements: int = 200,
        seq_len: int = 1,
        cache_dir: str = ".qpe_cache/profiles",
    ):
        self.gpu_spec        = gpu_spec
        self.batch_sizes     = batch_sizes or [1, 4, 16, 64]
        self.num_warmup      = num_warmup
        self.num_measurements = num_measurements
        self.seq_len         = seq_len
        self.cache_dir       = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._available_precisions = self._filter_precisions()

    # ── Precision filtering ────────────────────────────────────────────────

    def _filter_precisions(self) -> List[Precision]:
        """Return precisions supported by the target GPU spec."""
        supported = [Precision.FP16]
        if self.gpu_spec.supports_int8_tensor_core:
            supported += [Precision.W8A8_INT8, Precision.W4A16]
        if self.gpu_spec.supports_fp8:
            supported.append(Precision.W8A8_FP8)
        return supported

    # ── Cache ──────────────────────────────────────────────────────────────

    def _cache_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "_").replace(" ", "_")
        gpu  = self.gpu_spec.name.replace(" ", "_")
        return self.cache_dir / f"{safe}__{gpu}.json"

    def _load_cache(self, model_id: str) -> dict | None:
        path = self._cache_path(model_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            log.info("Profiler cache hit: %s", path)
            return data
        except (json.JSONDecodeError, KeyError):
            log.warning("Corrupt profiler cache at %s; re-profiling.", path)
            return None

    def _save_cache(self, model_id: str, results: dict) -> None:
        path = self._cache_path(model_id)
        path.write_text(json.dumps(results))
        log.info("Profiler cache saved: %s", path)

    # ── Public API ─────────────────────────────────────────────────────────

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
            { layer_name: {
                "memory_bytes":      {precision: int},
                "latency_us":        {precision: float},
                "peak_memory_bytes": {precision: int},
                "kernel_name":       {precision: str},
                "is_memory_bound":   {precision: bool},
            }}

        Only nn.Linear layers are profiled; others are skipped.
        Results are cached to disk and reloaded on subsequent calls.
        """
        cached = self._load_cache(model_id)
        if cached is not None:
            return cached

        self._lock_gpu_clocks()
        results: Dict[str, Dict] = {}
        named = dict(model.named_modules())

        try:
            for idx, name in enumerate(layer_names):
                if name not in named:
                    log.warning("Layer '%s' not in model — skipped.", name)
                    continue
                layer = named[name]
                if not isinstance(layer, nn.Linear):
                    log.debug("Skipping non-Linear layer '%s' (%s).", name, type(layer).__name__)
                    continue
                log.info("[%d/%d] Profiling %s ...", idx + 1, len(layer_names), name)
                results[name] = self._profile_single_layer(layer, target_batch_size)
        finally:
            self._unlock_gpu_clocks()

        self._save_cache(model_id, results)
        return results

    # ── Internal profiling ─────────────────────────────────────────────────

    def _profile_single_layer(self, layer: nn.Linear, batch_size: int) -> dict:
        result: dict = {
            "memory_bytes":      {},
            "latency_us":        {},
            "peak_memory_bytes": {},
            "kernel_name":       {},
            "is_memory_bound":   {},
        }
        inputs = torch.randn(batch_size, self.seq_len, layer.in_features, dtype=torch.float16)

        for prec in self._available_precisions:
            pv = prec.value
            q_layer = _quantize_layer(layer.half(), prec, self.gpu_spec)

            result["memory_bytes"][pv]      = _weight_bytes(layer, pv)
            result["latency_us"][pv]        = _time_layer(
                q_layer, inputs, self.num_warmup, self.num_measurements, self._device
            )
            result["peak_memory_bytes"][pv] = _measure_peak_memory(q_layer, inputs, self._device)
            result["kernel_name"][pv]       = self._kernel_name(prec)
            result["is_memory_bound"][pv]   = _is_memory_bound(layer, pv, batch_size, self.seq_len)

            del q_layer
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        return result

    def _kernel_name(self, precision: Precision) -> str:
        kernels = self.gpu_spec.available_kernels.get(precision.value, [])
        return kernels[0] if kernels else "generic"

    # ── GPU clock management ───────────────────────────────────────────────

    def _lock_gpu_clocks(self) -> None:
        """Lock GPU clocks to base frequency for stable latency measurements."""
        if self._device.type != "cuda":
            return
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.applications.graphics",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            clock = r.stdout.strip().split("\n")[0].strip()
            if clock.isdigit():
                subprocess.run(
                    ["nvidia-smi", "-lgc", f"{clock},{clock}"],
                    capture_output=True, timeout=5,
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
