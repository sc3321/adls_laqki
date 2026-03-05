from __future__ import annotations

import copy
import logging
import subprocess
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from qpe.profiler.gpu_specs import GPUSpec
from qpe.utils.types import Precision
from .cache import ProfileCache
from .models import LayerMeta, LayerProfile, ModelProfileResult

log = logging.Logger()

from qpe.utils.model_utils import (
    get_layer_names,
    _get_layer_dtype,
    _get_layer_shape,
    _quantize_layer,
)
from benchmark.util import _make_benchmark_input
from src.benchmark.measurements import (
    _time_layer,
    _is_memory_bound,
    _get_gpu_mem_usage,
    _get_ops_per_byte,
    _get_weight_memory_bytes,
    _measure_peak_memory,
)


def _get_torchao_version() -> str | None:
    try:
        import torchao  # type: ignore

        return getattr(torchao, "__version__", "unknown")
    except Exception:
        return None


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
    ) -> ModelProfileResult:
        """
        Profile all named layers across every supported precision.

        Returns a ModelProfileResult with typed LayerProfile and LayerMeta
        entries for all profiled layers.

        Only nn.Linear layers are profiled; others are skipped.
        Results are cached to disk and reloaded on subsequent calls.
        """
        layer_metas: Dict[str, LayerMeta] = {}

        # Cache-hit path: return cached profiles and recompute metas from live model.
        if self.profile_cache.is_model_complete(model_id):
            cached_entries = self._load_cached_profiles(
                model=model,
                layer_names=layer_names,
                target_batch_size=target_batch_size,
                model_id=model_id,
            )
            if cached_entries is not None:
                named = dict(model.named_modules())
                for name in layer_names:
                    layer = named.get(name)
                    if layer is None or not isinstance(layer, nn.Linear):
                        continue
                    layer_metas[name] = LayerMeta(
                        layer_type=type(layer).__name__,
                        layer_shape=list(_get_layer_shape(layer)),
                        dtype=_get_layer_dtype(layer),
                        param_count=int(sum(p.numel() for p in layer.parameters())),
                    )
                return ModelProfileResult(
                    qpe_version=self.qpe_version,
                    torch_version=torch.__version__,
                    torchao_version=_get_torchao_version(),
                    gpu_name=self.gpu_spec.name,
                    model_id=model_id,
                    batch_size=target_batch_size,
                    seq_len=self.seq_len,
                    entries=cached_entries,
                    layer_metas=layer_metas,
                )

        self._lock_gpu_clocks()
        entries: Dict[str, Dict[str, LayerProfile]] = {}
        named = dict(model.named_modules())

        try:
            for idx, name in enumerate(layer_names):
                if name not in named:
                    log.warning("Layer '%s' not in model - skipped.", name)
                    continue

                layer = named[name]
                if not isinstance(layer, nn.Linear):
                    log.debug(
                        "Skipping non-Linear layer '%s' (%s).",
                        name,
                        type(layer).__name__,
                    )
                    continue

                layer_metas[name] = LayerMeta(
                    layer_type=type(layer).__name__,
                    layer_shape=list(_get_layer_shape(layer)),
                    dtype=_get_layer_dtype(layer),
                    param_count=int(sum(p.numel() for p in layer.parameters())),
                )

                log.info("[%d/%d] Profiling %s ...", idx + 1, len(layer_names), name)
                entries[name] = self._profile_single_layer(
                    model_id=model_id,
                    layer_name=name,
                    layer=layer,
                    batch_size=target_batch_size,
                )
        finally:
            self._unlock_gpu_clocks()

        return ModelProfileResult(
            qpe_version=self.qpe_version,
            torch_version=torch.__version__,
            torchao_version=_get_torchao_version(),
            gpu_name=self.gpu_spec.name,
            model_id=model_id,
            batch_size=target_batch_size,
            seq_len=self.seq_len,
            entries=entries,
            layer_metas=layer_metas,
        )

    def _profile_single_layer(
        self,
        model_id: str,
        layer_name: str,
        layer: nn.Module,
        batch_size: int,
    ) -> Dict[str, LayerProfile]:
        # Hold precision to layer profile at that quantization
        result: Dict[str, LayerProfile] = {}

        inputs = _make_benchmark_input(batch_size, self.seq_len, layer, self._device)

        meta = LayerMeta(
            layer_type=type(layer).__name__,
            layer_shape=list(_get_layer_shape(layer)),
            dtype=_get_layer_dtype(layer),
            param_count=int(sum(p.numel() for p in layer.parameters())),
        )

        for prec in self._available_precisions:
            pv = prec.value
            if cached_profile := self.profile_cache.get(
                model_id, layer_name, batch_size, pv
            ):
                result[pv] = LayerProfile.from_dict(cached_profile)
                continue

            q_layer = _quantize_layer(layer.half(), prec, self.gpu_spec)

            lat = _time_layer(
                layer=q_layer,
                inputs=inputs,
                num_warmup=self.num_warmup,
                num_measurements=self.num_measurements,
                device=self._device,
            )
            lp = LayerProfile(
                latency_us=lat,
                memory_bytes=_get_weight_memory_bytes(layer),
                peak_memory_bytes=_measure_peak_memory(q_layer, inputs, self._device),
                is_memory_bound=_is_memory_bound(
                    module=layer,
                    input_tensor=inputs,
                    precision=prec,
                    gpu_spec=self.gpu_spec,
                    batch_size=batch_size,
                    sequence_length=self.seq_len,
                ),
                kernel_name=self._kernel_name(prec),
                p50_us=lat,
                p99_us=lat,
            )
            result[pv] = lp
            self.profile_cache.put(
                model_id=model_id,
                layer_name=layer_name,
                batch_size=batch_size,
                precision=pv,
                data=lp.to_dict(),
                layer_meta=meta.to_dict(),
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
    ) -> Dict[str, Dict[str, LayerProfile]] | None:
        """Load cached profiles for all requested layers and precisions."""
        named = dict(model.named_modules())
        cached: Dict[str, Dict[str, LayerProfile]] = {}
        for name in layer_names:
            if name not in named:
                continue

            layer = named[name]
            if not isinstance(layer, nn.Linear):
                continue

            layer_profiles: Dict[str, LayerProfile] = {}
            for precision in self._available_precisions:
                profile = self.profile_cache.get(
                    model_id=model_id,
                    layer_name=name,
                    batch_size=target_batch_size,
                    precision=precision.value,
                )
                if profile is None:
                    return None
                layer_profiles[precision.value] = LayerProfile.from_dict(profile)
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
            log.debug(
                "Could not lock GPU clocks (nvidia-smi unavailable or no permission)."
            )

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
