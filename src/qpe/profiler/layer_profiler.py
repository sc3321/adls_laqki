from __future__ import annotations

import logging
import subprocess
from typing import Dict, List

import torch
import torch.nn as nn

from src.qpe.profiler.gpu_specs import GPUSpec
from src.qpe.utils.types import Precision

from .cache import ProfileCache
from .models import LayerMeta, LayerProfile, ModelProfileResult

from src.qpe.utils.model_utils import (
    _get_layer_dtype,
    _get_layer_shape,
    _quantize_layer,
)
from src.benchmark.util import _make_benchmark_input
from src.benchmark.measurements import (
    _time_layer,
    _is_memory_bound,
    _get_weight_memory_bytes,
    _measure_peak_memory,
)

log = logging.getLogger(__name__)


def _get_torchao_version() -> str | None:
    try:
        import torchao  # type: ignore

        return getattr(torchao, "__version__", "unknown")
    except Exception:
        return None


class LayerProfiler:
    """
    Profiles individual model layers at multiple precisions.

    Profiling point identity:
      (model_id, layer_name, batch_size, seq_len, regime, precision)

    Notes:
      - Only nn.Linear layers are profiled.
      - Cache is keyed by batch_size + seq_len + regime, so decode/prefill
        do not collide.
      - latency_us is taken from p50_us.
    """

    def __init__(
        self,
        gpu_spec: GPUSpec,
        batch_sizes: List[int] | None = None,
        num_warmup: int = 50,
        num_measurements: int = 200,
        seq_len: int = 1,
        cache_dir: str = ".qpe_cache/profiles",
        qpe_version: str = "2.1",
    ) -> None:
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
        regime: str = "decode",
    ) -> ModelProfileResult:
        """
        Profile all requested linear layers across every supported precision.

        Returns a ModelProfileResult with typed LayerProfile and LayerMeta entries.

        Cache lookup is specific to:
          model_id, layer_name, batch_size, seq_len, regime, precision
        """
        layer_metas: Dict[str, LayerMeta] = {}
        named = dict(model.named_modules())

        cached_entries = self._load_cached_profiles(
            model=model,
            layer_names=layer_names,
            target_batch_size=target_batch_size,
            model_id=model_id,
            regime=regime,
        )
        if cached_entries is not None:
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
                schema_version=2,
                qpe_version=self.qpe_version,
                torch_version=torch.__version__,
                torchao_version=_get_torchao_version(),
                gpu_name=self.gpu_spec.name,
                model_id=model_id,
                batch_size=target_batch_size,
                seq_len=self.seq_len,
                regime=regime,
                entries=cached_entries,
                layer_metas=layer_metas,
            )

        self._lock_gpu_clocks()
        entries: Dict[str, Dict[str, LayerProfile]] = {}

        try:
            for idx, name in enumerate(layer_names):
                layer = named.get(name)
                if layer is None:
                    log.warning("Layer '%s' not found in model; skipped.", name)
                    continue

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

                log.info(
                    "[%d/%d] Profiling layer=%s batch=%d seq=%d regime=%s",
                    idx + 1,
                    len(layer_names),
                    name,
                    target_batch_size,
                    self.seq_len,
                    regime,
                )

                entries[name] = self._profile_single_layer(
                    model_id=model_id,
                    layer_name=name,
                    layer=layer,
                    batch_size=target_batch_size,
                    regime=regime,
                )
        finally:
            self._unlock_gpu_clocks()

        return ModelProfileResult(
            schema_version=2,
            qpe_version=self.qpe_version,
            torch_version=torch.__version__,
            torchao_version=_get_torchao_version(),
            gpu_name=self.gpu_spec.name,
            model_id=model_id,
            batch_size=target_batch_size,
            seq_len=self.seq_len,
            regime=regime,
            entries=entries,
            layer_metas=layer_metas,
        )

    def _profile_single_layer(
        self,
        model_id: str,
        layer_name: str,
        layer: nn.Module,
        batch_size: int,
        regime: str,
    ) -> Dict[str, LayerProfile]:
        """
        Profile one layer across all available precisions for the given
        (batch_size, seq_len, regime) point.
        """
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

            cached_profile = self.profile_cache.get(
                model_id=model_id,
                layer_name=layer_name,
                batch_size=batch_size,
                seq_len=self.seq_len,
                regime=regime,
                precision=pv,
            )
            if cached_profile is not None:
                result[pv] = LayerProfile.from_dict(cached_profile)
                continue

            q_layer, fallback_occurred = self._quantize_with_fallback(
                layer=layer,
                precision=prec,
            )

            timing = _time_layer(
                layer=q_layer,
                inputs=inputs,
                num_warmup=self.num_warmup,
                num_measurements=self.num_measurements,
                device=self._device,
            )

            weight_bytes = _get_weight_memory_bytes(q_layer)
            peak_memory_bytes = _measure_peak_memory(q_layer, inputs, self._device)
            is_memory_bound = _is_memory_bound(
                module=q_layer,
                input_tensor=inputs,
                weight_bytes=weight_bytes,
                precision=prec,
                gpu_spec=self.gpu_spec,
                batch_size=batch_size,
                sequence_length=self.seq_len,
            )

            lp = LayerProfile(
                precision=pv,
                batch_size=batch_size,
                seq_len=self.seq_len,
                regime=regime,
                latency_us=float(timing["latency_us"]),
                p50_us=float(timing["p50_us"]),
                p99_us=float(timing["p99_us"]),
                weight_bytes=weight_bytes,
                peak_memory_bytes=peak_memory_bytes,
                is_memory_bound=is_memory_bound,
                kernel_name=self._kernel_name(prec),
                fallback_occurred=fallback_occurred,
            )

            if bool(timing.get("unstable", False)):
                log.warning(
                    "Unstable timing detected for layer=%s precision=%s batch=%d seq=%d regime=%s "
                    "(mean=%.2f us, std=%.2f us, p50=%.2f us, p99=%.2f us)",
                    layer_name,
                    pv,
                    batch_size,
                    self.seq_len,
                    regime,
                    float(timing.get("mean_us", 0.0)),
                    float(timing.get("std_us", 0.0)),
                    float(timing["p50_us"]),
                    float(timing["p99_us"]),
                )

            result[pv] = lp

            self.profile_cache.put(
                model_id=model_id,
                layer_name=layer_name,
                batch_size=batch_size,
                seq_len=self.seq_len,
                regime=regime,
                precision=pv,
                data=lp.to_dict(),
                layer_meta=meta.to_dict(),
            )

            del q_layer
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        return result

    def _quantize_with_fallback(
        self,
        layer: nn.Module,
        precision: Precision,
    ) -> tuple[nn.Module, bool]:
        """
        Compatibility shim for quantization helper.

        Supports both helper contracts:
          old: _quantize_layer(...) -> nn.Module
          new: _quantize_layer(...) -> tuple[nn.Module, bool]

        Returns:
          (quantized_layer_or_fallback_layer, fallback_occurred)
        """
        requested_fp16 = precision == Precision.FP16

        quantized = _quantize_layer(layer.half(), precision, self.gpu_spec)

        if isinstance(quantized, tuple) and len(quantized) == 2:
            q_layer, fallback_occurred = quantized
            return q_layer, bool(fallback_occurred)

        q_layer = quantized

        if requested_fp16:
            return q_layer, False

        # Old helper cannot report fallback explicitly.
        # Conservative behavior: mark fallback as False here and let the
        # dedicated helper rewrite tighten this later.
        return q_layer, False

    def _kernel_name(self, precision: Precision) -> str:
        kernels = self.gpu_spec.available_kernels.get(precision.value, [])
        return kernels[0] if kernels else "generic"

    def _load_cached_profiles(
        self,
        model: nn.Module,
        layer_names: List[str],
        target_batch_size: int,
        model_id: str,
        regime: str,
    ) -> Dict[str, Dict[str, LayerProfile]] | None:
        """Load cached profiles for all requested layers and precisions."""
        named = dict(model.named_modules())
        cached: Dict[str, Dict[str, LayerProfile]] = {}

        for name in layer_names:
            layer = named.get(name)
            if layer is None or not isinstance(layer, nn.Linear):
                continue

            layer_profiles: Dict[str, LayerProfile] = {}
            for precision in self._available_precisions:
                profile = self.profile_cache.get(
                    model_id=model_id,
                    layer_name=name,
                    batch_size=target_batch_size,
                    seq_len=self.seq_len,
                    regime=regime,
                    precision=precision.value,
                )
                if profile is None:
                    return None
                layer_profiles[precision.value] = LayerProfile.from_dict(profile)

            cached[name] = layer_profiles

        if cached:
            log.info(
                "Profiler cache hit: model=%s gpu=%s batch=%s seq=%s regime=%s",
                model_id,
                self.gpu_spec.name,
                target_batch_size,
                self.seq_len,
                regime,
            )
        return cached if cached else None

    def _lock_gpu_clocks(self) -> None:
        """Lock GPU clocks to base frequency for more stable latency measurements."""
        if self._device.type != "cuda":
            return

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.applications.graphics",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            clock = result.stdout.strip().split("\n")[0].strip()

            if clock.isdigit():
                subprocess.run(
                    ["nvidia-smi", "-lgc", f"{clock},{clock}"],
                    capture_output=True,
                    timeout=5,
                )
                log.debug("GPU clocks locked to %s MHz.", clock)
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            log.debug(
                "Could not lock GPU clocks (nvidia-smi unavailable or insufficient permission)."
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
            supported.extend([Precision.W8A8_INT8, Precision.W4A16])

        if self.gpu_spec.supports_fp8:
            supported.append(Precision.W8A8_FP8)

        return supported
