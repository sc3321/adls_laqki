"""
Tests for qpe.profiler.layer_profiler.

All tests run on CPU — no GPU required. torchao is optional;
quantization falls back gracefully when unavailable.
"""
import json

import pytest
import torch
import torch.nn as nn

from qpe.profiler.gpu_specs import GPU_REGISTRY, GPUSpec
from qpe.profiler.layer_profiler import (
    LayerProfiler,
    _count_params,
    _is_memory_bound,
    _measure_peak_memory,
    _time_layer,
    _weight_bytes,
)
from qpe.solver.types import Precision

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_GPU = GPUSpec(
    name="NVIDIA Tesla T4",
    compute_capability=(7, 5),
    memory_gb=16.0,
    memory_bandwidth_tb_s=0.32,
    supports_fp8=False,
    supports_fp4=False,
    supports_int8_tensor_core=True,
    supports_int4_tensor_core=True,
    peak_fp16_tflops=65.0,
    peak_int8_tops=130.0,
    peak_int4_tops=260.0,
    available_kernels={
        "FP16":       ["cublas", "cutlass"],
        "W8A8_INT8":  ["cutlass", "cublas"],
        "W4A16":      ["marlin", "exllamav2", "autogptq"],
    },
)

CPU = torch.device("cpu")
GPU = torch.device("cuda")
requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def profiler(tmp_path):
    return LayerProfiler(
        gpu_spec=MOCK_GPU,
        batch_sizes=[1],
        num_warmup=2,
        num_measurements=5,
        seq_len=1,
        cache_dir=str(tmp_path),
    )


@pytest.fixture
def small_linear():
    return nn.Linear(64, 32, bias=False)


# ── _count_params ─────────────────────────────────────────────────────────────

class TestCountParams:
    def test_linear_no_bias(self):
        assert _count_params(nn.Linear(4, 8, bias=False)) == 32

    def test_linear_with_bias(self):
        assert _count_params(nn.Linear(4, 8, bias=True)) == 40  # 32 + 8

    def test_sequential(self):
        model = nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 2, bias=False))
        assert _count_params(model) == 32 + 16


# ── _weight_bytes ─────────────────────────────────────────────────────────────

class TestWeightBytes:
    def test_fp16_is_2_bytes_per_param(self):
        layer = nn.Linear(128, 128, bias=False)
        assert _weight_bytes(layer, Precision.FP16.value) == 128 * 128 * 2

    def test_int8_is_1_byte_per_param(self):
        layer = nn.Linear(128, 128, bias=False)
        assert _weight_bytes(layer, Precision.W8A8_INT8.value) == 128 * 128 * 1

    def test_int4_is_half_byte_per_param(self):
        layer = nn.Linear(128, 128, bias=False)
        assert _weight_bytes(layer, Precision.W4A16.value) == 128 * 128 // 2

    def test_fp16_larger_than_int4(self):
        layer = nn.Linear(256, 256, bias=False)
        assert _weight_bytes(layer, Precision.FP16.value) > _weight_bytes(layer, Precision.W4A16.value)


# ── _is_memory_bound ──────────────────────────────────────────────────────────

class TestIsMemoryBound:
    def test_small_batch_is_memory_bound(self):
        layer = nn.Linear(4096, 4096, bias=False)
        assert _is_memory_bound(layer, Precision.W4A16.value, batch_size=1, seq_len=1)

    def test_returns_bool(self):
        layer = nn.Linear(4096, 4096, bias=False)
        result = _is_memory_bound(layer, Precision.FP16.value, batch_size=256, seq_len=512)
        assert isinstance(result, bool)

    def test_non_linear_returns_false(self):
        assert not _is_memory_bound(nn.ReLU(), Precision.FP16.value, 1, 1)
        assert not _is_memory_bound(nn.LayerNorm(64), Precision.FP16.value, 1, 1)


# ── _time_layer ───────────────────────────────────────────────────────────────

class TestTimeLayer:
    @requires_gpu
    def test_returns_positive_microseconds(self):
        layer = nn.Linear(64, 64, bias=False).half()
        inputs = torch.randn(1, 1, 64, dtype=torch.float16)
        t = _time_layer(layer, inputs, num_warmup=1, num_measurements=3, device=GPU)
        assert t > 0.0

    @requires_gpu
    def test_larger_layer_slower(self):
        small = nn.Linear(32, 32, bias=False).half()
        large = nn.Linear(512, 512, bias=False).half()
        t_s = _time_layer(small, torch.randn(1, 1, 32,  dtype=torch.float16), 2, 10, GPU)
        t_l = _time_layer(large, torch.randn(1, 1, 512, dtype=torch.float16), 2, 10, GPU)
        assert t_l > t_s


# ── _measure_peak_memory ──────────────────────────────────────────────────────

class TestMeasurePeakMemory:
    @requires_gpu
    def test_returns_positive_bytes(self):
        layer = nn.Linear(64, 64, bias=False).half()
        inputs = torch.randn(1, 1, 64, dtype=torch.float16)
        mem = _measure_peak_memory(layer, inputs, GPU)
        assert mem > 0

    @requires_gpu
    def test_larger_layer_more_memory(self):
        small = nn.Linear(32,  32,  bias=False).half()
        large = nn.Linear(512, 512, bias=False).half()
        m_s = _measure_peak_memory(small, torch.randn(1, 1, 32,  dtype=torch.float16), GPU)
        m_l = _measure_peak_memory(large, torch.randn(1, 1, 512, dtype=torch.float16), GPU)
        assert m_l > m_s


# ── LayerProfiler._filter_precisions ─────────────────────────────────────────

class TestFilterPrecisions:
    def test_fp16_always_present(self, profiler):
        assert Precision.FP16 in profiler._available_precisions

    def test_fp8_excluded_when_unsupported(self, tmp_path):
        p = LayerProfiler(
            gpu_spec=MOCK_GPU.model_copy(update={"supports_fp8": False}),
            batch_sizes=[1], num_warmup=1, num_measurements=2,
            cache_dir=str(tmp_path),
        )
        assert Precision.W8A8_FP8 not in p._available_precisions

    def test_fp8_included_when_supported(self, tmp_path):
        p = LayerProfiler(
            gpu_spec=MOCK_GPU.model_copy(update={
                "supports_fp8": True,
                "available_kernels": {**MOCK_GPU.available_kernels, "W8A8_FP8": ["cublas_fp8"]},
            }),
            batch_sizes=[1], num_warmup=1, num_measurements=2,
            cache_dir=str(tmp_path),
        )
        assert Precision.W8A8_FP8 in p._available_precisions

    def test_int8_and_int4_included_when_int8_tc_supported(self, profiler):
        precisions = profiler._available_precisions
        assert Precision.W8A8_INT8 in precisions
        assert Precision.W4A16 in precisions


# ── LayerProfiler._profile_single_layer ──────────────────────────────────────

class TestProfileSingleLayer:
    def test_all_result_keys_present(self, profiler, small_linear):
        result = profiler._profile_single_layer(small_linear, batch_size=1)
        for key in ("memory_bytes", "latency_us", "peak_memory_bytes", "kernel_name", "is_memory_bound"):
            assert key in result

    def test_all_precisions_in_each_key(self, profiler, small_linear):
        result = profiler._profile_single_layer(small_linear, batch_size=1)
        for prec in profiler._available_precisions:
            pv = prec.value
            assert pv in result["memory_bytes"]
            assert result["latency_us"][pv] > 0
            assert isinstance(result["is_memory_bound"][pv], bool)

    def test_fp16_memory_greater_than_int4(self, profiler):
        layer = nn.Linear(256, 256, bias=False)
        result = profiler._profile_single_layer(layer, batch_size=1)
        assert result["memory_bytes"][Precision.FP16.value] > result["memory_bytes"][Precision.W4A16.value]

    def test_kernel_name_is_string(self, profiler, small_linear):
        result = profiler._profile_single_layer(small_linear, batch_size=1)
        for pv, kname in result["kernel_name"].items():
            assert isinstance(kname, str) and len(kname) > 0


# ── LayerProfiler.profile_all_layers ─────────────────────────────────────────

class TestProfileAllLayers:
    def _model(self):
        return nn.Sequential(
            nn.Linear(64, 32, bias=False),  # "0"
            nn.ReLU(),                       # "1" — skipped
            nn.Linear(32, 16, bias=False),  # "2"
        )

    def test_linear_layers_profiled(self, profiler):
        results = profiler.profile_all_layers(
            self._model(), layer_names=["0", "1", "2"], model_id="test"
        )
        assert "0" in results
        assert "2" in results

    def test_non_linear_skipped(self, profiler):
        results = profiler.profile_all_layers(
            self._model(), layer_names=["0", "1", "2"], model_id="skip_test"
        )
        assert "1" not in results  # ReLU

    def test_missing_layer_name_skipped(self, profiler):
        results = profiler.profile_all_layers(
            self._model(), layer_names=["0", "99"], model_id="partial"
        )
        assert "0" in results
        assert "99" not in results

    def test_cache_file_created(self, profiler, tmp_path):
        profiler.profile_all_layers(self._model(), ["0"], model_id="cache_write")
        assert any(tmp_path.glob("*.json"))

    def test_cache_hit_returns_same_results(self, profiler):
        r1 = profiler.profile_all_layers(self._model(), ["0"], model_id="cached")
        r2 = profiler.profile_all_layers(self._model(), ["0"], model_id="cached")
        assert r1 == r2

    def test_corrupt_cache_is_ignored(self, profiler):
        path = profiler._cache_path("corrupt")
        path.write_text("not valid json {{")
        # Should re-profile without raising
        result = profiler.profile_all_layers(self._model(), ["0"], model_id="corrupt")
        assert "0" in result

    def test_empty_layer_names_returns_empty(self, profiler):
        result = profiler.profile_all_layers(self._model(), [], model_id="empty")
        assert result == {}


# ── GPU_REGISTRY sanity ───────────────────────────────────────────────────────

class TestGPURegistry:
    @pytest.mark.parametrize("key", ["A100_80GB", "H100_SXM", "RTX_4090", "T4"])
    def test_registry_entries_are_valid(self, key):
        spec = GPU_REGISTRY[key]
        assert spec.compute_capability[0] >= 8
        assert spec.memory_gb > 0
        assert Precision.FP16.value in spec.available_kernels

    def test_h100_supports_fp8(self):
        assert GPU_REGISTRY["H100_SXM"].supports_fp8

    def test_a100_does_not_support_fp8(self):
        assert not GPU_REGISTRY["A100_80GB"].supports_fp8
