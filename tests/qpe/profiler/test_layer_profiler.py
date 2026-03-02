"""Profiler and cache tests for the hierarchical profile store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch.nn as nn

import qpe.profiler.layer_profiler as layer_profiler_module
from qpe.profiler.cache import ProfileCache
from qpe.profiler.gpu_specs import GPUSpec
from qpe.profiler.layer_profiler import LayerProfiler


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
        "FP16": ["cublas", "cutlass"],
        "W8A8_INT8": ["cutlass", "cublas"],
        "W4A16": ["marlin", "exllamav2", "autogptq"],
    },
)


def _make_cache(tmp_path: Path, precisions: list[str]) -> ProfileCache:
    return ProfileCache(
        root_dir=str(tmp_path),
        qpe_version="2.0",
        gpu_name=MOCK_GPU.name,
        supported_precisions=precisions,
    )


class TestProfileCache:
    def test_put_creates_hierarchy_and_status_files(self, tmp_path: Path):
        cache = _make_cache(tmp_path, ["FP16", "W8A8_INT8"])
        payload = {"latency_us": 10.0, "memory_bytes": 256}
        layer_meta = {"shape": [32, 64], "dtype": "torch.float16", "param_count": 2048}

        cache.put(
            model_id="org/model",
            layer_name="transformer.blocks.0.attn.q_proj",
            batch_size=4,
            precision="FP16",
            data=payload,
            layer_meta=layer_meta,
        )

        root = tmp_path / "v2.0" / "NVIDIA_Tesla_T4"
        profile_file = (
            root
            / "org--model"
            / "transformer.blocks.0.attn.q_proj"
            / "bs_4"
            / "FP16.json"
        )
        assert profile_file.exists()
        assert (root / "_precisions.json").exists()
        assert (root / "org--model" / "_status.json").exists()
        assert (
            root
            / "org--model"
            / "transformer.blocks.0.attn.q_proj"
            / "_meta.json"
        ).exists()

    def test_get_round_trip_removes_internal_meta(self, tmp_path: Path):
        cache = _make_cache(tmp_path, ["FP16"])
        payload = {"latency_us": 7.5, "memory_bytes": 64}
        layer_meta = {"shape": [8, 8], "dtype": "torch.float16", "param_count": 64}
        cache.put("model", "layer.0", 1, "FP16", payload, layer_meta)

        loaded = cache.get("model", "layer.0", 1, "FP16")
        assert loaded == payload

        on_disk = json.loads(
            (tmp_path / "v2.0" / "NVIDIA_Tesla_T4" / "model" / "layer.0" / "bs_1" / "FP16.json")
            .read_text(encoding="utf-8")
        )
        assert "_layer_meta" in on_disk

    def test_completion_rollup_tracks_batch_layer_and_model(self, tmp_path: Path):
        cache = _make_cache(tmp_path, ["FP16", "W8A8_INT8"])
        payload = {"latency_us": 1.0, "memory_bytes": 10}
        meta = {"shape": [4, 4], "dtype": "torch.float16", "param_count": 16}

        cache.put("model", "layer.0", 1, "FP16", payload, meta)
        assert not cache.is_batch_complete("model", "layer.0", 1)
        assert not cache.is_layer_complete("model", "layer.0")
        assert not cache.is_model_complete("model")

        cache.put("model", "layer.0", 1, "W8A8_INT8", payload, meta)
        assert cache.is_batch_complete("model", "layer.0", 1)
        assert cache.is_layer_complete("model", "layer.0")
        assert cache.is_model_complete("model")

        cache.put("model", "layer.0", 8, "FP16", payload, meta)
        assert not cache.is_layer_complete("model", "layer.0")
        assert not cache.is_model_complete("model")

    def test_verify_recovers_from_corrupt_status(self, tmp_path: Path):
        cache = _make_cache(tmp_path, ["FP16", "W8A8_INT8"])
        payload = {"latency_us": 2.5, "memory_bytes": 32}
        meta = {"shape": [2, 16], "dtype": "torch.float16", "param_count": 32}
        cache.put("model", "layer.0", 1, "FP16", payload, meta)
        cache.put("model", "layer.0", 1, "W8A8_INT8", payload, meta)

        status_path = tmp_path / "v2.0" / "NVIDIA_Tesla_T4" / "model" / "_status.json"
        status_path.write_text("{ not valid json", encoding="utf-8")

        rebuilt = cache.verify("model")
        assert rebuilt["model_complete"] is True
        assert rebuilt["layers"]["layer.0"]["batch_sizes"]["1"]["complete"] is True


@pytest.fixture
def model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(16, 16, bias=False),  # "0"
        nn.ReLU(),                      # "1" -> skipped
        nn.Linear(16, 8, bias=False),   # "2"
    )


@pytest.fixture
def fast_profiler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LayerProfiler:
    monkeypatch.setattr(
        layer_profiler_module,
        "_time_layer",
        lambda layer, inputs, num_warmup, num_measurements, device: 123.0,
    )
    monkeypatch.setattr(
        layer_profiler_module,
        "_measure_peak_memory",
        lambda layer, inputs, device: 4096,
    )
    monkeypatch.setattr(
        layer_profiler_module,
        "_is_memory_bound",
        lambda module, input_tensor, precision, gpu_spec, batch_size, sequence_length: False,
    )
    return LayerProfiler(
        gpu_spec=MOCK_GPU,
        batch_sizes=[1],
        num_warmup=1,
        num_measurements=1,
        seq_len=1,
        cache_dir=str(tmp_path),
        qpe_version="2.0",
    )


class TestLayerProfilerWithHierarchicalCache:
    def test_profile_all_layers_writes_hierarchical_profiles(
        self,
        fast_profiler: LayerProfiler,
        model: nn.Module,
        tmp_path: Path,
    ):
        results = fast_profiler.profile_all_layers(
            model=model,
            layer_names=["0", "1", "2"],
            target_batch_size=1,
            model_id="org/model",
        )

        assert "0" in results
        assert "2" in results
        assert "1" not in results

        base_dir = tmp_path / "v2.0" / "NVIDIA_Tesla_T4" / "org--model"
        assert (base_dir / "0" / "bs_1" / "FP16.json").exists()
        assert (base_dir / "2" / "bs_1" / "FP16.json").exists()
        assert fast_profiler.profile_cache.is_batch_complete("org/model", "0", 1)

    def test_cache_hit_avoids_reprofiling(
        self,
        fast_profiler: LayerProfiler,
        model: nn.Module,
        monkeypatch: pytest.MonkeyPatch,
    ):
        first = fast_profiler.profile_all_layers(
            model=model,
            layer_names=["0", "2"],
            target_batch_size=1,
            model_id="cache-model",
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("_profile_single_layer should not run on cache hit")

        monkeypatch.setattr(fast_profiler, "_profile_single_layer", fail_if_called)
        second = fast_profiler.profile_all_layers(
            model=model,
            layer_names=["0", "2"],
            target_batch_size=1,
            model_id="cache-model",
        )
        assert second == first

    def test_new_batch_size_still_profiles_when_needed(
        self,
        fast_profiler: LayerProfiler,
        model: nn.Module,
        monkeypatch: pytest.MonkeyPatch,
    ):
        fast_profiler.profile_all_layers(
            model=model,
            layer_names=["0"],
            target_batch_size=1,
            model_id="cache-model",
        )

        call_counter = {"count": 0}
        original = fast_profiler._profile_single_layer

        def wrapped(*args, **kwargs):
            call_counter["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(fast_profiler, "_profile_single_layer", wrapped)
        fast_profiler.profile_all_layers(
            model=model,
            layer_names=["0"],
            target_batch_size=8,
            model_id="cache-model",
        )
        assert call_counter["count"] == 1
