from __future__ import annotations

import argparse
import logging
from typing import List

import torch
import torch.nn as nn

from qpe.profiler.gpu_specs import GPUSpec
from qpe.profiler.layer_profiler import LayerProfiler
from qpe.profiler.exporter import build_profile_table, write_profile_json

log = logging.getLogger("qpe.cli")

# TODO : delete - it exists in gpu_specs.py - just adjust that one as you need
def _detect_gpu_spec() -> GPUSpec:
    """
    Minimal GPUSpec construction for CLI.
    Prefer a proper registry in gpu_specs.py later; this unblocks Day 1.
    """
    if not torch.cuda.is_available():
        # CPU fallback (values arbitrary; profiler will run on CPU path)
        return GPUSpec(
            name="cpu",
            compute_capability="0.0",
            memory_gb=0.0,
            memory_bandwidth_tb_s=0.0,
            supports_fp8=False,
            supports_fp4=False,
            supports_int8_tensor_core=False,
            peak_fp16_tflops=0.0,
            peak_int8_tops=0.0,
            available_kernels={},
        )

    props = torch.cuda.get_device_properties(0)
    name = props.name
    cc = (int(props.major), int(props.minor)) 
    mem_gb = float(props.total_memory) / (1024**3)

    # Lookup table for known GPUs (extend as needed).
    # TODO : move this into gpu_specs.py
    # Values are "good enough" for Day 1; Day 2+ we can make these precise.
    presets = {
        # Turing
        "RTX 2080 Ti": dict(
            memory_bandwidth_tb_s=0.616,     # ~616 GB/s
            supports_fp8=False,
            supports_fp4=False,
            supports_int8_tensor_core=True,
            peak_fp16_tflops=110.0,          # tensor-core fp16 peak (approx)
            peak_int8_tops=220.0,            # tensor-core int8 peak (approx)
        ),
        # Add others later if you want:
        # "A100": dict(memory_bandwidth_tb_s=1.555, supports_fp8=False, supports_fp4=False, supports_int8_tensor_core=True, peak_fp16_tflops=312.0, peak_int8_tops=624.0),
    }

    # Match keys loosely (since full name is "NVIDIA GeForce RTX 2080 Ti")
    preset = None
    for k, v in presets.items():
        if k in name:
            preset = v
            break

    if preset is None:
        # Conservative defaults so we still run.
        # Bandwidth and peaks mainly affect roofline heuristic; not Day 1 critical.
        preset = dict(
            memory_bandwidth_tb_s=0.5,
            supports_fp8=False,
            supports_fp4=False,
            supports_int8_tensor_core=True,
            peak_fp16_tflops=80.0,
            peak_int8_tops=160.0,
        )

    available_kernels = {
        "FP16": ["generic"],
        "W8A8_INT8": ["generic"],
        "W4A16": ["generic"],
        "W8A8_FP8": ["generic"],
    }

    return GPUSpec(
        name=name,
        compute_capability=cc,
        memory_gb=mem_gb,
        memory_bandwidth_tb_s=float(preset["memory_bandwidth_tb_s"]),
        supports_fp8=bool(preset["supports_fp8"]),
        supports_fp4=bool(preset["supports_fp4"]),
        supports_int8_tensor_core=bool(preset["supports_int8_tensor_core"]),
        peak_fp16_tflops=float(preset["peak_fp16_tflops"]),
        peak_int8_tops=float(preset["peak_int8_tops"]),
        available_kernels=available_kernels,
    )

def _toy_model() -> nn.Module:
    # Minimal fallback model so CLI works Day 1 even without HF integration
    return nn.Sequential(
        nn.Linear(4096, 4096, bias=False),
        nn.ReLU(),
        nn.Linear(4096, 11008, bias=False),
    )


def _default_layer_names(model: nn.Module) -> List[str]:
    # Only Linear layers, matching current profiler behavior
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            names.append(name)
    return names

# TODO : move this whole damn file into testing pls
def main() -> None:
    parser = argparse.ArgumentParser(prog="qpe")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="Profile model layers and write profile JSON")
    p.add_argument("--model-id", default="toy", help="Model identifier for provenance")
    p.add_argument("--out", required=True, help="Output path for profile JSON")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--layers", default="", help="Comma-separated layer names (default: all Linear layers)")
    p.add_argument("--num-warmup", type=int, default=50)
    p.add_argument("--num-measurements", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # TODO Day 3+: replace toy model with HF model loader
    model = _toy_model()

    # Minimal, robust: use the GPU name and let GPUSpec resolve defaults
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_spec = _detect_gpu_spec() 
    profiler = LayerProfiler(
        gpu_spec=gpu_spec,
        num_warmup=args.num_warmup,
        num_measurements=args.num_measurements,
        seq_len=args.seq_len,
        qpe_version="2.0",
    )

    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()] or _default_layer_names(model)

    profiles = profiler.profile_all_layers(
        model=model,
        layer_names=layer_names,
        target_batch_size=args.batch_size,
        model_id=args.model_id,
    )

    layer_metas = profiles.pop("__layer_metas__", {})

    table = build_profile_table(
        profiles=profiles,
        model_id=args.model_id,
        gpu_name=gpu_spec.name,
        qpe_version=profiler.qpe_version,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layer_metas=layer_metas,
    )

    write_profile_json(args.out, table)
    log.info("Wrote %d layer entries to %s", len(table.get("entries", {})), args.out)


if __name__ == "__main__":
    main()
