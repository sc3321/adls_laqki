# dtype : FP16
# warmup: 20 iters
# timing: 100 iters
# metric: p50 latency
# output: ../baseline_costs/results/fp16_baseline.json

#!/usr/bin/env python3
"""
FP16 Linear/GEMM microbench (CUDA events)

Task A – FP16 Baseline

- Benchmarks representative Linear shapes derived from transformer projections.
- Uses CUDA events for per-iteration GPU timing (ms).
- Writes JSON to bench/fp16_baseline_results.json by default.
"""

import argparse
import json
import os
import statistics
import time
import platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch


# -----------------------------
# Shape specification
# -----------------------------

@dataclass(frozen=True)
class ShapeSpec:
    name: str
    M: int
    K: int
    N: int
    bias: bool = False


def default_shapes() -> List[ShapeSpec]:
    """
    8 representative shapes:
      - Small width (768) + LLaMA-scale (4096)
      - Decode-like (M small)
      - Prefill-like (M large)
    """

    shapes = []

    # Small model width (BERT-ish)
    K_small = 768
    shapes += [
        ShapeSpec("mlp_expand",  M=1,    K=K_small, N=4*K_small),
        ShapeSpec("attn_proj",   M=1,    K=K_small, N=K_small),
        ShapeSpec("mlp_expand",  M=512,  K=K_small, N=4*K_small),
        ShapeSpec("mlp_project", M=512,  K=4*K_small, N=K_small),
    ]

    # LLaMA-7B-ish width
    K_med = 4096
    shapes += [
        ShapeSpec("mlp_expand",  M=1,    K=K_med, N=4*K_med),      # decode-like
        ShapeSpec("mlp_expand",  M=4,    K=K_med, N=4*K_med),      # small batch decode
        ShapeSpec("attn_proj",   M=512,  K=K_med, N=K_med),        # prefill-like
        ShapeSpec("mlp_expand",  M=2048, K=K_med, N=4*K_med),      # large prefill
    ]

    return shapes


# -----------------------------
# Environment metadata
# -----------------------------

def get_env_meta(device: torch.device) -> Dict[str, Any]:
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }

    if device.type == "cuda":
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        meta.update({
            "gpu_name": prop.name,
            "gpu_compute_capability": f"{prop.major}.{prop.minor}",
            "gpu_total_mem_gb": round(prop.total_memory / (1024**3), 2),
        })

    return meta


# -----------------------------
# Benchmark core
# -----------------------------

@torch.inference_mode()
def bench_shape(spec: ShapeSpec, device, warmup: int, iters: int, seed: int):
    torch.manual_seed(seed)

    linear = torch.nn.Linear(spec.K, spec.N, bias=spec.bias, device=device, dtype=torch.float16)
    x = torch.randn((spec.M, spec.K), device=device, dtype=torch.float16)

    # Warmup
    for _ in range(warmup):
        y = linear(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        for _ in range(iters):
            start.record()
            y = linear(x)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    else:
        # CPU fallback
        for _ in range(iters):
            t0 = time.perf_counter()
            y = linear(x)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    p50 = statistics.median(times_ms)
    p90 = sorted(times_ms)[int(0.9 * (len(times_ms) - 1))]
    p99 = sorted(times_ms)[int(0.99 * (len(times_ms) - 1))]

    return {
        "shape": asdict(spec),
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "latency_ms_p99": p99,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="exp/baseline_costs/out/fp16_baseline_result.sjson")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Disable TF32 for consistency (mostly affects FP32, but log it)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    shapes = default_shapes()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = {
        "meta": get_env_meta(device),
        "dtype": "fp16",
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "results": []
    }

    for i, spec in enumerate(shapes):
        print(f"[{i+1}/{len(shapes)}] Running {spec}")
        rec = bench_shape(spec, device, args.warmup, args.iters, args.seed + i)
        results["results"].append(rec)

    # Sort slowest first for meeting summary
    results["summary_top3_slowest"] = sorted(
        results["results"],
        key=lambda r: r["latency_ms_p50"],
        reverse=True
    )[:3]

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
