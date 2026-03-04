#!/usr/bin/env python3
"""
Task B — Quant Cost Isolation (separate script)

Measures, per shape:
  1) Activation quant time (FP16 -> INT8)
     - with scale computation (amax)  [realistic upper bound]
     - with fixed precomputed scale   [best-case lower bound]
  2) Dequant time (INT8 -> FP16)
  3) Optional: weight quant "once" time (K,N) FP16 -> INT8

Timing uses CUDA events per-iteration (like Task A).

Output:
  bench/quant_cost_isolation.json   (default)

Run:
  python quant_cost_isolation.py
  python quant_cost_isolation.py --out bench/quant_cost_isolation.json --warmup 20 --iters 100
"""

import argparse
import json
import os
import statistics
import time
import platform
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

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


def default_shapes() -> List[ShapeSpec]:
    # Same idea as Task A: small + LLaMA-ish widths, decode-like + prefill-like M.
    shapes: List[ShapeSpec] = []

    K_small = 768
    shapes += [
        ShapeSpec("mlp_expand",  M=1,    K=K_small, N=4 * K_small),
        ShapeSpec("attn_proj",   M=1,    K=K_small, N=K_small),
        ShapeSpec("mlp_expand",  M=512,  K=K_small, N=4 * K_small),
        ShapeSpec("mlp_project", M=512,  K=4 * K_small, N=K_small),
    ]

    K_med = 4096
    shapes += [
        ShapeSpec("mlp_expand",  M=1,    K=K_med, N=4 * K_med),
        ShapeSpec("mlp_expand",  M=4,    K=K_med, N=4 * K_med),
        ShapeSpec("attn_proj",   M=512,  K=K_med, N=K_med),
        ShapeSpec("mlp_expand",  M=2048, K=K_med, N=4 * K_med),
    ]

    return shapes


# -----------------------------
# Quant / dequant primitives
# -----------------------------

def compute_scale_per_tensor(x_fp16: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Scale = max(|x|) / 127  (scalar)
    # Use float32 for the reduction for a stable reference scale.
    amax = x_fp16.abs().to(torch.float32).amax()
    scale = amax / 127.0
    return torch.clamp(scale, min=eps).to(torch.float16)


def quantize_int8_per_tensor(x_fp16: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # q = clamp(round(x / scale), -127, 127).to(int8)
    # Compute in float32, then cast.
    x_f = x_fp16.to(torch.float32)
    s_f = scale.to(torch.float32)
    q = torch.round(x_f / s_f)
    q = torch.clamp(q, -127, 127).to(torch.int8)
    return q


def dequantize_int8_per_tensor(q_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x_hat = q * scale  (output FP16)
    return (q_int8.to(torch.float16) * scale)


# -----------------------------
# Timing helpers
# -----------------------------

def pctl(sorted_vals: List[float], p: float) -> float:
    assert 0.0 <= p <= 1.0
    if not sorted_vals:
        return float("nan")
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


@torch.inference_mode()
def time_cuda_events(fn, warmup: int, iters: int) -> Dict[str, Any]:
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    s = sorted(times_ms)
    return {
        "p50_ms": statistics.median(times_ms),
        "p90_ms": pctl(s, 0.90),
        "p99_ms": pctl(s, 0.99),
        # keep the list optional via flag in main
        "all_ms": times_ms,
    }


def get_env_meta(device: torch.device) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
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
            "gpu_cc": f"{prop.major}.{prop.minor}",
            "gpu_total_mem_gb": round(prop.total_memory / (1024**3), 2),
        })
    return meta


# -----------------------------
# Bench per shape
# -----------------------------

@torch.inference_mode()
def bench_one_shape(spec: ShapeSpec, device: torch.device, warmup: int, iters: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)

    # Allocate once (no allocations inside timed fns)
    x = torch.randn((spec.M, spec.K), device=device, dtype=torch.float16)

    # For dequant timing, create a representative INT8 output tensor of shape (M, N)
    qy = torch.randint(-127, 128, (spec.M, spec.N), device=device, dtype=torch.int8)

    # Fixed scale for lower-bound timing (computed once, outside timed loop)
    scale_x_fixed = compute_scale_per_tensor(x)
    scale_y_fixed = torch.tensor(1.0, device=device, dtype=torch.float16)

    # --- Activation quant WITH scale compute (upper bound) ---
    def act_quant_with_scale():
        scale = compute_scale_per_tensor(x)
        _ = quantize_int8_per_tensor(x, scale)

    # --- Activation quant WITHOUT scale compute (lower bound) ---
    def act_quant_fixed_scale():
        _ = quantize_int8_per_tensor(x, scale_x_fixed)

    # --- Dequant ---
    def dequant():
        _ = dequantize_int8_per_tensor(qy, scale_y_fixed)

    act_with = time_cuda_events(act_quant_with_scale, warmup=warmup, iters=iters)
    act_fixed = time_cuda_events(act_quant_fixed_scale, warmup=warmup, iters=iters)
    deq = time_cuda_events(dequant, warmup=warmup, iters=iters)

    # Optional: weight quant once (K, N) — useful for “amortise weights” story
    W = torch.randn((spec.K, spec.N), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    scale_w = compute_scale_per_tensor(W)
    _ = quantize_int8_per_tensor(W, scale_w)
    end.record()
    torch.cuda.synchronize()
    wq_once_ms = start.elapsed_time(end)

    return {
        "shape": asdict(spec),
        "act_quant_with_scale": act_with,
        "act_quant_fixed_scale": act_fixed,
        "dequant": deq,
        "weight_quant_once_ms": wq_once_ms,
    }


def strip_all_ms(d: Dict[str, Any]) -> None:
    for k in ["act_quant_with_scale", "act_quant_fixed_scale", "dequant"]:
        if k in d and isinstance(d[k], dict):
            d[k].pop("all_ms", None)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="exp/baseline_cost/out/quant_cost_isolation.json")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-all", action="store_true", help="Include per-iteration timing arrays in JSON.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script is intended for CUDA (Task B).")

    shapes = default_shapes()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results: Dict[str, Any] = {
        "meta": get_env_meta(device),
        "dtype": "fp16_inputs_int8_quant",
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "results": [],
    }

    for i, spec in enumerate(shapes):
        print(f"[{i+1}/{len(shapes)}] {spec.name} M={spec.M} K={spec.K} N={spec.N}")
        rec = bench_one_shape(spec, device, warmup=args.warmup, iters=args.iters, seed=args.seed + i)
        if not args.save_all:
            strip_all_ms(rec)
        results["results"].append(rec)

    # Meeting-friendly summary: top-3 by (act_with_scale p50 + dequant p50)
    def overhead_key(r: Dict[str, Any]) -> float:
        return float(r["act_quant_with_scale"]["p50_ms"]) + float(r["dequant"]["p50_ms"])

    sorted_by_overhead = sorted(results["results"], key=overhead_key, reverse=True)
    results["summary_top3_highest_quant_plus_dequant"] = [
        {
            "shape": r["shape"],
            "act_quant_with_scale_p50_ms": r["act_quant_with_scale"]["p50_ms"],
            "dequant_p50_ms": r["dequant"]["p50_ms"],
            "sum_p50_ms": overhead_key(r),
            "weight_quant_once_ms": r["weight_quant_once_ms"],
        }
        for r in sorted_by_overhead[:3]
    ]

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {args.out}")
    if not args.save_all:
        print("Note: per-iteration timing arrays were omitted (use --save-all to include).")


if __name__ == "__main__":
    main()
