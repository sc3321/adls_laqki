from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .export_models import LayerCostEntry, LayerCostTable


def _get_torchao_version() -> Optional[str]:
    try:
        import torchao  # type: ignore
        return getattr(torchao, "__version__", "unknown")
    except Exception:
        return None


def build_cost_table(
    *,
    profiles: Dict[str, Dict[str, Dict[str, Any]]],
    model_id: str,
    gpu_name: str,
    qpe_version: str,
    batch_size: int,
    seq_len: int,
    regime: str,
    layer_metas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> LayerCostTable:
    """
    Convert LayerProfiler.profile_all_layers output into a LayerCostTable.

    profiles format today:
      { layer_name: { precision_str: layer_profile_dict } }
    """
    entries: list[LayerCostEntry] = []
    layer_metas = layer_metas or {}

    for layer_name, per_prec in profiles.items():
        meta = layer_metas.get(layer_name, {})
        layer_type = meta.get("layer_type", "unknown")
        layer_shape = meta.get("shape", [])
        param_count = int(meta.get("param_count", 0))
        dtype = meta.get("dtype", "unknown")

        for precision, prof in per_prec.items():
            # Day 1 compatibility:
            # existing profiler returns "latency_us" (median) and "memory_bytes"
            p50 = float(prof.get("p50_us", prof.get("latency_us", 0.0)))
            p99 = float(prof.get("p99_us", p50))  # until Day 2

            weight_bytes = int(prof.get("weight_bytes", prof.get("memory_bytes", 0)))
            peak_mem = int(prof.get("peak_mem_bytes", prof.get("peak_memory_bytes", 0)))

            entries.append(
                LayerCostEntry(
                    model_id=model_id,
                    layer_name=layer_name,
                    layer_type=str(prof.get("layer_type", layer_type)),
                    gpu_name=gpu_name,
                    precision=precision,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    regime=regime,  # "decode" or "prefill"
                    p50_us=p50,
                    p99_us=p99,
                    weight_bytes=weight_bytes,
                    peak_mem_bytes=peak_mem,
                    kernel_name=str(prof.get("kernel_name", "generic")),
                    is_memory_bound=bool(prof.get("is_memory_bound", False)),
                    layer_shape=list(layer_shape) if layer_shape else [],
                    param_count=param_count,
                    dtype=str(prof.get("dtype", dtype)),
                )
            )

    return LayerCostTable(
        qpe_version=qpe_version,
        torch_version=torch.__version__,
        torchao_version=_get_torchao_version(),
        gpu_name=gpu_name,
        model_id=model_id,
        metadata={
            "note": "p99_us equals p50_us until Day 2 timing upgrade",
        },
        entries=entries,
    )


def write_cost_table_json(path: str | Path, table: LayerCostTable) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(table.model_dump_json(indent=2), encoding="utf-8")


def write_cost_table_jsonl(path: str | Path, table: LayerCostTable) -> None:
    """
    Optional: JSONL can be convenient later. Day 1 not required.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in table.entries:
            f.write(json.dumps(e.model_dump(), sort_keys=True) + "\n")
