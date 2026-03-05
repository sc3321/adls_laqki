from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .models import LayerMeta, LayerProfile


def _get_torchao_version() -> Optional[str]:
    try:
        import torchao  # type: ignore
        return getattr(torchao, "__version__", "unknown")
    except Exception:
        return None


def build_profile_table(
    *,
    profiles: Dict[str, Dict[str, Dict[str, Any]]],
    model_id: str,
    gpu_name: str,
    qpe_version: str,
    batch_size: int,
    seq_len: int,
    layer_metas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convert LayerProfiler.profile_all_layers output into a serialisable dict.

    Each per-precision dict is validated through LayerProfile.from_dict() and
    each layer meta through LayerMeta.from_dict() before inclusion.

    Returns:
        {
            "schema_version": 1,
            "generated_at":   str,
            "qpe_version":    str,
            "torch_version":  str,
            "torchao_version": str | None,
            "gpu_name":       str,
            "model_id":       str,
            "batch_size":     int,
            "seq_len":        int,
            "entries": {
                layer_name: {
                    precision: LayerProfile dict (7 fields),
                    ...
                },
                ...
            },
            "layer_metas": {
                layer_name: LayerMeta dict (4 fields),
                ...
            },
        }
    """
    layer_metas = layer_metas or {}

    validated_entries: Dict[str, Dict[str, Any]] = {}
    for layer_name, per_prec in profiles.items():
        validated_precs: Dict[str, Any] = {}
        for precision, prof in per_prec.items():
            validated_precs[precision] = LayerProfile.from_dict(prof).to_dict()
        validated_entries[layer_name] = validated_precs

    validated_metas: Dict[str, Any] = {}
    for layer_name, meta in layer_metas.items():
        validated_metas[layer_name] = LayerMeta.from_dict(meta).to_dict()

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qpe_version": qpe_version,
        "torch_version": torch.__version__,
        "torchao_version": _get_torchao_version(),
        "gpu_name": gpu_name,
        "model_id": model_id,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "entries": validated_entries,
        "layer_metas": validated_metas,
    }

# TODO : @shree, this is done in the cache xo (see cache pls)
# def write_profile_json(path: str | Path, table: Dict[str, Any]) -> None:
#     """Write a profile table dict to a JSON file."""
#     p = Path(path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     p.write_text(json.dumps(table, indent=2), encoding="utf-8")
