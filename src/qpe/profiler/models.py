from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LayerMeta(BaseModel):
    """Per-layer metadata -- constant across precisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer_type: str
    layer_shape: list[int] = Field(default_factory=list)
    dtype: str = "unknown"
    param_count: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> LayerMeta:
        return cls(
            layer_type=str(d.get("layer_type", "unknown")),
            layer_shape=list(d.get("shape", d.get("layer_shape", []))),
            dtype=str(d.get("dtype", "unknown")),
            param_count=int(d.get("param_count", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class LayerProfile(BaseModel):
    """
    Per-precision measurement for one profiling point.

    Identity of a record is:
      (layer, precision, batch_size, seq_len, regime)

    Notes:
      - latency_us is kept as the canonical solver-facing latency field.
      - In the new contract, latency_us should equal p50_us.
      - weight_bytes replaces the older ambiguous memory_bytes field.
      - from_dict remains backward-compatible with older cache entries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    precision: str
    batch_size: int
    seq_len: int
    regime: str

    latency_us: float
    p50_us: float = 0.0
    p99_us: float = 0.0

    weight_bytes: int
    peak_memory_bytes: int

    is_memory_bound: bool
    kernel_name: str
    fallback_occurred: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> LayerProfile:
        p50 = float(d.get("p50_us", d.get("latency_us", 0.0)))
        latency = float(d.get("latency_us", p50))
        return cls(
            precision=str(d.get("precision", "unknown")),
            batch_size=int(d.get("batch_size", 0)),
            seq_len=int(d.get("seq_len", 0)),
            regime=str(d.get("regime", "unknown")),
            latency_us=latency,
            p50_us=p50,
            p99_us=float(d.get("p99_us", p50)),
            weight_bytes=int(d.get("weight_bytes", d.get("memory_bytes", 0))),
            peak_memory_bytes=int(d.get("peak_memory_bytes", 0)),
            is_memory_bound=bool(d.get("is_memory_bound", False)),
            kernel_name=str(d.get("kernel_name", "generic")),
            fallback_occurred=bool(d.get("fallback_occurred", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class ModelProfileResult(BaseModel):
    """Top-level output of profile_all_layers -- complete profile payload."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 2
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    qpe_version: str
    torch_version: str
    torchao_version: str | None = None
    gpu_name: str
    model_id: str
    batch_size: int
    seq_len: int
    regime: str = "unknown"

    entries: dict[str, dict[str, LayerProfile]]
    layer_metas: dict[str, LayerMeta]

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"entries", "layer_metas"})
        payload["entries"] = {
            layer_name: {
                precision: profile.to_dict()
                for precision, profile in per_precision.items()
            }
            for layer_name, per_precision in self.entries.items()
        }
        payload["layer_metas"] = {
            layer_name: layer_meta.to_dict()
            for layer_name, layer_meta in self.layer_metas.items()
        }
        return payload

    def to_json(self, path: str | Path) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
