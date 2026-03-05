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

    def to_dict(self) -> dict:
        return self.model_dump()


class LayerProfile(BaseModel):
    """Per-precision measurement -- one per (layer, batch_size, precision)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_us: float
    memory_bytes: int
    peak_memory_bytes: int
    is_memory_bound: bool
    kernel_name: str
    p50_us: float = 0.0
    p99_us: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> LayerProfile:
        lat = float(d.get("latency_us", 0.0))
        return cls(
            latency_us=lat,
            memory_bytes=int(d.get("memory_bytes", 0)),
            peak_memory_bytes=int(d.get("peak_memory_bytes", 0)),
            is_memory_bound=bool(d.get("is_memory_bound", False)),
            kernel_name=str(d.get("kernel_name", "generic")),
            p50_us=float(d.get("p50_us", lat)),
            p99_us=float(d.get("p99_us", lat)),
        )

    def to_dict(self) -> dict:
        return self.model_dump()


class ModelProfileResult(BaseModel):
    """Top-level output of profile_all_layers -- complete profile payload."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
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
