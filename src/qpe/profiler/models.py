from __future__ import annotations

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
