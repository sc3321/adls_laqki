from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Regime = Literal["decode", "prefill"]


class LayerCostEntry(BaseModel):
    """
    One profiled datapoint for a layer under a specific config.
    This is the atomic unit the solver consumes.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Identity / config
    model_id: str
    layer_name: str
    layer_type: str
    gpu_name: str
    precision: str
    batch_size: int
    seq_len: int
    regime: Regime

    # Latency (Day 1: p50 populated; p99 may equal p50 until Day 2)
    p50_us: float
    p99_us: float

    # Memory
    weight_bytes: int
    peak_mem_bytes: int

    # Other useful signals
    kernel_name: str
    is_memory_bound: bool

    # Optional but useful metadata (keep stable)
    layer_shape: list[int] = Field(default_factory=list)
    param_count: int = 0
    dtype: str = "unknown"


class LayerCostTable(BaseModel):
    """
    Versioned payload: the thing written to disk and consumed by solver.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    qpe_version: str
    torch_version: str
    torchao_version: Optional[str] = None

    gpu_name: str
    model_id: str

    # Free-form metadata for provenance/debugging (safe to ignore downstream)
    metadata: dict[str, Any] = Field(default_factory=dict)

    entries: list[LayerCostEntry]
