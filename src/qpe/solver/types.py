"""
Solver-agnostic data types that form the contract between the optimization
engine and all other QPE components. These types are the ONLY interface
through which the solver communicates with the rest of the system.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class Precision(str, Enum):
    """Candidate quantization precisions.

    Each value corresponds to a specific weight format and the dominant
    kernel family used for inference. The naming convention encodes
    weight format and activation format: W{w}A{a}_{type}.
    """

    FP16 = "FP16"           # No quantization. Baseline.
    W8A8_FP8 = "W8A8_FP8"  # FP8 e4m3 weights + activations. Requires SM >= 8.9.
    W8A8_INT8 = "W8A8_INT8" # INT8 SmoothQuant weights + activations. SM >= 8.0.
    W4A16 = "W4A16"         # INT4 weight-only (AWQ/GPTQ), FP16 activations. SM >= 8.0.


#  SolverInput is everything the solver needs

class LayerDescriptor(BaseModel):
    """
    Complete description of a single quantizable layer, combining
    sensitivity scoring outputs and hardware profiling outputs.

    This is the per-layer "row" in the solver's input table. It contains
    all the information any solver might need, without dictating how that
    information is used (as objective coefficient vs. constraint bound).

    Produced by: SensitivityScorer (sensitivity fields) +
                 HardwareProfiler (resource fields)
    Consumed by: any QuantizationSolver implementation
    """

    model_config = ConfigDict(frozen=True)

    # ── Identity ──
    layer_name: str         # e.g., "model.layers.12.mlp.gate_proj"
    layer_type: str         # "linear", "attn_qkv", "attn_out", "mlp_gate", etc
    layer_index: int        # Sequential position in model
    relative_depth: float   # layer_index / total_layers
    param_count: int        # Number of parameters in this layer

    # ── Sensitivity signals (from SensitivityScorer) ──
    # These are raw signals. The solver decides how to weight/combine them.
    hessian_trace: float            # Tr(H_i) / n_params — normalized curvature
    hessian_is_psd: bool            # False for ~15% of LLM layers [LLM-MQ]
    gradient_norm: float            # ||∇_{W_i} L||₂ — first-order sensitivity
    fisher_diagonal_mean: float     # mean(diag(F_i)) — information content
    activation_kurtosis: float      # Excess kurtosis — outlier severity
    channel_outlier_rate: float     # Fraction of channels > 6σ
    dynamic_range_ratio: float      # max(|X|) / median(|X|) per channel
    activation_max_magnitude: float # Peak activation value
    weight_range: float             # max(W) - min(W)
    weight_kurtosis: float          # Weight distribution shape

    # ── Per-precision resource measurements (from HardwareProfiler) ──
    # Keyed by Precision enum value. Only precisions supported by the
    # target GPU are present. Missing keys = precision unavailable.
    memory_bytes: dict[str, int]        # {precision: weight memory in bytes}
    latency_us: dict[str, float]        # {precision: decode latency in µs at target batch}
    peak_memory_bytes: dict[str, int]   # {precision: peak memory including activations}
    kernel_name: dict[str, str]         # {precision: which kernel was selected}
    is_memory_bound: dict[str, bool]    # {precision: True if memory-bandwidth-limited}


class SolverInput(BaseModel):
    """
    Complete input to any QuantizationSolver. Solver-agnostic.

    Contains everything needed to formulate the optimization problem
    under ANY formulation. The solver selects which fields become
    objectives and which become constraints based on its own logic.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: int = 2

    # ── Per-layer data ──
    layers: list[LayerDescriptor]

    # ── Global model metadata ──
    model_id: str
    total_param_count: int
    num_transformer_blocks: int

    # ── Hardware context ──
    gpu_name: str
    gpu_memory_bytes: int                       # Physical GPU RAM
    gpu_compute_capability: tuple[int, int]
    gpu_memory_bandwidth_tb_s: float

    # ── Available precision set (intersection of model support & GPU support) ──
    available_precisions: list[Precision]

    # ── KV cache sizing (global, not per-layer) ──
    kv_cache_bytes_per_token: dict[str, int]    # {kv_dtype: bytes per token per layer}
    max_sequence_length: int
    max_batch_size: int


# ──────────────────────────────────────────────────────────────────────
#  SolverOutput: the solver's decision, nothing solver-specific
# ──────────────────────────────────────────────────────────────────────

class LayerAssignment(BaseModel):
    """Precision assignment for a single layer."""

    model_config = ConfigDict(frozen=True)

    layer_name: str
    assigned_precision: Precision

    # Estimated per-layer contributions (for diagnostics and validation)
    estimated_quality_cost: float   # Ω_i · δ(b) — sensitivity × error scale
    estimated_memory_bytes: int
    estimated_latency_us: float


class SolverOutput(BaseModel):
    """
    Complete output from any QuantizationSolver. Solver-agnostic.

    Downstream components (Validator, Exporter) consume this without
    knowledge of which solver produced it or which formulation was used.

    The solver populates both resource totals and quality estimates
    regardless of which was the objective and which was the constraint.
    This enables the Validator to check all dimensions unconditionally.
    """

    model_config = ConfigDict(frozen=True)
    schema_version: int = 2

    # ── Per-layer decisions ──
    assignments: list[LayerAssignment]

    # ── Aggregate metrics (populated by all solvers) ──
    total_estimated_quality_cost: float     # Σ estimated_quality_cost
    total_memory_bytes: int                 # Σ estimated_memory_bytes (weights only)
    total_memory_with_kv_bytes: int         # Weights + KV cache at chosen kv_dtype
    total_latency_us: float                 # Σ estimated_latency_us
    average_bitwidth: float                 # Weighted by param_count

    # ── Solver diagnostics ──
    solver_name: str        # e.g., "ILPQualityMinimizer", "ILPResourceMinimizer"
    solver_status: str      # "optimal", "feasible", "infeasible", "timeout"
    solve_time_seconds: float
    formulation_used: str   # "quality_minimizing" | "resource_minimizing" | "pareto"

    # ── KV cache decision ──
    kv_cache_dtype: str     # "fp16", "fp8_e4m3", "int8"

    # ── Layer ranking (for diagnostics / visualization) ──
    sensitivity_ranking: list[str]  # Layer names, most-sensitive first

    def to_assignment_dict(self) -> dict[str, str]:
        """Convenience: {layer_name: precision_string} for export."""
        return {a.layer_name: a.assigned_precision.value for a in self.assignments}


#  FeedbackSignal: validation results, solver-agnostic

class FeedbackSignal(BaseModel):
    """
    The Validation Engine's output, consumed by the Pipeline Orchestrator
    to decide whether to accept the solution or re-invoke the solver.

    Critically, this type contains NO solver-specific correction logic.
    It provides facts (what the actual quality/resource numbers are) and
    suggestions (which layers are problematic). The Pipeline Orchestrator
    translates these facts into solver-specific constraint modifications
    via the SolverConfig's feedback_handler.
    """

    model_config = ConfigDict(frozen=True)

    # ── Measured quality ──
    fp16_perplexity: float
    quantized_perplexity: float
    perplexity_increase: float          # quantized - fp16
    perplexity_increase_pct: float      # (increase / fp16) × 100
    benchmark_scores: dict[str, float]  # {task_name: accuracy}
    fp16_benchmark_scores: dict[str, float]

    # ── Measured resources ──
    actual_peak_memory_gb: float
    actual_decode_latency_ms: float | None  # If serving benchmark was run

    # ── Per-layer diagnostics ──
    per_layer_kl_divergence: dict[str, float]   # KL(quantized || fp16) per layer
    layers_exceeding_kl_threshold: list[str]    # Layers with KL > threshold

    # ── Outcome ──
    passed: bool

    # ── Correction suggestions (solver-agnostic) ──
    suggested_pin_to_fp16: list[str]        # Layers that should be at higher precision
    suggested_quality_budget_scale: float   # 1.0 = no change, 0.7 = tighten, 1.3 = loosen