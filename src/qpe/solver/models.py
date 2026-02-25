from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Tuple

from .types import Precision

class LayerDescriptor(BaseModel) :
    """
    Complete description of a single quantizable layer, combining sensitivity scoring outputs and hardware profiling outputs
    This is the per-layer "row" in the solver's input table. It contains all the information any solver might need, without dictating how that information is used (as objective coefficient vs. constraint bound)
    """

    model_config = ConfigDict(frozen = True)

    layer_name : str
    layer_type : str
    layer_index : int
    relative_depth : float # layer_index / total_layers
    param_count : int 

    # Sensititvy signals 
    hessian_trace : float # Tr(H_i) / n_params - normalized curvature
    hessian_is_psd : bool # false for about 15% of LLM layers
    gradient_norm : float  # first order sensitiviy
    fisher_diagonal_mean : float
    activation_kurtosis : float
    chanel_outlier_rate : float
    dynamic_range_ratio : float 
    activation_max_magnitude : float 
    weight_range : float
    weight_kurtosis : float 

    # Per precision resoruce measurements - keyed by prec enum value - missing keys = prec unavaialble
    memory_bytes: Dict[str, int]            # {precision: weight memory in bytes}
    latency_us: Dict[str, float]            # {precision: decode latency in micro-s at target batch}
    peak_memory_bytes: Dict[str, int]       # {precision: peak memory including activations}
    kernel_name: Dict[str, str]             # {precision: which kernel was selected}
    is_memory_bound: Dict[str, bool]        # {precision: true if mem-bw-limited}


class SolverInput(BaseModel) :
    """
    Contains everything needed to formulate the optimization problem under a given formulation 
    The solver selects which fields become objectives and which become constraints based on its own logic
    """

    model_config = ConfigDict(frozen=True)

    layers : List[LayerDescriptor]

    model_id : str 
    total_param_count : int
    num_transformer_blocks : int

    gpu_name : str
    gpu_memory_bytes : int
    gpu_compute_capability : Tuple[int,int]
    gpu_memory_bandwidth_tb_s : float 

    available_precisions : List[Precision]

    kv_cache_bytes_per_token : Dict[str,int]
    max_sequence_length : int
    max_batch_size : int


class LayerAssignment(BaseModel) :
    model_config = ConfigDict(frozen=True)
    layer_name : str
    assigned_precision : Precision 

    estimated_quality_cost : float # sensitivity x eror scale
    estimated_memory_bytes : int
    estimated_latency_us : float 

class SolverOutput(BaseModel) :
    """
    The solver populates both resource totals and quality estimates regardless of which was the objective and which was the constraint
    This enables the Validator to check all dimensions unconditionally
    """

    model_config = ConfigDict(frozen=True)

    assignments : List[LayerAssignment]

    total_estimated_quality_cost : float 
    total_memory_bytes : int
    total_memory_with_kv_bytes : int # Weights + KV cache at chosen kv_dtype
    total_latency_us : float
    average_bitwidth : float 

    solver_name: str 
    solver_status: str   # optimal, feasible, infeasible, timeout
    solve_time_seconds: float
    formulation_used: str  # quality_minimizing | resource_minimizing | pareto
    
    # KV cache decision 
    kv_cache_dtype: str 
    
    # Layer ranking (for diagnostics / visualization) 
    sensitivity_ranking: List[str] # Layer names, most-sensitive first
    
    def to_assignment_dict(self) -> Dict[str, str]:
        """Convenience: {layer_name: precision_string} for export."""
        return {a.layer_name: a.assigned_precision.value for a in self.assignments}



class FeedbackSignal(BaseModel):
    """
    The Validation Engine's output, used to decide whether to accept the solution or re-invoke the solver
    """
    model_config = ConfigDict(frozen=True)
    
    # Measured quality
    fp16_perplexity: float
    quantized_perplexity: float
    perplexity_increase: float              # quantized - fp16
    perplexity_increase_pct: float          # (increase / fp16) x 100
    benchmark_scores: Dict[str, float]      # {task_name: accuracy}
    fp16_benchmark_scores: Dict[str, float]
    
    # Measured resources
    actual_peak_memory_gb: float
    actual_decode_latency_ms: float | None  # If serving benchmark was run
    
    # Per-layer diagnostics
    per_layer_kl_divergence: Dict[str, float]  # KL(quantized || fp16) per layer
    layers_exceeding_kl_threshold: List[str]   # Layers with KL > threshold
    
    # Outcome
    passed: bool
    
    # Correction suggestions
    suggested_pin_to_fp16: List[str]  # Layers that should be at higher precision
    suggested_quality_budget_scale: float  # 1.0 = no change, 0.7 = tighten, 1.3 = loosen