"""
Solver configurations
 - The base SolverConfig contains fields common to all solvers
 - Subclasses add formulation-specific parameters

Currently implemented two kinds of optimization formulations : 
    (A) Given a set of hardware constraints, objective is minmizing model quality loss
    (B) Given a constraint on quality degradation (delta(b)), objective is to optimize hardware performance metrics 
"""
from .types import Precision
from pydantic import BaseModel
from typing import Optional 

class SolverConfig(BaseModel):
    """Base configuration for all solvers."""
    solver_name: str # Registry key (e.g., quality_minimizer)
    precision_candidates: list[Precision] = [
        Precision.FP16, Precision.W8A8_FP8, Precision.W8A8_INT8, Precision.W4A16
    ]
    kv_cache_candidates: list[str] = ["fp16", "fp8_e4m3"]
    
    # Sensitivity aggregation weights - how raw signals combine into a single per-layer sensitivity score - used by all ILP-family solvers
    sensitivity_weights: dict[str, float] = {
        "hessian_trace": 0.40,
        "gradient_norm": 0.20,
        "activation_kurtosis": 0.15,
        "channel_outlier_rate": 0.10,
        "fisher_diagonal_mean": 0.10,
        "dynamic_range_ratio": 0.05,
    }
    
    # Per-precision expected quantization error scale factors
    # delta(b): how much error a given precision introduces, relative to INT4=1.0
    # Calibrated from empirical data [Liu et al., 2025 - Quantization Hurts Reasoning?] : W8A8 is near-lossless, INT4 introduces measurable degradation on reasoning tasks
    error_scale: dict[str, float] = {
        "FP16": 0.0,
        "W8A8_FP8": 0.05,
        "W8A8_INT8": 0.10,
        "W4A16": 1.00,
    }
    
    # Layers to force to a specific precision regardless of optimization
    # Common use: pin embedding layers and lm_head to FP16 (they are small and highly sensitive to quantization in most models)
    pinned_layers: dict[str, str] = {}
    
    # Solver time budget
    timeout_seconds: float = 60.0


class QualityMinimizerConfig(SolverConfig):
    """
    Configuration for formulation (A) : minimize quality loss subject to hardware resource budgets.
    
        - The user specifies how much hardware they have 
        - The solver finds the assignment with the lowest total quality degradation that fits
    
    Use case: 
        -> Research/benchmarking : "what's the best quality I can getin 40GB?"
        -> Hardware-constrained deployment : "I have exactly one A100"
    """
    solver_name: str = "quality_minimizer"
    
    # Hard resource budgets
    memory_budget_gb: float                 # Max model weight memory
    latency_budget_us: Optional[float] = None  # Max per-token decode latency (optional)
    target_batch_size: int = 1              # Batch size for latency constraint
    
    # Optional soft constraint: average bit-width range
    min_avg_bitwidth: Optional[float] = None
    max_avg_bitwidth: Optional[float] = None


class ResourceMinimizerConfig(SolverConfig):
    """
    Configuration for formulation (B): minimize hardware resource consumption subject to a quality degradation budget.
        - The user specifies how much quality loss they will tolerate. 
        - The solver finds the most compressed assignment (smallest memory, lowest latency) that stays within that tolerance.
    
    Use case: 
        -> Production deployment : "I need <2% degradation, minimize cost"
        -> Fleet optimization : "find the cheapest GPU that runs this model well"
    
    Implementation note 
        - The quality constraint currently uses the proxy metric : sum[Omega_i x delta(b) x c_(i,b)] <= eps, which is an approx of actual quality loss. 
    
    The proxy's accuracy degrades under aggressive compression.
        - This is handled by the validation feedback loop, which performs bisection search on eps to find the value that corresponds to the user's actual quality budget. 
        - The solver itself does not need to know about this calibration - it treats eps as a given number.
    """
    solver_name: str = "resource_minimizer"
    
    # Quality budget (proxy scale - calibrated by the feedback loop)
    quality_budget_proxy: float             # eps: max (sum[ Omega_i x delta(b) x c_{i,b}])
    
    # Resource objective weights - relative importance of each resource dimension in the combined objective
    # Default: memory-only 
    resource_weights: dict[str, float] = {
        "memory": 1.0,
        "latency": 0.0,  # Set >0 to include latency in objective
    }
    
    # Physical feasibility bound - even when minimizing resources, the model must physically fit on the GPU
    # -> This is distinct from the quality budget
    hard_memory_cap_gb: Optional[float] = None  # Defaults to GPU physical memory
    
    # Batch size for latency terms (if latency weight > 0)
    target_batch_size: int = 1


class ParetoExplorerConfig(SolverConfig):
    """
    Configuration for multi-objective Pareto frontier exploration
    
    Does not commit to either formulation :
        - Explores the full accuracy-vs-resource tradeoff space using Bayesian optimization
        - Returns a set of Pareto-optimal configurations
    
    Use case: 
        -> Understanding tradeoffs before committing to a deployment point 
        -> Generating data for reports/papers
        -> Informing constraint selection for subsequent ILP runs
    """
    solver_name: str = "pareto_explorer"
    
    num_trials: int = 200
    parallelism: int = 4 # Parallel BO evaluations
    objectives: dict[str, str] = {
        "quality_cost": "minimize", # Proxy quality degradation
        "memory_gb": "minimize",
    }
    outcome_constraints: list[str] = []     # e.g., ["memory_gb <= 80"]
    
    # Physical feasibility
    hard_memory_cap_gb: Optional[float] = None