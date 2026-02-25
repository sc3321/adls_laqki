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
    Configuration for formulation (B): minimize hardware resource consumption subject to quality degradation budget
    
    User specifies how much quality loss they will tolerate
    Solver finds most compressed assignment (smallest memory, lowest latency) that stays within tolerance
    
    Use case:
        Production deployment: I need <2% degradation, minimize cost
        Fleet optimization: find the cheapest GPU that runs this model well
    
    Implementation note:
        Quality constraint uses proxy metric: sum[Omega_i * delta(b) * c_(i,b)] <= eps
        This is an approximation of actual quality loss
        
        Proxy accuracy degrades under aggressive compression
        Handled by validation feedback loop which performs bisection search on eps to find 
        value corresponding to user actual quality budget
        Solver treats eps as a given number
    """
    solver_name: str = "resource_minimizer"
    
    quality_budget_proxy: float
    
    resource_weights: dict[str, float] = {
        "memory": 1.0,
        "latency": 0.0,
    }
    
    hard_memory_cap_gb: Optional[float] = None
    
    target_batch_size: int = 1


class ParetoExplorerConfig(SolverConfig):
    """
    Configuration for multi-objective Pareto frontier exploration
    
    Does not commit to either formulation
    Explores full accuracy-vs-resource tradeoff space using Bayesian optimization
    Returns a set of Pareto-optimal configurations
    
    Use case:
        Understanding tradeoffs before committing to deployment point
        Generating data for reports/papers
        Informing constraint selection for subsequent ILP runs
    """
    solver_name: str = "pareto_explorer"
    
    num_trials: int = 200
    parallelism: int = 4
    objectives: dict[str, str] = {
        "quality_cost": "minimize",
        "memory_gb": "minimize",
    }
    outcome_constraints: list[str] = []
    
    hard_memory_cap_gb: Optional[float] = None