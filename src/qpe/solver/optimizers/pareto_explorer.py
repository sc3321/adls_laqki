"""
Multi-objective Bayesian optimization for Pareto frontier exploration

Uses Ax Platform with BoTorch's qLogNEHVI (quasi-Log Noisy Expected Hypervolume Improvement) acquisition function 
    - state-of-the-art for multi-objective optimization with discrete parameters and noisy evaluations

This solver does not commit to either formulation 
    - It explores the full accuracy-vs-resource tradeoff space, returning a set of Pareto-optimal configurations 
    - Can use it to understand tradeoffs before committing to a deployment point, or to generate data for reports

Reference: Ax benchmarks show it is the best platform for mixed/discrete, multi-objective, constrained, and noisy problems [Meta, 2023]
"""
import numpy as np
from ax.service.ax_client import AxClient, ObjectiveProperties
from ..config import ParetoExplorerConfig
from ..models import SolverInput, SolverOutput, LayerAssignment
from ..types import Precision

class ParetoExplorer:
    """Explore Pareto frontier via multi-objective BO"""
    
    def __init__(self, config: ParetoExplorerConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        return "ParetoExplorer"
    
    def solve(self, input: SolverInput) -> SolverOutput:
        """Returns the Pareto-optimal point with lowest quality cost"""
        frontier = self.explore(input)
        best = min(frontier, key=lambda r: r.total_estimated_quality_cost)
        return best
    
    def explore(self, input: SolverInput) -> list[SolverOutput]:
        """Return full Pareto frontier as list of SolverOutput"""
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {
                    "name": f"bits_{layer.layer_name}",
                    "type": "choice",
                    "values": list(layer.memory_bytes.keys()),
                    "is_ordered": True,
                }
                for layer in input.layers
                if layer.layer_name not in self.config.pinned_layers
            ],
            objectives={
                name: ObjectiveProperties(minimize=(direction == "minimize"))
                for name, direction in self.config.objectives.items()
            },
            outcome_constraints=self.config.outcome_constraints,
        )
        
        for _ in range(self.config.num_trials):
            trial_params, trial_index = ax_client.get_next_trial()
            metrics = self._evaluate_configuration(trial_params, input)
            ax_client.complete_trial(trial_index, raw_data=metrics)
        
        pareto_params = ax_client.get_pareto_optimal_parameters()
        return [self._params_to_output(p, input) for p in pareto_params.values()]
    
    def _evaluate_configuration(
        self, trial_params: dict[str, str], input: SolverInput
    ) -> dict[str, float]:
        """
        Evaluate a single Pareto trial precision assignment
        
        Translates Ax trial parameters (layer_name -> precision string) into aggregate 
        quality and resource metrics WITHOUT running full model inference
        Uses proxy metrics from SolverInput (sensitivity scores and profiled latency/memory)
        
        This is intentionally cheap - Pareto exploration requires hundreds of evaluations
        Full model inference would be prohibitively expensive
        Proxy metrics are sufficient for Pareto frontier discovery - Pipeline validates 
        final selected point with real evaluation
        
        Args:
            trial_params: {f"bits_{layer_name}": precision_string, ...}
            input: The SolverInput with per-layer data
        
        Returns:
            {"quality_cost": float, "memory_gb": float, "latency_ms": float}
        """
        sensitivities = self._aggregate_sensitivities(input.layers)
        
        total_quality = 0.0
        total_memory = 0
        total_latency = 0.0
        
        for i, layer in enumerate(input.layers):
            param_key = f"bits_{layer.layer_name}"
            if param_key in trial_params:
                precision = trial_params[param_key]
            elif layer.layer_name in self.config.pinned_layers:
                precision = self.config.pinned_layers[layer.layer_name]
            else:
                precision = "FP16"  # Default for layers not in trial
            
            error_scale = self.config.error_scale.get(precision, 0.0)
            total_quality += sensitivities[i] * error_scale
            total_memory += layer.memory_bytes.get(precision, 0)
            total_latency += layer.latency_us.get(precision, 0.0)
        
        return {
            "quality_cost": total_quality,
            "memory_gb": total_memory / 1e9,
            "latency_ms": total_latency / 1000.0,
        }

    def _params_to_output(
        self, params: dict, input: SolverInput
    ) -> SolverOutput:
        """
        Convert Ax Pareto-optimal trial parameters into a SolverOutput
        
        Bridges Ax API (dict of parameter names -> values) to solver-agnostic SolverOutput type
        Ensures Pareto results are consumable by Validator and Exporter with zero changes
        """
        sensitivities = self._aggregate_sensitivities(input.layers)
        
        assignments = []
        for i, layer in enumerate(input.layers):
            param_key = f"bits_{layer.layer_name}"
            if param_key in params:
                precision_str = params[param_key]
            elif layer.layer_name in self.config.pinned_layers:
                precision_str = self.config.pinned_layers[layer.layer_name]
            else:
                precision_str = "FP16"
            
            error_scale = self.config.error_scale.get(precision_str, 0.0)
            assignments.append(LayerAssignment(
                layer_name=layer.layer_name,
                assigned_precision=Precision(precision_str),
                estimated_quality_cost=sensitivities[i] * error_scale,
                estimated_memory_bytes=layer.memory_bytes.get(precision_str, 0),
                estimated_latency_us=layer.latency_us.get(precision_str, 0.0),
            ))
        
        bitwidth_map = {"FP16": 16, "W8A8_FP8": 8, "W8A8_INT8": 8, "W4A16": 4}
        total_params = sum(l.param_count for l in input.layers)
        
        return SolverOutput(
            assignments=assignments,
            total_estimated_quality_cost=sum(a.estimated_quality_cost for a in assignments),
            total_memory_bytes=sum(a.estimated_memory_bytes for a in assignments),
            total_memory_with_kv_bytes=sum(a.estimated_memory_bytes for a in assignments),
            total_latency_us=sum(a.estimated_latency_us for a in assignments),
            average_bitwidth=sum(
                bitwidth_map.get(a.assigned_precision.value, 16) * input.layers[i].param_count
                for i, a in enumerate(assignments)
            ) / max(total_params, 1),
            solver_name=self.name,
            solver_status="optimal",
            solve_time_seconds=0.0,
            formulation_used="pareto",
            kv_cache_dtype="fp16",
            sensitivity_ranking=[],
        )
    
    def _aggregate_sensitivities(self, layers) -> np.ndarray:
        """Aggregate per-layer sensitivity scores using configured weights"""
        sensitivities = []
        for layer in layers:
            score = (
                self.config.sensitivity_weights.get("hessian_trace", 0.0) * layer.hessian_trace +
                self.config.sensitivity_weights.get("gradient_norm", 0.0) * layer.gradient_norm +
                self.config.sensitivity_weights.get("activation_kurtosis", 0.0) * layer.activation_kurtosis +
                self.config.sensitivity_weights.get("channel_outlier_rate", 0.0) * layer.chanel_outlier_rate +
                self.config.sensitivity_weights.get("fisher_diagonal_mean", 0.0) * layer.fisher_diagonal_mean +
                self.config.sensitivity_weights.get("dynamic_range_ratio", 0.0) * layer.dynamic_range_ratio
            )
            sensitivities.append(score)
        return np.array(sensitivities)