import pulp
import numpy as np
from ..config import QualityMinimizerConfig
from ..models import SolverInput, SolverOutput, LayerDescriptor, LayerAssignment
from ..types import Precision

class ILPSolverMixin:
    """
    Shared helper methods for ILP-family solvers
    
    Both QualityMinimizer and ResourceMinimizer use identical logic for candidate set 
    construction, layer lookup, sensitivity aggregation, and output assembly
    """

    def _build_candidate_set(
        self, input: SolverInput
    ) -> list[list[str]]:
        """
        For each layer, determine which precisions are candidates
        
        A precision is a candidate for layer i if:
        1. It is in SolverInput available_precisions list
        2. Layer i has profiling data for that precision (key exists in layer.memory_bytes and layer.latency_us)
        3. The layer is not pinned to a different precision
        
        Returns list of lists where candidates[i] is the list of precision strings available for layer i
        """
        candidates = []
        pinned = self.config.pinned_layers
        available = {p.value for p in input.available_precisions}
        
        for layer in input.layers:
            if layer.layer_name in pinned:
                # Pinned: only the pinned precision is a candidate
                candidates.append([pinned[layer.layer_name]])
            else:
                layer_candidates = [
                    p for p in available
                    if p in layer.memory_bytes and p in layer.latency_us
                ]
                if not layer_candidates:
                    raise ValueError(
                        f"Layer {layer.layer_name} has no valid precision candidates. "
                        f"Available: {available}, profiled: {set(layer.memory_bytes.keys())}"
                    )
                candidates.append(layer_candidates)
        
        return candidates
    
    def _layer_index(
        self, layers: list[LayerDescriptor], layer_name: str
    ) -> int | None:
        """Find the index of a layer by name - returns None if not found"""
        for i, layer in enumerate(layers):
            if layer.layer_name == layer_name:
                return i
        return None
    
    def _build_output(
        self,
        problem: pulp.LpProblem,
        c: dict[tuple[int, str], pulp.LpVariable],
        input: SolverInput,
        sensitivities: np.ndarray,
        candidates: list[list[str]],
    ) -> SolverOutput:
        """
        Extract solution from solved PuLP problem into SolverOutput
        
        Reads binary variable values, computes per-layer and aggregate metrics, 
        and assembles immutable SolverOutput
        """
        import pulp
        
        status_map = {
            pulp.constants.LpStatusOptimal: "optimal",
            pulp.constants.LpStatusNotSolved: "timeout",
            pulp.constants.LpStatusInfeasible: "infeasible",
            pulp.constants.LpStatusUnbounded: "unbounded",
            pulp.constants.LpStatusUndefined: "undefined",
        }
        solver_status = status_map.get(problem.status, "unknown")
        
        if solver_status in ("infeasible", "unbounded", "undefined", "timeout"):
            return SolverOutput(
                assignments=[],
                total_estimated_quality_cost=float("inf"),
                total_memory_bytes=0,
                total_memory_with_kv_bytes=0,
                total_latency_us=0.0,
                average_bitwidth=0.0,
                solver_name=self.name,
                solver_status=solver_status,
                solve_time_seconds=problem.solutionTime,
                formulation_used=self._formulation_name(),
                kv_cache_dtype="fp16",
                sensitivity_ranking=[],
            )
        
        assignments = []
        for i, layer in enumerate(input.layers):
            for b in candidates[i]:
                if c[i, b].varValue is not None and c[i, b].varValue > 0.5:
                    error_scale = self.config.error_scale.get(b, 1.0)
                    assignments.append(LayerAssignment(
                        layer_name=layer.layer_name,
                        assigned_precision=Precision(b),
                        estimated_quality_cost=sensitivities[i] * error_scale,
                        estimated_memory_bytes=layer.memory_bytes[b],
                        estimated_latency_us=layer.latency_us[b],
                    ))
                    break
        
        total_quality = sum(a.estimated_quality_cost for a in assignments)
        total_memory = sum(a.estimated_memory_bytes for a in assignments)
        total_latency = sum(a.estimated_latency_us for a in assignments)
        
        bitwidth_map = {"FP16": 16, "W8A8_FP8": 8, "W8A8_INT8": 8, "W4A16": 4}
        total_params = sum(l.param_count for l in input.layers)
        avg_bw = sum(
            bitwidth_map.get(a.assigned_precision.value, 16) 
            * input.layers[i].param_count
            for i, a in enumerate(assignments)
        ) / max(total_params, 1)
        
        kv_dtype = "fp16"
        kv_bytes = input.kv_cache_bytes_per_token.get(kv_dtype, 0)
        total_kv = kv_bytes * input.max_sequence_length * input.max_batch_size * input.num_transformer_blocks
        
        ranked = sorted(
            zip(sensitivities, [l.layer_name for l in input.layers]),
            reverse=True,
        )
        
        return SolverOutput(
            assignments=assignments,
            total_estimated_quality_cost=total_quality,
            total_memory_bytes=total_memory,
            total_memory_with_kv_bytes=total_memory + total_kv,
            total_latency_us=total_latency,
            average_bitwidth=avg_bw,
            solver_name=self.name,
            solver_status=solver_status,
            solve_time_seconds=problem.solutionTime,
            formulation_used=self._formulation_name(),
            kv_cache_dtype=kv_dtype,
            sensitivity_ranking=[name for _, name in ranked],
        )
    
    def _formulation_name(self) -> str:
        """Return the formulation identifier for SolverOutput"""
        raise NotImplementedError