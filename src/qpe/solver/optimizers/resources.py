"""
ILP solver that minimizes resource consumption subject to a quality budget.

Mathematical formulation:

    minimize    alpha x Sum_i [ Sum_b [M_i_b x c(i,b)  +  beta]] x Sum_i [ Sum_b [L(i,b) x c(i,b)]]

    subject to:
        Sum_i [Sum_b  [Omega_i x delta(b) x c(i,b)]]  <= eps                                        (quality budget)
        Sum_i Sum_b M_i_b x c(i,b)                    <= M_cap                                      (physical memory cap)
        Sum_b [c(i,b)]                                 = 1          for all i                       (one precision per layer)
        c(i,b)                                         = 0          if precision b unavailable      (hardware filter)
        c(i,b)                                         = 1          if layer i pinned to b          (user overrides)

    where:
        alpha, beta   - resource weights (user-configurable, default: alpha=1, beta=0 -> memory-only)
        eps           - quality budget on the proxy metric
                        not directly the user's target - the Pipeline's feedback loop calibrates eps to match the user's actual quality tolerance via bisection over empirical validation results

This is the Lagrangian dual of the Quality Minimizer. 
    -> For continuous LPs, strong duality guarantees both formulations trace the same Pareto frontier. 
    -> For ILPs, there is a small integrality gap, but in practice it is negligible at our problem scale (~640 variables).

Difference from Quality Minimizer:
    - Quality Minimizer has no incentive to compress beyond satisfying the memory budget
    - If a 38GB solution is quality-optimal under a 40GB budget, it stops
    - Resource Minimizer always pushes toward maximum compression 
    - Actively seeks the most aggressive quantization the quality budget allows
    - This makes Resource Minimizer more suitable for cost-sensitive production deployments and for fleet-wide optimization where smaller = cheaper

Key risk vs Quality Minimizer:
  - The quality constraint uses a proxy (Sum Omega_i x delta(b) x c(i,b)) that becomes less accurate under aggressive compression 
  - The solver pushes against this proxy boundary, amplifying any proxy inaccuracy Mitigated by the validation feedback loop's bisection on eps

Reference: CLADO [Deng et al., DAC 2025] demonstrated both formulations on ResNet architectures - confirms practical equivalence within ILP duality gap
"""
import pulp
import numpy as np
from ..models import SolverInput, SolverOutput, LayerDescriptor
from ..config import ResourceMinimizerConfig

class ILPResourceMinimizer:
    """Minimize resource usage subject to quality budget."""
    
    def __init__(self, config: ResourceMinimizerConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        return "ILPResourceMinimizer"
    
    def solve(self, input: SolverInput) -> SolverOutput:
        sensitivities = self._aggregate_sensitivities(input.layers)
        candidates = self._build_candidate_set(input)
        
        prob = pulp.LpProblem("QPE_ResourceMinimizer", pulp.LpMinimize)
        
        # Decision variables
        c = {}
        for i, layer in enumerate(input.layers):
            for b in candidates[i]:
                c[i, b] = pulp.LpVariable(f"c_{i}_{b}", cat=pulp.LpBinary)
        
        # Objective: minimize alpha x memory + beta x latency
        alpha = self.config.resource_weights.get("memory", 1.0)
        beta = self.config.resource_weights.get("latency", 0.0)
        
        prob += (
            alpha * pulp.lpSum(
                input.layers[i].memory_bytes[b] * c[i, b]
                for i in range(len(input.layers))
                for b in candidates[i]
            ) +
            beta * pulp.lpSum(
                input.layers[i].latency_us[b] * c[i, b]
                for i in range(len(input.layers))
                for b in candidates[i]
            )
        )
        
        # Constraint: quality budget (proxy)
        prob += pulp.lpSum(
            sensitivities[i] * self.config.error_scale[b] * c[i, b]
            for i in range(len(input.layers))
            for b in candidates[i]
        ) <= self.config.quality_budget_proxy
        
        # Constraint: one precision per layer
        for i in range(len(input.layers)):
            prob += pulp.lpSum(c[i, b] for b in candidates[i]) == 1
        
        # Constraint: physical memory cap (model must fit on GPU)
        memory_cap = self.config.hard_memory_cap_gb
        if memory_cap is None:
            memory_cap = input.gpu_memory_bytes / 1e9
        prob += pulp.lpSum(
            input.layers[i].memory_bytes[b] * c[i, b]
            for i in range(len(input.layers))
            for b in candidates[i]
        ) <= memory_cap * 1e9
        
        # Constraint: pinned layers
        for layer_name, pinned_prec in self.config.pinned_layers.items():
            i = self._layer_index(input.layers, layer_name)
            if i is not None and pinned_prec in candidates[i]:
                prob += c[i, pinned_prec] == 1
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.config.timeout_seconds))
        
        return self._build_output(prob, c, input, sensitivities, candidates)
    
    # _aggregate_sensitivities is identical to QualityMinimizer - reused via mixin.
    # - The same sensitivity computation feeds both formulations.
    # - The solver decides how to use it (objective vs. constraint), not the scorer.