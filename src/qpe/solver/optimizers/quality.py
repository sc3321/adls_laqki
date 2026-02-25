"""
ILP solver that minimizes total quality degradation subject to resource budgets.

Formulation:
    
    minimize    Sum_i[ Sum_b [Omega_i x delta(b) x c(i,b)]]
    
    subject to:
        Sum_i[ Sum_b [M(i,b) x c(i,b)]]  <=  M_budget                               (memory budget)
        Sum_i[ Sum_b [L(i,b) x c(i,b)]]  <=  L_budget                               (latency budget, optional)
        Sum_b c(i,b)                      =  1        for all i                     (exactly one precision per layer)
        c(i,b)                            =  0        if precision b unavailable    (hardware filter)
        c(i,b)                            =  1        if layer i pinned to b        (user overrides)

    where:
        c(i,b) in [0, 1]     =    binary: layer i assigned precision b
        Omega_i              =    composite sensitivity score (weighted combination of raw signals)
        delta(b)             =    per-precision error scale factor (FP16=0, FP8=0.05, INT8=0.1, INT4=1.0)
        M(i,b)               =    memory consumption of layer i at precision b (measured)
        L(i,b)               =    decode latency of layer i at precision b (measured)

Solver: 
    -> PuLP with CBC backend (open-source). 
    -> Problem size for 80-layer model with 4 precisions: 320 binary variables, ~85 constraints. Solves in <1 second.

Reference: HAWQ-V3 [Yao et al., ICML 2021] - proved ILP with Hessian trace is 120x faster than HAQ's DDPG while matching accuracy on ResNet-50.
"""
import pulp
import numpy as np
from ..config import QualityMinimizerConfig
from ..models import SolverInput, SolverOutput, LayerDescriptor

class ILPQualityMinimizer:
    """Minimize quality loss subject to hardware budgets."""
    
    def __init__(self, config: QualityMinimizerConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        return "ILPQualityMinimizer"
    
    def solve(self, input: SolverInput) -> SolverOutput:
        # Compute composite sensitivity per layer
        sensitivities = self._aggregate_sensitivities(input.layers)
        
        # Build candidate set per layer (filter by GPU + availability)
        candidates = self._build_candidate_set(input)
        
        # Formulate ILP
        prob = pulp.LpProblem("QPE_QualityMinimizer", pulp.LpMinimize)
        
        # Decision variables
        c = {}  # c[i, b] = 1 if layer i gets precision b
        for i, layer in enumerate(input.layers):
            for b in candidates[i]:
                c[i, b] = pulp.LpVariable(f"c_{i}_{b}", cat=pulp.LpBinary)
        
        # Objective: minimize Sum Omega_i x delta(b) x c[i,b]
        prob += pulp.lpSum(
            sensitivities[i] * self.config.error_scale[b] * c[i, b]
            for i in range(len(input.layers))
            for b in candidates[i]
        )
        
        # Constraint: one precision per layer
        for i in range(len(input.layers)):
            prob += pulp.lpSum(c[i, b] for b in candidates[i]) == 1
        
        # Constraint: mem budget
        prob += pulp.lpSum(
            input.layers[i].memory_bytes[b] * c[i, b]
            for i in range(len(input.layers))
            for b in candidates[i]
        ) <= self.config.memory_budget_gb * 1e9
        
        # Constraint: latency budget (optional)
        if self.config.latency_budget_us is not None:
            prob += pulp.lpSum(
                input.layers[i].latency_us[b] * c[i, b]
                for i in range(len(input.layers))
                for b in candidates[i]
            ) <= self.config.latency_budget_us
        
        # Constraint: pinned layers
        for layer_name, pinned_prec in self.config.pinned_layers.items():
            i = self._layer_index(input.layers, layer_name)
            if i is not None and pinned_prec in candidates[i]:
                prob += c[i, pinned_prec] == 1
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.config.timeout_seconds))
        
        # Extract solution
        return self._build_output(prob, c, input, sensitivities, candidates)
    
    def _aggregate_sensitivities(self, layers: list[LayerDescriptor]) -> np.ndarray:
        """
        Combine raw sensitivity signals into a single composite score per layer.
        
        Uses weighted geometric mean after min-max normalization.
        Geometric mean is scale-invariant - used because signals span different scales (Hessian trace ~1e-3 to 1e+3, kurtosis ~1 to 1800+)
        """
        signals = np.zeros((len(layers), len(self.config.sensitivity_weights)))
        weights = np.array(list(self.config.sensitivity_weights.values()))
        
        for j, (signal_name, _) in enumerate(self.config.sensitivity_weights.items()):
            raw = np.array([getattr(layer, signal_name) for layer in layers])
            # Min-max normalize to [eps, 1.0] (eps avoids log(0) in geometric mean)
            mn, mx = raw.min(), raw.max()
            if mx > mn:
                signals[:, j] = (raw - mn) / (mx - mn) * 0.99 + 0.01
            else:
                signals[:, j] = 0.5
        
        # Weighted geometric mean: exp(Sum w_j x log(s_j))
        log_signals = np.log(signals)
        composite = np.exp(log_signals @ weights / weights.sum())
        return composite
    
    # ... helper methods _build_candidate_set, _layer_index, _build_output ...