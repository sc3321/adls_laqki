"""
Multi-objective Bayesian optimization for Pareto frontier exploration

Uses Ax Platform with BoTorch's qLogNEHVI (quasi-Log Noisy Expected Hypervolume Improvement) acquisition function 
    - state-of-the-art for multi-objective optimization with discrete parameters and noisy evaluations

This solver does not commit to either formulation 
    - It explores the full accuracy-vs-resource tradeoff space, returning a set of Pareto-optimal configurations 
    - Can use it to understand tradeoffs before committing to a deployment point, or to generate data for reports

Reference: Ax benchmarks show it is the best platform for mixed/discrete, multi-objective, constrained, and noisy problems [Meta, 2023]
"""
from ax.service.ax_client import AxClient, ObjectiveProperties
from ..config import ParetoExplorerConfig
from ..models import SolverInput, SolverOutput

class ParetoExplorer:
    """Explore Pareto frontier via multi-objective BO."""
    
    def __init__(self, config: ParetoExplorerConfig):
        self.config = config
    
    @property
    def name(self) -> str:
        return "ParetoExplorer"
    
    def solve(self, input: SolverInput) -> SolverOutput:
        """Returns the Pareto-optimal point with lowest quality cost."""
        frontier = self.explore(input)
        # Select the point closest to the quality axis (lowest latency/memory among configurations with lowest quality cost)
        best = min(frontier, key=lambda r: r.total_estimated_quality_cost)
        return best
    
    def explore(self, input: SolverInput) -> list[SolverOutput]:
        """Return full Pareto frontier as list of SolverOutput."""
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