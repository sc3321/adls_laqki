"""
Solver package initialization : registers all built-in solvers.

To add a new solver:
  1. Implement the QuantizationSolver protocol in a new module
  2. Create a SolverConfig subclass with your parameters
  3. Register it here with SolverFactory.register()
  
"""
from .protocol import SolverFactory
from .models import (
    SolverInput, 
    SolverOutput, 
    LayerAssignment, 
    LayerDescriptor
)
from .config import SolverConfig

__all__ = [
    "SolverFactory",
    "SolverInput",
    "SolverOutput",
    "SolverConfig",
    "LayerAssignment", 
    "LayerDescriptor"
]



try:
    from .optimizers.quality import ILPQualityMinimizer
    from .optimizers.resources import ILPResourceMinimizer
    from .optimizers.pareto_explorer import ParetoExplorer

    SolverFactory.register("quality_minimizer", ILPQualityMinimizer)
    SolverFactory.register("resource_minimizer", ILPResourceMinimizer)
    SolverFactory.register("pareto_explorer", ParetoExplorer)
except ImportError:
    pass  # pulp not installed; solvers unavailable but package is importable
