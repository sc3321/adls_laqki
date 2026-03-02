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



import logging as _log

try:
    from .optimizers.quality import ILPQualityMinimizer
    SolverFactory.register("quality_minimizer", ILPQualityMinimizer)
except ImportError as e:
    if "pulp" in str(e).lower():
        _log.getLogger(__name__).warning("PuLP not installed; ILPQualityMinimizer unavailable")
    else:
        raise

try:
    from .optimizers.resources import ILPResourceMinimizer
    SolverFactory.register("resource_minimizer", ILPResourceMinimizer)
except ImportError as e:
    if "pulp" in str(e).lower():
        _log.getLogger(__name__).warning("PuLP not installed; ILPResourceMinimizer unavailable")
    else:
        raise

try:
    from .optimizers.pareto_explorer import ParetoExplorer
    SolverFactory.register("pareto_explorer", ParetoExplorer)
except ImportError as e:
    if "ax" in str(e).lower():
        _log.getLogger(__name__).warning("Ax Platform not installed; ParetoExplorer unavailable")
    else:
        raise
