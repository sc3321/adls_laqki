from typing import Protocol, runtime_checkable

@runtime_checkable
class QuantizationSolver(Protocol):
    """
    Contract for all quantization optimization engines.
    Implementations receive a SolverInput (containing per-layer sensitivity scores and hardware measurements) and return a SolverOutput (containing per-layer precision assignments and aggregate metrics).
    """
    from .models import SolverInput, SolverOutput
    
    def solve(self, input: SolverInput) -> SolverOutput:
        """Produce a per-layer precision assignment."""
        ...
    
    @property
    def name(self) -> str:
        """Human-readable solver name for logging and diagnostics."""
        ...

class SolverFactory:
    """
    Constructs solver instances from configuration
    
    Pipeline Orchestrator receives a QuantizationSolver (the protocol) and never imports 
    concrete implementations directly
    Adding a new solver requires only: (1) implementing the class (2) registering it here
    """
    from .config import SolverConfig
    
    _registry: dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, solver_class: type) -> None:
        """Register a solver implementation"""
        if not isinstance(solver_class, type) or not issubclass(solver_class, QuantizationSolver):
            raise TypeError(f"{solver_class} does not implement QuantizationSolver protocol")
        cls._registry[name] = solver_class
    
    @classmethod
    def create(cls, config: SolverConfig) -> QuantizationSolver:
        """Instantiate a solver from config"""
        if config.solver_name not in cls._registry:
            raise ValueError(
                f"Unknown solver '{config.solver_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[config.solver_name](config)