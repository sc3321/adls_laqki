from abc import ABC, abstractmethod

from .models import SolverInput, SolverOutput
from .config import SolverConfig


class QuantizationSolver(ABC):
    """
    Contract for all quantization optimization engines.
    Implementations receive a SolverInput (containing per-layer sensitivity
    scores and hardware measurements) and return a SolverOutput (containing
    per-layer precision assignments and aggregate metrics).
    """

    @abstractmethod
    def solve(self, input: SolverInput) -> SolverOutput:
        """Produce a per-layer precision assignment."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name for logging and diagnostics."""
        ...


class SolverFactory:
    """Constructs solver instances from configuration."""

    _registry: dict[str, type[QuantizationSolver]] = {}

    @classmethod
    def register(cls, name: str, solver_class: type) -> None:
        """Register a solver implementation."""
        if not isinstance(solver_class, type) or not issubclass(solver_class, QuantizationSolver):
            raise TypeError(f"{solver_class} does not inherit from QuantizationSolver")
        cls._registry[name] = solver_class

    @classmethod
    def create(cls, config: SolverConfig) -> QuantizationSolver:
        """Instantiate a solver from config."""
        if config.solver_name not in cls._registry:
            raise ValueError(
                f"Unknown solver '{config.solver_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[config.solver_name](config)
