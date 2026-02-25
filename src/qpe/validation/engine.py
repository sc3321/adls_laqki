from .config import ValidationConfig
from solver.models import SolverOutput, FeedbackSignal
class ValidationEngine:
    """
    Solver-agnostic validation. Consumes SolverOutput, produces FeedbackSignal
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self._fp16_baseline: dict | None = None
    
    def compute_fp16_baseline(self, model_id: str) -> None:
        """Compute and cache FP16 perplexity + benchmark scores"""
        ...
    
    def validate(
        self,
        model_id: str,
        solver_output: SolverOutput,
        stage: str = "screening",       # screening | full
    ) -> FeedbackSignal:
        """
        Apply quantization config, evaluate quality, return FeedbackSignal
        
        The FeedbackSignal contains:
        - Measured perplexity (absolute and relative to FP16)
        - Measured benchmark scores (Stage 2 only)
        - Per-layer KL divergence
        - Suggested corrections (which layers to pin, how to adjust quality budget)
        
        Suggestions are formulation-agnostic facts 
            - The Pipeline Orchestrator translates them into solver-specific constraint modifications
        """
        ...
    
    def _apply_quantization(
        self, model_id: str, assignment_dict: dict[str, str]
    ) -> str:
        """
        Apply per-layer mixed-precision quantization via LLM Compressor.
        Returns path to quantized model.
        
        LLM Compressor recipe supports non-uniform quantization: different layers can have different QuantizationModifier configs
        """
        ...