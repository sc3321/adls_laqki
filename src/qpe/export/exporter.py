from solver.models import SolverOutput
import numpy as np
class ConfigurationExporter:
    """
    Each backend has different configuration semantics:
    
    vLLM: Non-uniform quantization via LLM Compressor's QuantizationModifier with per-layer recipes. 
    Output: compressed HuggingFace checkpoint.
    Kernel auto-selection: Marlin for W4A16, DeepGEMM for FP8 on SM90+.
    
    TensorRT-LLM: Per-layer config in ModelOpt checkpoint format.
    JSON mapping: {"**/layers.0/q_proj": {"quant_algo": "FP8"}, ...}
    Output: ModelOpt checkpoint + trtllm-build command.
    
    llama.cpp: Per-tensor quant type in GGUF metadata.
    Maps FP16->F16, FP8->Q8_0, INT4->Q4_K_M.
    Uses imatrix from CalibrationDataManager for importance-weighted quantization.
    """
    from .models import ExportResult
    
    def export(
        self,
        solver_output: SolverOutput,
        model_id: str,
        target: str,            # vllm | trtllm | llama_cpp
        output_dir: str,
        importance_matrix: dict[str, np.ndarray] | None = None,
    ) -> ExportResult:
        """Returns ExportResult with output path and launch command."""
        ...