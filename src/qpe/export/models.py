from pydantic import BaseModel, ConfigDict 

class ExportResult(BaseModel):
    """Output metadata from a successful export operation."""
    model_config = ConfigDict(frozen=True)
    
    target: str                             # vllm | trtllm | llama_cpp
    output_path: str                        # Path to the exported artifact
    model_id: str                           # Source model
    
    # What was exported
    assignment_hash: str                    # SHA256 of the layer -> precision mapping
    average_bitwidth: float
    total_weight_size_gb: float
    
    # Backend-specific launch info
    launch_command: str                     # e.g., vllm serve ./quantized --quantization mixed
    requires_packages: list[str]            # e.g., [vllm>=0.5, auto-awq>=0.2]
    
    # Provenance
    solver_name: str                        # Which solver produced the assignment
    formulation_used: str                   # quality_minimizing | resource_minimizing | pareto
    qpe_version: str                        # QPE package version for reproducibility