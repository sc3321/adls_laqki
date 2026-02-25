from pydantic import BaseModel
from typing import Optional, Dict, List

class GPUSpec(BaseModel) :
    """
        HW Spec for a given GPU Architecture
    """
    name: str
    compute_capability: tuple[int, int]      # e.g., (9, 0) for H100
    memory_gb: float
    memory_bandwidth_tb_s: float             # HBM bandwidth

    # Precision support flags (determined by SM ver)
    supports_fp8 : bool
    supports_fp4 : bool
    supports_int8_tensor_core : bool
    supports_in4_tensor_core : bool

    # Peak throuput by prec 
    peak_fp16_tflops: float
    peak_fp8_tflops: Optional[float] 
    peak_int8_tops: float
    peak_int4_tops: Optional[float] 

    avilable_kernels : Dict[str, List[str]]
    # Example struct :
    # {
    #   "W4A16"    : ["marlin", "exllamav2", "cutlass"],
    #   "W8A8_FP8" : ["deepgemm", "cublas", "cutlass"],
    #   "W8A8_INT8": ["cutlass", "cublas"]
    # }