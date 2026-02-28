from pydantic import BaseModel
from typing import Optional, Dict, List

class GPUSpec(BaseModel) :
    """
        HW Spec for a given GPU Architecture
    """
    name: str
    compute_capability: tuple[int, int]      # (9, 0) for H100
    memory_gb: float
    memory_bandwidth_tb_s: float             # HBM bandwidth

    # Precision support flags (determined by SM ver)
    supports_fp8 : bool
    supports_fp4 : bool
    supports_int8_tensor_core: bool
    supports_int4_tensor_core: bool = False

    # Peak throughput by precision
    peak_fp16_tflops: float
    peak_fp8_tflops: Optional[float] = None
    peak_int8_tops: float
    peak_int4_tops: Optional[float] = None

    available_kernels: Dict[str, List[str]]
    # Example struct :
    # {
    #   "W4A16"    : ["marlin", "exllamav2", "cutlass"],
    #   "W8A8_FP8" : ["deepgemm", "cublas", "cutlass"],
    #   "W8A8_INT8": ["cutlass", "cublas"]
    # }


GPU_REGISTRY: dict[str, GPUSpec] = {
    "A100_80GB": GPUSpec(
        name="NVIDIA A100 80GB SXM", compute_capability=(8, 0),
        memory_gb=80.0, memory_bandwidth_tb_s=2.0,
        supports_fp8=False, supports_fp4=False, supports_int8_tensor_core=True,
        peak_fp16_tflops=312.0, peak_fp8_tflops=None, peak_int8_tops=624.0,
        available_kernels={
            "W4A16": ["marlin", "exllamav2", "autogptq"],
            "W8A8_INT8": ["cutlass", "cublas"],
            "FP16": ["cublas", "cutlass"],
        },
    ),
    "H100_SXM": GPUSpec(
        name="NVIDIA H100 80GB SXM", compute_capability=(9, 0),
        memory_gb=80.0, memory_bandwidth_tb_s=3.35,
        supports_fp8=True, supports_fp4=False, supports_int8_tensor_core=True,
        peak_fp16_tflops=989.0, peak_fp8_tflops=1979.0, peak_int8_tops=1979.0,
        available_kernels={
            "W4A16": ["marlin", "exllamav2", "cutlass"],
            "W8A8_FP8": ["deepgemm", "cublas_fp8", "cutlass"],
            "W8A8_INT8": ["cutlass", "cublas"],
            "FP16": ["cublas", "cutlass"],
        },
    ),
    "RTX_4090": GPUSpec(
        name="NVIDIA GeForce RTX 4090", compute_capability=(8, 9),
        memory_gb=24.0, memory_bandwidth_tb_s=1.008,
        supports_fp8=True, supports_fp4=False, supports_int8_tensor_core=True,
        peak_fp16_tflops=165.0, peak_fp8_tflops=330.0, peak_int8_tops=660.0,
        available_kernels={
            "W4A16": ["marlin", "exllamav2"],
            "W8A8_FP8": ["cublas_fp8"],
            "W8A8_INT8": ["cutlass", "cublas"],
            "FP16": ["cublas"],
        },
    ),
    "T4": GPUSpec(
        name="NVIDIA Tesla T4", compute_capability=(7, 5),
        memory_gb=16.0, memory_bandwidth_tb_s=0.32,
        supports_fp8=False, supports_fp4=False, supports_int8_tensor_core=True,
        supports_int4_tensor_core=True,
        peak_fp16_tflops=65.0, peak_int8_tops=130.0, peak_int4_tops=260.0,
        available_kernels={
            "W4A16": ["marlin", "exllamav2", "autogptq"],
            "W8A8_INT8": ["cutlass", "cublas"],
            "FP16": ["cublas", "cutlass"],
        },
    ),
}