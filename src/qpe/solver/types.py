"""
Solver-agnostic data types that form the contract between the optimization
engine and all other QPE components. These types are the ONLY interface
through which the solver communicates with the rest of the system.
"""
from __future__ import annotations

from enum import Enum



class Precision(str, Enum):
    """Candidate quantization precisions.

    Each value corresponds to a specific weight format and the dominant
    kernel family used for inference. The naming convention encodes
    weight format and activation format: W{w}A{a}_{type}.
    """

    FP16 = "FP16"           # No quantization. Baseline.
    W8A8_FP8 = "W8A8_FP8"  # FP8 e4m3 weights + activations. Requires SM >= 8.9.
    W8A8_INT8 = "W8A8_INT8" # INT8 SmoothQuant weights + activations. SM >= 8.0.
    W4A16 = "W4A16"         # INT4 weight-only (AWQ/GPTQ), FP16 activations. SM >= 8.0.


