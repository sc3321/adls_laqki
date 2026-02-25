from __future__ import annotations
from enum import Enum

class Precision(str, Enum) :
    """
    Each value corresponds to a specific weight format and the dominant kernel family used for inference. 
    The naming convention encodes weight format and activation format: W{w}A{a}_{type}
    """
    FP16      = "FP16"        # No quantization. Baseline.
    W8A8_FP8  = "W8A8_FP8"    # FP8 e4m3 weights + activations. Requires SM >= 8.9
    W8A8_INT8 = "W8A8_INT8"   # INT8 SmoothQuant weights + activations. SM >= 8.0.
    W4A16     = "W4A16"       # INT4 weight-only (AWQ/GPTQ), FP16 activations. SM >= 8.0.
