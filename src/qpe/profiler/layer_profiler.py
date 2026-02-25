from .gpu_specs import GPUSpec
import torch.nn as nn
from solver.types import Precision 
from typing import Dict 

class LayerProfiler:
    """
    Profiles individual model layers at multiple precisions and batch sizes.
    
    1. Lock GPU clocks to base frequency:
        nvidia-smi -lgc <base_clock>,<base_clock>
       Prevents thermal throttling from biasing measurements
       Important : unlocked clocks cause 5-15% latency variance between runs

    2 Allocate layer in isolation on GPU 
        - This avoids memory pressure interference from other layers competing for L2 cache and HBM bandwidth

    3 Generate representative input tensors matching expected dimensions
        - Use the actual (batch_size, seq_len, hidden_dim) shapes from the model
    4. Warmup: 
        - 50 iterations. 
        - Fills CUDA graph cache, JIT compiles kernels (torch.compile), and reaches thermal steady state

    5 Measure: 
        - 200 iterations with torchcudaEvent timing (not wall-clock)
        - CUDA event timing measures GPU-side execution time, excluding CPU-side overhead, Python GIL contention, and CUDA launch latency
    
    6 Record statistics: 
        - mean, median, p99, std
    
    7 Repeat for each (precision x batch_size) combination
    
    Design decision 
        (1) Isolated layer profiling vs full-model profiling:
            -> Profile layers in isolation because 
                (a) it is faster
                (b) total latency is well-approximated by sum of layer latencies for sequential transformer blocks
                (c) this isolates kernel performance from scheduling artifacts
    
    """
    
    def __init__(
        self,
        gpu_spec: GPUSpec,
        batch_sizes: list[int] = [1, 4, 16, 64],
        num_warmup: int = 50,
        num_measurements: int = 200,
        cache_dir: str = ".qpe_cache/profiles",
    ):
        ...
    
    def profile_all_layers(
        self,
        model: nn.Module,
        layer_names: list[str],
        target_batch_size: int = 1,
    ) -> dict[str, dict]:
        """
        Profile all layers, return resource data keyed by layer name
        
        Returns:
            <layer_name>: {
                memory_bytes: {precision: bytes},
                latency_us: {precision: microseconds},
                peak_memory_bytes: {precision: bytes},
                kernel_name: {precision: kernel_string},
                is_memory_bound: {precision: bool},
            }
        
        Data is merged into LayerDescriptor by Pipeline Orchestrator
        """
        ...
    
    def _lock_gpu_clocks(self) -> None:
        """nvidia-smi -lgc <base>,<base> - lock to base frequency."""
        ...
    
    def _unlock_gpu_clocks(self) -> None:
        """nvidia-smi -rgc - restore default clock management."""
        ...
    
    def _quantize_layer(self, layer: nn.Module, precision: Precision) -> nn.Module:
        """
        Quantize a single layer in isolation using torchao/AutoAWQ/AutoGPTQ.
            - FP8: torchao.quantize_(layer, float8_weight_only())
            - INT8 SmoothQuant: torchao.quantize_(layer, int8_dynamic_activation_int8_weight())
            - INT4 AWQ: AutoAWQ with group_size=128 on isolated layer
        """
        ...