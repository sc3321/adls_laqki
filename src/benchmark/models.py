from pydantic import BaseModel, ConfigDict
from typing import List, Dict
class ServingBenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: str = "vllm"
    num_requests: int = 500
    concurrency_levels: List[int] = [1, 8, 32, 128]
    input_distribution: str = "sharegpt"  
    output_length: int = 256
    warmup_requests: int = 50


class ServingBenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float 
    tpot_p99_ms: float
    output_token_throughput_tps: float
    peak_gpu_memory_gb: float