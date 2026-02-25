from pydantic import BaseModel, ConfigDict
from typing import List 

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


class ServingBenchmark:
    """
    Wraps vLLM/TRT-LLM benchmark tools
    Does not re-implement measurement logic - uses the backend's own harness for correctness
    
    vLLM: vllm bench serve / vllm bench throughput
    TRT-LLM: trtllm-bench with pre-built engine
    GenAI-Perf: GPU telemetry correlation
    
    When measuring:
    - Lock GPU clocks during measurement
    - min_tokens = max_tokens = output_len to prevent variable-length skew
    - Run >= 500 requests for p99 stability
    - CUDA graph warm-up before measurement window
    - Report ITL excluding TTFT (GenAI-Perf convention, not LLMPerf)
    """
    
    def run(
        self, model_path: str, quant_config: dict, config: ServingBenchmarkConfig
    ) -> ServingBenchmarkResult:
        ...