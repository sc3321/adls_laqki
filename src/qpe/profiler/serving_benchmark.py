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

def _find_free_port() -> int : 
    """Get available tcp port for vLLM server""" 
    import socket 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s :
        s.bind(("", 0))
        return s.getsockname()[1]

def _wait_for_vllm_server(port: int, timeout: int = 300) -> None:
    """
    Poll the vLLM server's health endpoint until it responds.
    """
    import urllib.request
    import urllib.error
    import time

    start = time.time()
    # TODO : update url paths for production
    url = f"http://localhost:{port}/health"
    
    while time.time() - start < timeout:
        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.status == 200:
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    
    raise TimeoutError(
        f"vLLM server did not become healthy within {timeout}s. "
    )


def _get_gpu_mem_usage() -> float : 
    import pynvml
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 3)
    except Exception:
        return 0.0

def _build_vllm_server_cmd(
    model_path : str, 
    port : int, 
    quant_config : Dict, 
    config : "ServingBenchmarkConfig"
) -> List[str] : 
    server_cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--max-model-len", str(config.output_length * 4),  # Headroom
        "--disable-log-requests",
    ]

    if "quantization" in quant_config:
        server_cmd.extend(["--quantization", quant_config["quantization"]])
    
    return server_cmd 

def _build_vllm_bench_cmd(
    num_requests : int,
    concurrency : int, 
    output_length : int,
    input_distribution : str,
    *,
    url : str,
    model_name : str = "default",  # Model name on the server
    output_format : str = "json"
) -> List[str] : 
    """
    The --min-tokens and --max-tokens are set equal to force fixed output length 
        -> this is done to prevent variable completion length from skewing latency statistics
    """
    cmd = [
        "vllm", "bench", "serve",
        "--base-url", url,
        "--model", model_name,
        "--num-requests", str(num_requests),
        "--concurrency", str(concurrency),
        "--min-tokens", str(output_length),
        "--max-tokens", str(output_length),
        "--output-format", output_format,
    ]
    
    if input_distribution == "sharegpt":
        cmd.extend(["--dataset", "sharegpt"])
    
    return cmd

def _build_trtllm_run_cmd(
    model_path : str,
    config : "ServingBenchmarkConfig",
    *,
    benchmark_type : str = "throughput",
    output_format : str = "json"
) -> List[str] :
    return [
        "trtllm-bench",
        "--engine_dir", model_path,
        "--benchmark_type", benchmark_type,
        "--num_requests", str(config.num_requests),
        "--max_output_len", str(config.output_length),
        "--output_format", output_format,
    ]


class ServingBenchmark:
    """
    Wraps vLLM/TRT-LLM benchmark tools
    
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
        self, 
        model_path: str, 
        quant_config: dict, 
        config: ServingBenchmarkConfig
    ) -> ServingBenchmarkResult:
        if config.backend == "vllm" :
            return self._run_vllm(model_path, quant_config, config)
        elif config.backend == "trtllm" : 
            return self._run_trtllm(model_path, quant_config, config)
        else : 
            raise ValueError(f"Unsupported backend : {config.backend}")

    def _run_vllm(
        self, 
        model_path : str,
        quant_config : Dict, 
        config : ServingBenchmarkConfig
    ) -> ServingBenchmarkResult : 
        port = _find_free_port()
        server_process = None 

        try : 

            # Setup server
            import subprocess
            server_process = subprocess.Popen(
                args = _build_vllm_server_cmd(model_path, port, quant_config, config),
                stdout = subprocess.PIPE, # TODO : probably redirect this if submitting jobs
                stderr = subprocess.PIPE 
            )
            _wait_for_vllm_server(port)

            # Run warmup iterations
            warmup_result = self._run_vllm_bench_client(
                port = port,
                num_requests = config.warmup_requests,
                concurrency = 4,
                output_length = config.output_length,
                input_distribution = config.input_distribution
            )

            # Run benchmark at various concurrency levels
            best_result = None 
            for concurrency in config.concurrency_levels : 
                result = self._run_vllm_bench_client(
                    port=port,
                    num_requests=config.num_requests,
                    concurrency=concurrency,
                    output_length=config.output_length,
                    input_distribution=config.input_distribution,
                )
                if best_result is None or result.get("output_throughput", 0) > best_result.get("output_throughput", 0):
                    best_result = result
            
            peak_memory_gb = _get_gpu_mem_usage()
                        
            return ServingBenchmarkResult(
                ttft_p50_ms=best_result.get("ttft_p50", 0.0),
                ttft_p99_ms=best_result.get("ttft_p99", 0.0),
                tpot_p50_ms=best_result.get("tpot_p50", 0.0),
                tpot_p99_ms=best_result.get("tpot_p99", 0.0),
                output_token_throughput_tps=best_result.get("output_throughput", 0.0),
                peak_gpu_memory_gb=peak_memory_gb,
            )


        finally:
            #  Kill server
            if server_process is not None:
                server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_process.kill()


    def _run_vllm_bench_client(
        self,
        port: int,
        num_requests: int,
        concurrency: int,
        output_length: int,
        input_distribution: str,
    ) -> Dict : 

        import subprocess 
        result = subprocess.run(
            _build_vllm_bench_cmd(
                num_requests, concurrency, output_length, input_distribution,
                url = f"http://localhost:{port}"
            ),
            capture_output = True,
            text = True,
            timeout = 800
        )

        if result.returncode != 0 : 
            # TODO : handle this shit better
            return {}

        import json
        try : 
            return json.loads(result.stdout)
        except json.JSONDecodeError : 
            # try and find the json output in case there are logs present b4 json
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return {}

    def _run_trtllm(
        self,
        model_path : str,
        quant_config : Dict,
        config : ServingBenchmarkConfig
    ) -> ServingBenchmarkResult :
        """
        Expects model_path to be an engine directory (already built by the ConfigurationExporter)
        """

        import tempfile
        import json
        import subprocess 

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete = False) as tmp_file : 
            bench_config = {
                "max_num_tokens" : config.output_length,
                "num_requests" : config.num_requests 
            }
            json.dump(bench_config, tmp_file)
            config_path = tmp_file.name

        result = subprocess.run(
            _build_trtllm_run_cmd(model_path, config),
            capture_output = True,
            text = True,
            timeout = 1800
        )
        if result.returncode != 0 :
            raise RuntimeError(f"trtllm-bench failed w/ code {result.returncode} \n\n {result.stdout} ")
        
        data = json.loads(result.stdout)
        
        return ServingBenchmarkResult(
            ttft_p50_ms=data.get("ttft_p50_ms", 0.0),
            ttft_p99_ms=data.get("ttft_p99_ms", 0.0),
            tpot_p50_ms=data.get("tpot_p50_ms", 0.0),
            tpot_p99_ms=data.get("tpot_p99_ms", 0.0),
            output_token_throughput_tps=data.get("tokens_per_second", 0.0),
            peak_gpu_memory_gb=data.get("peak_gpu_memory_gb", 0.0),
        )