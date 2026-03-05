from typing import List, Dict, Any
import torch 
import torch.nn as nn 

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


def _build_vllm_server_cmd(
    model_path : str, 
    port : int, 
    quant_config : Dict, 
    config : Any
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
    config : Any,
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

def _make_benchmark_input(
    batch_size: int,
    sequence_length: int,
    module: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Benchmark input for a layer.

    We keep inputs floating-point. For CUDA profiling we use fp16 inputs
    to match layer.half() profiling and avoid dtype mismatch.
    """
    if isinstance(module, nn.Linear):
        in_features = module.in_features
    else:
        in_features = next(module.parameters()).shape[-1]

    # Always use fp16 on CUDA for profiling (matches layer.half()).
    if device.type == "cuda":
        inp_dtype = torch.float16
    else:
        # CPU fallback: match module dtype if possible
        try:
            inp_dtype = next(module.parameters()).dtype
        except StopIteration:
            inp_dtype = torch.float32

    return torch.randn(
        batch_size,
        sequence_length,
        in_features,
        dtype=inp_dtype,
        device=device,
    )
