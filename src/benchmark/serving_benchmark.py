from pydantic import BaseModel, ConfigDict
from typing import List, Dict
from .models import ServingBenchmarkConfig, ServingBenchmarkResult
from .util import (
    _find_free_port,
    _build_trtllm_run_cmd,
    _build_vllm_bench_cmd,
    _build_vllm_server_cmd,
    _wait_for_vllm_server,
)

from .measurements import _get_gpu_mem_usage


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
        self, model_path: str, quant_config: dict, config: ServingBenchmarkConfig
    ) -> ServingBenchmarkResult:
        if config.backend == "vllm":
            return self._run_vllm(model_path, quant_config, config)
        elif config.backend == "trtllm":
            return self._run_trtllm(model_path, quant_config, config)
        else:
            raise ValueError(f"Unsupported backend : {config.backend}")

    def _run_vllm(
        self, model_path: str, quant_config: Dict, config: ServingBenchmarkConfig
    ) -> ServingBenchmarkResult:
        port = _find_free_port()
        server_process = None

        try:

            # Setup server
            import subprocess

            server_process = subprocess.Popen(
                args=_build_vllm_server_cmd(model_path, port, quant_config, config),
                stdout=subprocess.PIPE,  # TODO : probably redirect this if submitting jobs
                stderr=subprocess.PIPE,
            )
            _wait_for_vllm_server(port)

            # Run warmup iterations
            warmup_result = self._run_vllm_bench_client(
                port=port,
                num_requests=config.warmup_requests,
                concurrency=4,
                output_length=config.output_length,
                input_distribution=config.input_distribution,
            )

            # Run benchmark at various concurrency levels
            best_result = None
            for concurrency in config.concurrency_levels:
                result = self._run_vllm_bench_client(
                    port=port,
                    num_requests=config.num_requests,
                    concurrency=concurrency,
                    output_length=config.output_length,
                    input_distribution=config.input_distribution,
                )
                if best_result is None or result.get(
                    "output_throughput", 0
                ) > best_result.get("output_throughput", 0):
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
    ) -> Dict:

        import subprocess

        result = subprocess.run(
            _build_vllm_bench_cmd(
                num_requests,
                concurrency,
                output_length,
                input_distribution,
                url=f"http://localhost:{port}",
            ),
            capture_output=True,
            text=True,
            timeout=800,
        )

        if result.returncode != 0:
            # TODO : handle this shit better
            return {}

        import json

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
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
        self, model_path: str, quant_config: Dict, config: ServingBenchmarkConfig
    ) -> ServingBenchmarkResult:
        """
        Expects model_path to be an engine directory (already built by the ConfigurationExporter)
        """

        import tempfile
        import json
        import subprocess

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            bench_config = {
                "max_num_tokens": config.output_length,
                "num_requests": config.num_requests,
            }
            json.dump(bench_config, tmp_file)
            config_path = tmp_file.name

        result = subprocess.run(
            _build_trtllm_run_cmd(model_path, config),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"trtllm-bench failed w/ code {result.returncode} \n\n {result.stdout} "
            )

        data = json.loads(result.stdout)

        return ServingBenchmarkResult(
            ttft_p50_ms=data.get("ttft_p50_ms", 0.0),
            ttft_p99_ms=data.get("ttft_p99_ms", 0.0),
            tpot_p50_ms=data.get("tpot_p50_ms", 0.0),
            tpot_p99_ms=data.get("tpot_p99_ms", 0.0),
            output_token_throughput_tps=data.get("tokens_per_second", 0.0),
            peak_gpu_memory_gb=data.get("peak_gpu_memory_gb", 0.0),
        )
