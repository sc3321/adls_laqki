import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..solver.models import SolverOutput
from .models import ExportResult

log = logging.getLogger(__name__)


class ConfigurationExporter:
    """
    Exports solver output to backend-specific quantization configurations.

    Each backend has different configuration semantics:

    vLLM: Non-uniform quantization via LLM Compressor's QuantizationModifier
    with per-layer recipes. Output: compressed HuggingFace checkpoint.
    Kernel auto-selection: Marlin for W4A16, DeepGEMM for FP8 on SM90+.

    TensorRT-LLM: Per-layer config in ModelOpt checkpoint format.
    JSON mapping: {"**/layers.0/q_proj": {"quant_algo": "FP8"}, ...}
    Output: ModelOpt checkpoint + trtllm-build command.

    llama.cpp: Per-tensor quant type in GGUF metadata.
    Maps FP16->F16, FP8->Q8_0, INT4->Q4_K_M.
    Uses imatrix from CalibrationDataManager for importance-weighted
    quantization.
    """

    QPE_VERSION = "2.0"

    def export(
        self,
        solver_output: SolverOutput,
        model_id: str,
        target: str,
        output_dir: str,
        importance_matrix: dict[str, np.ndarray] | None = None,
    ) -> ExportResult:
        """
        Export the quantization assignment to the target backend format.

        Args:
            solver_output: The solver's precision assignment.
            model_id: HuggingFace model ID or local path.
            target: Backend identifier (vllm | trtllm | llama_cpp).
            output_dir: Directory for exported artifacts.
            importance_matrix: Optional imatrix for llama.cpp export.

        Returns:
            ExportResult with output path, launch command, and metadata.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        assignment_dict = solver_output.to_assignment_dict()
        assignment_hash = hashlib.sha256(
            json.dumps(assignment_dict, sort_keys=True).encode()
        ).hexdigest()[:16]

        precision_dist: dict[str, int] = {}
        for prec in assignment_dict.values():
            precision_dist[prec] = precision_dist.get(prec, 0) + 1

        if target == "vllm":
            return self._export_vllm(
                solver_output, model_id, output_path,
                assignment_dict, assignment_hash, precision_dist,
            )
        elif target == "trtllm":
            return self._export_trtllm(
                solver_output, model_id, output_path,
                assignment_dict, assignment_hash, precision_dist,
            )
        elif target == "llama_cpp":
            return self._export_llama_cpp(
                solver_output, model_id, output_path,
                assignment_dict, assignment_hash, precision_dist,
                importance_matrix,
            )
        else:
            raise ValueError(
                f"Unknown export target '{target}'. "
                f"Supported: vllm, trtllm, llama_cpp"
            )

    def _export_vllm(
        self,
        solver_output: SolverOutput,
        model_id: str,
        output_path: Path,
        assignment_dict: dict[str, str],
        assignment_hash: str,
        precision_dist: dict[str, int],
    ) -> ExportResult:
        """
        Generate LLM Compressor recipe for vLLM serving.

        Creates a per-layer QuantizationModifier config that maps each
        layer to its assigned precision. vLLM's LLM Compressor applies
        the recipe to produce a compressed HuggingFace checkpoint.

        Precision mapping:
          FP16    -> no modifier (keep original weights)
          W8A8_FP8 -> QuantizationModifier(scheme="FP8", targets=[layer])
          W8A8_INT8 -> QuantizationModifier(scheme="W8A8", targets=[layer])
          W4A16   -> QuantizationModifier(scheme="W4A16", group_size=128)
        """
        precision_to_scheme = {
            "FP16": None,
            "W8A8_FP8": "FP8",
            "W8A8_INT8": "W8A8",
            "W4A16": "W4A16",
        }

        modifiers = []
        for layer_name, precision in assignment_dict.items():
            scheme = precision_to_scheme.get(precision)
            if scheme is not None:
                modifiers.append({
                    "QuantizationModifier": {
                        "scheme": scheme,
                        "targets": [layer_name],
                        "group_size": 128 if precision == "W4A16" else None,
                    }
                })

        recipe = {"modifiers": modifiers}
        recipe_path = output_path / "recipe.yaml"
        recipe_path.write_text(
            json.dumps(recipe, indent=2), encoding="utf-8"
        )

        compressed_path = str(output_path / "compressed_model")
        launch_cmd = (
            f"vllm serve {compressed_path} "
            f"--quantization compressed-tensors"
        )

        log.info("vLLM recipe written to %s (%d modifiers)", recipe_path, len(modifiers))

        return ExportResult(
            target="vllm",
            output_path=str(output_path),
            model_id=model_id,
            assignment_hash=assignment_hash,
            average_bitwidth=solver_output.average_bitwidth,
            total_weight_size_gb=solver_output.total_memory_bytes / 1e9,
            precision_distribution=precision_dist,
            kv_cache_dtype=solver_output.kv_cache_dtype,
            launch_command=launch_cmd,
            requires_packages=["vllm>=0.5", "llmcompressor>=0.1"],
            backend_config={"recipe_path": str(recipe_path)},
            solver_name=solver_output.solver_name,
            formulation_used=solver_output.formulation_used,
            qpe_version=self.QPE_VERSION,
            export_timestamp=datetime.now(timezone.utc).isoformat(),
            solver_output_hash=assignment_hash,
        )

    def _export_trtllm(
        self,
        solver_output: SolverOutput,
        model_id: str,
        output_path: Path,
        assignment_dict: dict[str, str],
        assignment_hash: str,
        precision_dist: dict[str, int],
    ) -> ExportResult:
        """
        Generate TensorRT-LLM ModelOpt checkpoint config.

        Creates a JSON mapping from layer glob patterns to quant_algo
        identifiers that ModelOpt understands.

        Precision mapping:
          FP16     -> null (no quantization)
          W8A8_FP8 -> "FP8"
          W8A8_INT8 -> "W8A8_SQ_PER_CHANNEL"
          W4A16    -> "W4A16_AWQ"
        """
        precision_to_algo = {
            "FP16": None,
            "W8A8_FP8": "FP8",
            "W8A8_INT8": "W8A8_SQ_PER_CHANNEL",
            "W4A16": "W4A16_AWQ",
        }

        quant_config = {}
        for layer_name, precision in assignment_dict.items():
            algo = precision_to_algo.get(precision)
            glob_pattern = f"**/{layer_name.replace('.', '/')}"
            quant_config[glob_pattern] = {"quant_algo": algo}

        config_path = output_path / "modelopt_config.json"
        config_path.write_text(
            json.dumps(quant_config, indent=2), encoding="utf-8"
        )

        engine_dir = str(output_path / "trtllm_engine")
        build_cmd = (
            f"trtllm-build --model_dir {model_id} "
            f"--quant_config {config_path} "
            f"--output_dir {engine_dir}"
        )

        log.info("TRT-LLM config written to %s", config_path)

        return ExportResult(
            target="trtllm",
            output_path=str(output_path),
            model_id=model_id,
            assignment_hash=assignment_hash,
            average_bitwidth=solver_output.average_bitwidth,
            total_weight_size_gb=solver_output.total_memory_bytes / 1e9,
            precision_distribution=precision_dist,
            kv_cache_dtype=solver_output.kv_cache_dtype,
            launch_command=build_cmd,
            requires_packages=["tensorrt-llm>=0.9", "nvidia-modelopt>=0.11"],
            backend_config={"config_path": str(config_path), "engine_dir": engine_dir},
            solver_name=solver_output.solver_name,
            formulation_used=solver_output.formulation_used,
            qpe_version=self.QPE_VERSION,
            export_timestamp=datetime.now(timezone.utc).isoformat(),
            solver_output_hash=assignment_hash,
        )

    def _export_llama_cpp(
        self,
        solver_output: SolverOutput,
        model_id: str,
        output_path: Path,
        assignment_dict: dict[str, str],
        assignment_hash: str,
        precision_dist: dict[str, int],
        importance_matrix: dict[str, np.ndarray] | None = None,
    ) -> ExportResult:
        """
        Generate llama.cpp GGUF metadata mapping.

        Maps QPE precision assignments to GGUF quant types.
        Optionally integrates importance matrix for weighted quantization.

        Precision mapping:
          FP16     -> F16
          W8A8_FP8 -> Q8_0
          W8A8_INT8 -> Q8_0
          W4A16    -> Q4_K_M
        """
        precision_to_gguf = {
            "FP16": "F16",
            "W8A8_FP8": "Q8_0",
            "W8A8_INT8": "Q8_0",
            "W4A16": "Q4_K_M",
        }

        tensor_types = {}
        for layer_name, precision in assignment_dict.items():
            gguf_type = precision_to_gguf.get(precision, "F16")
            tensor_types[layer_name] = gguf_type

        metadata = {"tensor_quant_types": tensor_types}
        metadata_path = output_path / "gguf_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        imatrix_flag = ""
        if importance_matrix is not None:
            imatrix_path = output_path / "importance_matrix.dat"
            np.savez_compressed(str(imatrix_path), **importance_matrix)
            imatrix_flag = f" --imatrix {imatrix_path}"

        gguf_output = str(output_path / "model.gguf")
        convert_cmd = (
            f"python llama.cpp/convert_hf_to_gguf.py {model_id} "
            f"--outfile {gguf_output} "
            f"--quant-metadata {metadata_path}"
            f"{imatrix_flag}"
        )

        log.info("llama.cpp metadata written to %s", metadata_path)

        return ExportResult(
            target="llama_cpp",
            output_path=str(output_path),
            model_id=model_id,
            assignment_hash=assignment_hash,
            average_bitwidth=solver_output.average_bitwidth,
            total_weight_size_gb=solver_output.total_memory_bytes / 1e9,
            precision_distribution=precision_dist,
            kv_cache_dtype=solver_output.kv_cache_dtype,
            launch_command=convert_cmd,
            requires_packages=["llama-cpp-python>=0.2"],
            backend_config={
                "metadata_path": str(metadata_path),
                "gguf_output": gguf_output,
            },
            solver_name=solver_output.solver_name,
            formulation_used=solver_output.formulation_used,
            qpe_version=self.QPE_VERSION,
            export_timestamp=datetime.now(timezone.utc).isoformat(),
            solver_output_hash=assignment_hash,
        )
