import logging
import time
from typing import Dict, List

import numpy as np
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader

from .calibration.manager import CalibrationDataManager
from .calibration.models import CalibrationConfig
from .export import ConfigurationExporter, ExportResult
from .profiler import LayerProfiler, GPUSpec
from .profiler.gpu_specs import GPU_REGISTRY, detect_gpu
from .profiler.models import ModelProfileResult
from .scorer.base import SensitivityScorer
from .solver import (
    SolverConfig,
    SolverInput,
    SolverOutput,
    SolverFactory,
    LayerDescriptor,
)
from .solver.config import ResourceMinimizerConfig, ParetoExplorerConfig
from .solver.models import FeedbackSignal
from .solver.protocol import QuantizationSolver
from .utils.types import Precision
from .utils.model_utils import load_model, get_quantizable_layers
from .validation.config import ValidationConfig
from .validation.engine import ValidationEngine

log = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_id: str
    calibration: CalibrationConfig = CalibrationConfig()
    validation: ValidationConfig = ValidationConfig()
    solver: SolverConfig
    export_target: str = "vllm"
    output_dir: str = "./qpe_output"

    gpu_spec_name: str | None = None
    target_batch_size: int = 1

    max_feedback_iterations: int = 5
    wandb_project: str = "qpe"
    wandb_enabled: bool = True


class PipelineResult(BaseModel):
    """Complete output of a QPE pipeline run."""
    model_config = ConfigDict(frozen=True)

    solver_output: SolverOutput
    validation: FeedbackSignal
    export: ExportResult

    iterations_used: int
    total_wall_time_seconds: float
    scoring_time_seconds: float
    profiling_time_seconds: float
    solving_time_seconds: float
    validation_time_seconds: float

    pipeline_config: PipelineConfig
    warnings: list[str] = []


class Pipeline:
    """Orchestrates Score -> Profile -> Solve -> Validate -> Export with feedback."""

    def __init__(
        self,
        calibration_manager: CalibrationDataManager,
        scorer: SensitivityScorer,
        profiler: LayerProfiler,
        validator: ValidationEngine,
        exporter: ConfigurationExporter,
        config: PipelineConfig,
    ):
        self.calibration_manager = calibration_manager
        self.scorer = scorer
        self.profiler = profiler
        self.validator = validator
        self.exporter = exporter
        self.config = config
        self.solver : QuantizationSolver = SolverFactory.create(config.solver)
        self.gpu_spec = self._resolve_gpu_spec()

    #  GPU resolution

    def _resolve_gpu_spec(self) -> GPUSpec:
        """Resolve GPU spec from config name or auto-detect."""
        if self.config.gpu_spec_name is not None:
            if self.config.gpu_spec_name not in GPU_REGISTRY:
                raise ValueError(
                    f"Unknown gpu_spec_name '{self.config.gpu_spec_name}'. "
                    f"Available: {list(GPU_REGISTRY.keys())}"
                )
            return GPU_REGISTRY[self.config.gpu_spec_name]
        return detect_gpu()

    #  Top-level run: delegates to stage methods
    def run(self) -> PipelineResult:
        pipeline_start = time.perf_counter()
        warnings: list[str] = []

        model, dataloader = self._setup_stage()

        scoring_start = time.perf_counter()
        scorer_output = self._scoring_stage(model, dataloader)
        scoring_time = time.perf_counter() - scoring_start

        profiling_start = time.perf_counter()
        profiler_output = self._profiling_stage(model, scorer_output)
        profiling_time = time.perf_counter() - profiling_start

        solver_input = self._assembly_stage(scorer_output, profiler_output)

        self.validator.compute_fp16_baseline(self.config.model_id)

        if isinstance(self.config.solver, ResourceMinimizerConfig):
            initial_eps = self._estimate_initial_epsilon(
                solver_input,
                self.config.validation.max_perplexity_increase_pct,
            )
            new_config = self.config.solver.model_copy(
                update={"quality_budget_proxy": initial_eps}
            )
            self.solver = SolverFactory.create(new_config)

        solving_start = time.perf_counter()
        solver_output, feedback, iterations_used = self._solve_validate_stage(
            model, solver_input
        )
        solving_time = time.perf_counter() - solving_start

        validation_time = 0.0

        export_result = self._export_stage(solver_output)

        total_wall_time = time.perf_counter() - pipeline_start

        return PipelineResult(
            solver_output=solver_output,
            validation=feedback,
            export=export_result,
            iterations_used=iterations_used,
            total_wall_time_seconds=total_wall_time,
            scoring_time_seconds=scoring_time,
            profiling_time_seconds=profiling_time,
            solving_time_seconds=solving_time,
            validation_time_seconds=validation_time,
            pipeline_config=self.config,
            warnings=warnings,
        )

    #  Stage 1: Setup
    def _setup_stage(self) -> tuple[nn.Module, DataLoader]:
        """Load model and calibration data."""
        model = load_model(self.config.model_id)
        dataloader = self.calibration_manager.get_dataloader()
        return model, dataloader

    #  Stage 2: Scoring
    def _scoring_stage(
        self, model: nn.Module, dataloader: DataLoader
    ) -> List[LayerDescriptor]:
        """Run sensitivity scorer on all quantizable layers."""
        layer_names = get_quantizable_layers(model)
        return self.scorer.score(model, dataloader, layer_names=layer_names)

    #  Stage 3: Profiling
    def _profiling_stage(
        self, model: nn.Module, scorer_output: List[LayerDescriptor]
    ) -> ModelProfileResult:
        """Run hardware profiler on all scored layers."""
        layer_names = [ld.layer_name for ld in scorer_output]
        return self.profiler.profile_all_layers(
            model,
            layer_names=layer_names,
            target_batch_size=self.config.target_batch_size,
            model_id=self.config.model_id,
        )

    #  Stage 4: Assembly
    def _assembly_stage(
        self,
        scorer_output: List[LayerDescriptor],
        profiler_output: ModelProfileResult,
    ) -> SolverInput:
        """Merge scorer and profiler outputs into SolverInput."""
        return self._assemble_solver_input(
            scorer_output=scorer_output,
            profiler_output=profiler_output,
            model_id=self.config.model_id,
            gpu_spec=self.gpu_spec,
        )

    #  Stage 5: Solve-Validate feedback loop
    def _solve_validate_stage(
        self, model: nn.Module, solver_input: SolverInput
    ) -> tuple[SolverOutput, FeedbackSignal, int]:
        """Run the solve -> validate -> feedback loop."""
        solver_output = None
        feedback = None

        for iteration in range(self.config.max_feedback_iterations):
            if feedback is not None:
                self.solver = self._apply_feedback(self.solver, feedback, solver_output)

            solver_output = self.solver.solve(solver_input)

            if solver_output.solver_status == "infeasible":
                from .error_handling.types import InfeasibleError
                raise InfeasibleError(
                    "No valid assignment exists under current constraints - "
                    "consider relaxing memory budget, quality budget, or "
                    "unpinning forced-FP16 layers"
                )

            feedback = self.validator.validate(
                model, self.config.model_id, solver_output, stage="screening"
            )

            if feedback.passed:
                full_feedback = self.validator.validate(
                    model, self.config.model_id, solver_output, stage="full"
                )
                if full_feedback.passed:
                    feedback = full_feedback
                    break
                feedback = full_feedback

            log.info(
                "Iteration %d: PPL increase = %.2f%% (budget: %.2f%%)",
                iteration + 1,
                feedback.perplexity_increase_pct,
                self.config.validation.max_perplexity_increase_pct,
            )

        return solver_output, feedback, iteration + 1

    #  Stage 6: Export
    def _export_stage(self, solver_output: SolverOutput) -> ExportResult:
        """Export the final quantization assignment."""
        return self.exporter.export(
            solver_output=solver_output,
            model_id=self.config.model_id,
            target=self.config.export_target,
            output_dir=self.config.output_dir,
        )

    #  Feedback application

    def _apply_feedback(
        self,
        solver: QuantizationSolver,
        feedback: FeedbackSignal,
        solver_output: SolverOutput,
    ) -> QuantizationSolver:
        """
        Translate validation feedback into solver constraint modifications.

        Quality-minimizing mode: pin highest-KL layers to FP16, re-create solver.
        Resource-minimizing mode: scale quality_budget_proxy by the suggested
            factor; optionally pin worst layers. Re-create solver.
        Pareto mode: no feedback adjustment, return solver unchanged.
        """
        from .solver.config import QualityMinimizerConfig

        config = solver.config

        if isinstance(config, ParetoExplorerConfig):
            return solver

        if isinstance(config, QualityMinimizerConfig):
            new_pinned = dict(config.pinned_layers)
            for layer_name in feedback.suggested_pin_to_fp16:
                new_pinned[layer_name] = Precision.FP16.value
            new_config = config.model_copy(update={"pinned_layers": new_pinned})
            return SolverFactory.create(new_config)

        if isinstance(config, ResourceMinimizerConfig):
            updates: dict = {
                "quality_budget_proxy": (
                    config.quality_budget_proxy * feedback.suggested_quality_budget_scale
                ),
            }
            if feedback.suggested_pin_to_fp16:
                new_pinned = dict(config.pinned_layers)
                for layer_name in feedback.suggested_pin_to_fp16:
                    new_pinned[layer_name] = Precision.FP16.value
                updates["pinned_layers"] = new_pinned
            new_config = config.model_copy(update=updates)
            return SolverFactory.create(new_config)

        return solver

    #  Epsilon estimation for resource-minimizing mode

    def _estimate_initial_epsilon(
        self,
        solver_input: SolverInput,
        user_quality_budget_pct: float,
    ) -> float:
        """
        Heuristic initialization of eps for resource-minimizing mode.

        Strategy:
        1. Compute proxy score for all-INT4 (eps_max = Sum Omega_i * 1.0).
        2. Set initial eps = eps_max * (user_budget_pct / 100) * safety_margin.

        The safety_margin (0.5) accounts for the proxy's tendency to
        underestimate error at aggressive compression. The feedback loop
        will bisect toward the true epsilon.
        """
        sensitivities = self.solver._aggregate_sensitivities(solver_input.layers)
        epsilon_max = float(np.sum(sensitivities * 1.0))
        safety_margin = 0.5
        return epsilon_max * (user_quality_budget_pct / 100.0) * safety_margin

    #  Solver input assembly

    def _assemble_solver_input(
        self,
        scorer_output: list[LayerDescriptor],
        profiler_output: ModelProfileResult,
        model_id: str,
        gpu_spec: GPUSpec,
    ) -> SolverInput:
        """
        Merge scorer and profiler outputs into complete LayerDescriptors.

        The profiler returns {layer: {precision: {metric: value}}}.
        This method transposes to {metric: {precision: value}} per layer.
        """
        complete_layers = []
        for layer_desc in scorer_output:
            resource_data = profiler_output.entries[layer_desc.layer_name]

            memory_bytes: Dict[str, int] = {}
            latency_us: Dict[str, float] = {}
            peak_memory_bytes: Dict[str, int] = {}
            kernel_name: Dict[str, str] = {}
            is_memory_bound: Dict[str, bool] = {}

            for prec_str, profile in resource_data.items():
                memory_bytes[prec_str] = profile.memory_bytes
                latency_us[prec_str] = profile.latency_us
                peak_memory_bytes[prec_str] = profile.peak_memory_bytes
                kernel_name[prec_str] = profile.kernel_name
                is_memory_bound[prec_str] = profile.is_memory_bound

            complete_layers.append(layer_desc.model_copy(update={
                "memory_bytes": memory_bytes,
                "latency_us": latency_us,
                "peak_memory_bytes": peak_memory_bytes,
                "kernel_name": kernel_name,
                "is_memory_bound": is_memory_bound,
            }))

        model_config = self._get_model_config(model_id)
        num_blocks = model_config.get("num_hidden_layers", len(complete_layers) // 7)
        hidden_size = model_config.get("hidden_size", 4096)
        num_kv_heads = model_config.get("num_key_value_heads", 32)
        head_dim = hidden_size // model_config.get("num_attention_heads", 32)
        kv_bytes_fp16 = 2 * num_kv_heads * head_dim * 2
        kv_bytes_fp8 = 2 * num_kv_heads * head_dim * 1

        return SolverInput(
            layers=complete_layers,
            model_id=model_id,
            total_param_count=sum(l.param_count for l in complete_layers),
            num_transformer_blocks=num_blocks,
            gpu_name=gpu_spec.name,
            gpu_memory_bytes=int(gpu_spec.memory_gb * 1e9),
            gpu_compute_capability=gpu_spec.compute_capability,
            gpu_memory_bandwidth_tb_s=gpu_spec.memory_bandwidth_tb_s,
            available_precisions=self._filter_precisions(gpu_spec),
            kv_cache_bytes_per_token={
                "fp16": kv_bytes_fp16 * num_blocks,
                "fp8_e4m3": kv_bytes_fp8 * num_blocks,
            },
            max_sequence_length=model_config.get("max_position_embeddings", 4096),
            max_batch_size=self.config.target_batch_size,
        )

    def _get_model_config(self, model_id: str) -> dict:
        """Load HuggingFace model config as a dict for metadata extraction."""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id)
            return config.to_dict()
        except Exception:
            return {}

    def _filter_precisions(self, gpu_spec: GPUSpec) -> list[Precision]:
        """
        Determine which Precision candidates are available on target GPU.
        Filters by GPU hardware capabilities and kernel availability.
        """
        available = []
        candidate_set = set(p.value for p in self.config.solver.precision_candidates)

        for precision in Precision:
            if precision.value not in candidate_set:
                continue

            if precision == Precision.FP16:
                available.append(precision)
            elif precision == Precision.W8A8_FP8:
                if gpu_spec.supports_fp8 and "W8A8_FP8" in gpu_spec.available_kernels:
                    available.append(precision)
            elif precision == Precision.W8A8_INT8:
                if gpu_spec.supports_int8_tensor_core and "W8A8_INT8" in gpu_spec.available_kernels:
                    available.append(precision)
            elif precision == Precision.W4A16:
                if "W4A16" in gpu_spec.available_kernels:
                    available.append(precision)

        if len(available) < 2:
            raise ValueError(
                f"GPU {gpu_spec.name} supports fewer than 2 precision candidates "
                f"({[p.value for p in available]}) - mixed-precision optimization "
                f"requires at least 2 options - check GPU spec and kernel availability"
            )

        return available
