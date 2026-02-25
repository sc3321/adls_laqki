from pydantic import BaseModel, ConfigDict
from typing import Dict, List

from calibration.models import CalibrationConfig
from solver import (
    SolverConfig, 
    SolverInput, 
    SolverOutput, 
    SolverFactory,
    LayerDescriptor
)
from solver.types import Precision
from solver.models import FeedbackSignal
from validation.config import ValidationConfig 
from calibration.manager import CalibrationDataManager 
from export import ConfigurationExporter, ExportResult
from validation.engine import ValidationEngine 
from scorer.base import SensitivityScorer 
from profiler import LayerProfiler, GPUSpec
from utils.model_utils import load_model, get_quantizable_layers

class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_id: str
    calibration: CalibrationConfig = CalibrationConfig()
    validation: ValidationConfig = ValidationConfig()
    solver: SolverConfig 
    export_target: str = "vllm"
    output_dir: str = "./qpe_output"
    
    max_feedback_iterations: int = 5
    wandb_project: str = "qpe"
    wandb_enabled: bool = True

class PipelineResult(BaseModel):
    """
    Complete output of a QPE pipeline run
    
    Bundles the solver decision, validator assessment, exported artifact metadata, and pipeline execution diagnostics
    """
    model_config = ConfigDict(frozen=True)
    schema_version: int = 2
    
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
    """
    Orchestrates Score -> Solve -> Validate -> Export with feedback.
    """
    
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
        self.solver = SolverFactory.create(config.solver)

    def run(self) -> PipelineResult:

        # Score + Profile (formulation-independent)
        model = load_model(self.config.model_id)
        dataloader = self.calibration_manager.get_dataloader()
        
        scorer_output = self.scorer.score(model, dataloader, get_quantizable_layers(model))
        profiler_output = self.profiler.profile_all_layers(model, ...)
        
        solver_input = self._assemble_solver_input(scorer_output, profiler_output, ...)
        
        self.validator.compute_fp16_baseline(self.config.model_id)
        
        # Solve -> Validate -> Feedback (formulation-aware)
        solver_output = None
        feedback = None
        
        for iteration in range(self.config.max_feedback_iterations):
            # Apply any corrections from previous iteration
            if feedback is not None:
                self.solver = self._apply_feedback(self.solver, feedback, solver_output)
            
            solver_output = self.solver.solve(solver_input)
            
            if solver_output.solver_status == "infeasible":
                from error_handling.types import InfeasibleError
                raise InfeasibleError(
                    "No valid assignment exists under current constraints - "
                    "consider relaxing memory budget, quality budget, or "
                    "unpinning forced-FP16 layers"
                )
            
            feedback = self.validator.validate(
                self.config.model_id, solver_output, stage="screening"
            )
            
            if feedback.passed:
                full_feedback = self.validator.validate(
                    self.config.model_id, solver_output, stage="full"
                )
                if full_feedback.passed:
                    feedback = full_feedback
                    break
                feedback = full_feedback
            
            log.info(
                f"Iteration {iteration + 1}: PPL increase = "
                f"{feedback.perplexity_increase_pct:.2f}% "
                f"(budget: {self.config.validation.max_perplexity_increase_pct}%)"
            )
        
        output = self.exporter.export(solver_output, self.config.model_id, ...)
        
        return PipelineResult(
            solver_output=solver_output,
            validation=feedback,
            export=output,
            iterations_used=iteration + 1,
        )


    def _estimate_initial_epsilon(
        self,
        solver_input: SolverInput,
        user_quality_budget_pct: float,
    ) -> float:
        """
        Heuristic initialization of eps for resource-minimizing mode.
        
        Strategy:
        1. Compute proxy score for all-FP16 (eps_min = 0).
        2. Compute proxy score for all-INT4 (eps_max = Sum Omega_i · 1.0).
        3. Set initial eps = eps_max x (user_budget_pct / 100) x safety_margin.
        
        The safety_margin (default 0.5) accounts for the proxy's tendency to underestimate error at aggressive compression. 
        Starting conservative means the first iteration likely passes validation, and subsequent iterations loosen eps toward the actual boundary.
        
        The feedback loop will bisect toward the true epsilon.
        """
        import numpy as np
        sensitivities = self.solver._aggregate_sensitivities(solver_input.layers)
        epsilon_max = float(np.sum(sensitivities * 1.0))  # all INT4
        safety_margin = 0.5
        return epsilon_max * (user_quality_budget_pct / 100.0) * safety_margin

    def _assemble_solver_input(
        self,
        scorer_output: list[LayerDescriptor],
        profiler_output: Dict[str, Dict],
        model_id: str,
        gpu_spec: GPUSpec,
    ) -> SolverInput:
        """
        Merge scorer and profiler outputs into complete LayerDescriptors
        
        - The scorer produces LayerDescriptors with sensitivity fields filled and resource fields empty 
        - The profiler produces resource data keyed by layer name 
        - This method combines them
        
        Validation: asserts every layer has both sensitivity and resource data
        Missing data causes a clear error rather than silent default values
        """
        complete_layers = []
        for layer_desc in scorer_output:
            resource_data = profiler_output[layer_desc.layer_name]
            complete_layers.append(layer_desc.model_copy(update={
                "memory_bytes": resource_data["memory_bytes"],
                "latency_us": resource_data["latency_us"],
                "peak_memory_bytes": resource_data["peak_memory_bytes"],
                "kernel_name": resource_data["kernel_name"],
                "is_memory_bound": resource_data["is_memory_bound"],
            }))
        
        return SolverInput(
            layers=complete_layers,
            model_id=model_id,
            total_param_count=sum(l.param_count for l in complete_layers),
            num_transformer_blocks=...,
            gpu_name=gpu_spec.name,
            gpu_memory_bytes=int(gpu_spec.memory_gb * 1e9),
            gpu_compute_capability=gpu_spec.compute_capability,
            gpu_memory_bandwidth_tb_s=gpu_spec.memory_bandwidth_tb_s,
            available_precisions=self._filter_precisions(gpu_spec),
            kv_cache_bytes_per_token=...,
            max_sequence_length=...,
            max_batch_size=...,
        )

    def _filter_precisions(self, gpu_spec: GPUSpec) -> list[Precision]:
        """
        Determine which Precision candidates are available on target GPU
        
        Filters solver config precision_candidates by GPU hardware capabilities (compute capability determines FP8/FP4 support) and kernel availability
        
        Ensures solver never proposes a precision that would fail at runtime
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