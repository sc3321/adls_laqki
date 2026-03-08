"""
End-to-end QPE pipeline run on BERT for SST-2 sentiment classification.

Model:   textattack/bert-base-uncased-SST-2  (already fine-tuned on SST-2)
Solver:  ILPQualityMinimizer
Target:  minimize quality loss (CE loss increase) under a memory budget
Export:  quantization_config.json (pytorch target)

Usage:
    python scripts/run_bert.py [--memory-budget-gb 0.3] [--gpu a100_40gb]
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from transformers import AutoTokenizer

from qpe.calibration.manager import CalibrationDataManager
from qpe.calibration.models import CalibrationConfig
from qpe.export.exporter import ConfigurationExporter
from qpe.pipeline import Pipeline, PipelineConfig
from qpe.profiler.gpu_specs import GPU_REGISTRY, detect_gpu
from qpe.profiler.layer_profiler import LayerProfiler
from qpe.scorer.hessian import HessianTraceScorer, HessianTraceScorerConfig
from qpe.solver.config import QualityMinimizerConfig
from qpe.utils.types import Precision
from qpe.validation.bert_engine import BERTValidationEngine
from qpe.validation.config import ValidationConfig

MODEL_ID = "textattack/bert-base-uncased-SST-2"


def main(memory_budget_gb: float, gpu_name: str | None) -> None:
    gpu_spec = GPU_REGISTRY[gpu_name] if gpu_name else detect_gpu()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    calibration_config = CalibrationConfig(
        num_samples=256,
        sequence_length=128,  # SST-2 sentences are short; 128 covers 99%+
        datasets=["sst2"],
        validation_split=0.2,
    )
    calibration_manager = CalibrationDataManager(calibration_config, tokenizer)

    scorer = HessianTraceScorer(
        HessianTraceScorerConfig(
            saliency_mode="guided",
            compute_hessian_trace=True,
            n_hutchinson_samples=50,  # reduced from 200: BERT is small, signal is stable
        )
    )

    profiler = LayerProfiler(
        gpu_spec=gpu_spec,
        num_warmup=10,
        num_measurements=50,
        seq_len=128,
    )

    validation_config = ValidationConfig(
        perplexity_dataset="glue/sst2",
        max_perplexity_increase_pct=5.0,
        benchmark_tasks=["sst2_accuracy"],
        kl_threshold_per_layer=0.1,
    )
    validator = BERTValidationEngine(
        config=validation_config,
        val_dataloader=calibration_manager.get_validation_dataloader(batch_size=32),
        gpu_spec=gpu_spec,
    )

    # BERT-base weights are ~440 MB in FP32. A 0.3 GB budget forces most
    # layers into INT8 or INT4; adjust upward to allow more FP16 layers.
    solver_config = QualityMinimizerConfig(
        memory_budget_gb=memory_budget_gb,
        precision_candidates=[Precision.FP16, Precision.W8A8_INT8, Precision.W4A16],
        # pooler and classifier are excluded by get_quantizable_layers(), but
        # pin them explicitly as a safety net in case the model has unusual naming.
        pinned_layers={
            "bert.pooler.dense": Precision.FP16.value,
            "classifier": Precision.FP16.value,
        },
    )

    pipeline_config = PipelineConfig(
        model_id=MODEL_ID,
        calibration=calibration_config,
        validation=validation_config,
        solver=solver_config,
        export_target="pytorch",
        output_dir="./bert_qpe_output",
        task="sequence_classification",
        gpu_spec_name=gpu_name,
        target_batch_size=1,
        max_feedback_iterations=3,
    )

    pipeline = Pipeline(
        calibration_manager=calibration_manager,
        scorer=scorer,
        profiler=profiler,
        validator=validator,
        exporter=ConfigurationExporter(),
        config=pipeline_config,
    )

    result = pipeline.run()

    so = result.solver_output
    fb = result.validation
    print("\n=== QPE BERT Results ===")
    print(f"Solver status  : {so.solver_status}")
    print(f"Avg bitwidth   : {so.average_bitwidth:.2f} bits")
    print(f"Weight memory  : {so.total_memory_bytes / 1e6:.1f} MB")
    print(f"Iterations     : {result.iterations_used}")
    print(f"Accuracy FP16  : {fb.fp16_benchmark_scores.get('sst2_accuracy', 0):.4f}")
    print(f"Accuracy quant : {fb.benchmark_scores.get('sst2_accuracy', 0):.4f}")
    print(f"PPL increase   : {fb.perplexity_increase_pct:.2f}%")
    print(f"Passed         : {fb.passed}")
    print(f"Output         : {result.export.output_path}")
    print(f"Precision dist : {so.to_assignment_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-budget-gb", type=float, default=0.3)
    parser.add_argument("--gpu", type=str, default=None,
                        help=f"GPU spec name. Available: {list(GPU_REGISTRY.keys())}")
    args = parser.parse_args()
    main(args.memory_budget_gb, args.gpu)
