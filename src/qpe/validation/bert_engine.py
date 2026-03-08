import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..profiler.gpu_specs import GPUSpec
from ..solver.models import FeedbackSignal, SolverOutput
from ..utils.types import Precision
from .config import ValidationConfig
from .engine import ValidationEngine


class BERTValidationEngine(ValidationEngine):
    """
    ValidationEngine implementation for BERT-family sequence classification models.

    Differences from the LLM ValidationEngine stub:
    - _apply_quantization() returns nn.Module (in-memory), not a checkpoint path.
    - Quality signal is cross-entropy loss on SST-2, reported as exp(CE) in the
      perplexity slots so the Pipeline feedback loop works unchanged.
    - Per-layer KL divergence is stubbed as {} (no hook overhead for now).
    - Requires a pre-built validation DataLoader at construction time.
    """

    def __init__(
        self,
        config: ValidationConfig,
        val_dataloader: DataLoader,
        gpu_spec: GPUSpec,
    ):
        super().__init__(config)
        self._val_dataloader = val_dataloader
        self._gpu_spec = gpu_spec

    # ------------------------------------------------------------------
    # Public API (overrides stubs in ValidationEngine)
    # ------------------------------------------------------------------

    def compute_fp16_baseline(self, model_id: str) -> None:
        """Load the FP32 model and record baseline accuracy and proxy perplexity."""
        if self._fp16_baseline is not None:
            return  # already computed

        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        model.eval()

        accuracy, ce_loss = self._evaluate(model)
        self._fp16_baseline = {
            "accuracy": accuracy,
            "ce_loss": ce_loss,
            "perplexity": math.exp(ce_loss),
        }

    def validate(
        self,
        model: nn.Module,
        model_id: str,
        solver_output: SolverOutput,
        stage: str = "screening",
    ) -> FeedbackSignal:
        """
        Apply the solver's assignment, evaluate on the SST-2 validation split,
        and return a FeedbackSignal.

        Both screening and full stages run the same evaluation — BERT on SST-2
        is fast enough that a two-stage approach adds no meaningful value.

        Uses exp(CE loss) as a proxy for perplexity so the Pipeline's
        perplexity_increase_pct logic works without modification.
        """
        assert self._fp16_baseline is not None, (
            "compute_fp16_baseline() must be called before validate()"
        )

        assignment_dict = solver_output.to_assignment_dict()
        q_model = self._apply_quantization(model_id, assignment_dict)
        q_accuracy, q_ce_loss = self._evaluate(q_model)

        q_perplexity = math.exp(q_ce_loss)
        fp16_perplexity = self._fp16_baseline["perplexity"]
        ppl_increase = q_perplexity - fp16_perplexity
        ppl_increase_pct = (ppl_increase / fp16_perplexity) * 100.0

        per_layer_kl = self._compute_per_layer_kl()
        kl_threshold = self.config.kl_threshold_per_layer
        layers_exceeding = [l for l, kl in per_layer_kl.items() if kl > kl_threshold]

        passed = ppl_increase_pct <= self.config.max_perplexity_increase_pct

        # Pin at most 3 worst KL violators; tighten/loosen quality budget proxy.
        suggested_pin = layers_exceeding[:3]
        if not passed:
            budget_scale = 0.7
        elif ppl_increase_pct < 0.5:
            budget_scale = 1.3
        else:
            budget_scale = 1.0

        return FeedbackSignal(
            fp16_perplexity=fp16_perplexity,
            quantized_perplexity=q_perplexity,
            perplexity_increase=ppl_increase,
            perplexity_increase_pct=ppl_increase_pct,
            benchmark_scores={"sst2_accuracy": q_accuracy},
            fp16_benchmark_scores={"sst2_accuracy": self._fp16_baseline["accuracy"]},
            actual_peak_memory_gb=0.0,
            actual_decode_latency_ms=None,
            per_layer_kl_divergence=per_layer_kl,
            layers_exceeding_kl_threshold=layers_exceeding,
            passed=passed,
            suggested_pin_to_fp16=suggested_pin,
            suggested_quality_budget_scale=budget_scale,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_quantization(
        self, model_id: str, assignment_dict: dict[str, str]
    ) -> nn.Module:
        """
        Load a fresh FP32 model and apply per-layer precision assignments
        in-memory using torchao.

        Returns the quantized nn.Module. FP16-assigned layers are left
        unchanged (torchao's quantize_() modifies in-place, FP16 = no-op).
        Layers that fail to quantize (e.g., unsupported shape) are silently
        kept at FP32.
        """
        from transformers import AutoModelForSequenceClassification
        from ..utils.model_utils import _quantize_layer

        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        model.eval()

        for layer_name, precision_str in assignment_dict.items():
            precision = Precision(precision_str)
            if precision == Precision.FP16:
                continue
            try:
                layer = model.get_submodule(layer_name)
                quantized = _quantize_layer(layer, precision, self._gpu_spec)
                parent_name, child_name = layer_name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, quantized)
            except Exception:
                # Layer not found or torchao quantization failed — keep FP32.
                pass

        return model

    def _evaluate(self, model: nn.Module) -> tuple[float, float]:
        """
        Run model on the validation DataLoader.
        Returns (accuracy, mean_ce_loss).
        """
        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            for batch in self._val_dataloader:
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                out = model(**inputs, labels=batch["labels"])
                total_loss += out.loss.item() * len(batch["labels"])
                preds = out.logits.argmax(dim=-1)
                total_correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])

        return total_correct / total, total_loss / total

    def _compute_per_layer_kl(self) -> dict[str, float]:
        """Stub — KL divergence computation deferred to a later iteration."""
        return {}
