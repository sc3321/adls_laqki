import torch
from torch.utils.data.dataloader import DataLoader
from typing import Optional, Dict, List, Tuple 
from qpe.scorer.models import ActivationSpec, StatsConfig
from qpe.scorer.statistics import ActivationStatsCollector, HutchinsonTraceCalculator, compute_weight_stats
from ..solver.models import LayerDescriptor
from typing import List
import torch.nn as nn
from dataclasses import dataclass

def _tensor_to_scalar(
    t: Optional[torch.Tensor],
    reduction: str = "mean",
) -> float:
    """
    Safely reduce a per-channel tensor to a single float.
    """
    if t is None:
        return 0.0
    if t.numel() == 0:
        return 0.0
    if reduction == "mean":
        return t.mean().item()
    elif reduction == "max":
        return t.max().item()
    elif reduction == "min":
        return t.min().item()
    elif reduction == "sum":
        return t.sum().item()
    else:
        return t.mean().item()

@dataclass
class HessianTraceScorerConfig:
    """
    Configuration for the HessianTraceScorer.
    """
    n_hutchinson_samples: int = 200         # T random vectors per layer
    layers_per_group: Optional[int] = None  # layers per forward pass (None -> auto)

    collect_fisher: bool = True
    num_fisher_batches: int = 50            # batches to average over

    fix_batches: bool = False               # same batch for all layer groups?

    eps: float = 1e-8
    outlier_sigma: float = 6.0


class HessianTraceScorer:
    """
    HAWQ-V2/V3 style multi-signal sensitivity scorer.

    Computes per-layer sensitivity signals that quantify how much each layer's output changes under weight perturbation
        - These signals feed the ILP solver objective/constraint via LayerDescriptor

    Signals:
      o hessian_trace        — via Hutchinson estimator
      o gradient_norm        — via single backward pass
      o fisher_diagonal_mean — avgd over batches
      o activation stats     — kurtosis, outlier rate, dynamic range, max magnitude
      o weight_range         — max(w) - min(w)
      o weight_kurtosis      — 4th moment of weight distribution
    """

    def __init__(
        self,
        config: HessianTraceScorerConfig,
        dtype: torch.dtype = torch.float32,
        trace_calculator: Optional[HutchinsonTraceCalculator] = None,
    ):
        self.config = config
        self.num_hutchinson_samples = config.n_hutchinson_samples
        self.collect_fisher = config.collect_fisher
        self.num_fisher_batches = config.num_fisher_batches
        self.dtype = dtype
        self.eps = config.eps

        # The trace calculator handles grouped-layer memory management.
        # If not provided, construct one from config.
        self.trace_calculator = trace_calculator or HutchinsonTraceCalculator(
            layers_per_group=config.layers_per_group,
            fix_batches=config.fix_batches,
        )


    def score(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
    ) -> List[LayerDescriptor]:
        """
        Run the full scoring pipeline and return one LayerDescriptor per layer
        Resource fields (memory_bytes, latency_us, ...) left for the LayerProfiler
        """
        model.eval()
        device = next(model.parameters()).device

        act_stats = self._collect_activation_stats(model, dataloader, layer_names, device)
        weight_ranges, weight_kurtoses = compute_weight_stats(model, layer_names)
        grad_norms = self._compute_gradient_norms(model, dataloader, layer_names, device)

        fisher_means: Dict[str, float] = {name: 0.0 for name in layer_names}
        if self.collect_fisher:
            fisher_means = self._compute_fisher_diagonal(
                model, dataloader, layer_names, device
            )


        trace_results = self.trace_calculator.compute_trace(
            model=model,
            dataloader=dataloader,
            layer_names=layer_names,
            n_samples=self.num_hutchinson_samples,
            dtype=self.dtype,
        )


        descriptors = self._assemble_descriptors(
            model=model,
            layer_names=layer_names,
            act_stats=act_stats,
            weight_ranges=weight_ranges,
            weight_kurtoses=weight_kurtoses,
            grad_norms=grad_norms,
            fisher_means=fisher_means,
            trace_results=trace_results,
        )

        return descriptors

    def _collect_activation_stats(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
        device: torch.device,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Run the full dataloader through the model with forward hooks attached to each target layer
            - Returns finalized per-channel statistics for each layer
        """
        specs = [ActivationSpec(layer_name=name) for name in layer_names]
        stats_config = StatsConfig(
            eps=self.eps,
            outlier_sigma=self.config.outlier_sigma,
        )

        collector = ActivationStatsCollector(
            model, specs, config=stats_config, strict=False,
        )

        with collector:
            for batch in dataloader:
                batch_on_device = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    model(**batch_on_device)
                del batch_on_device

        return collector.finalize()

    def _compute_fisher_diagonal(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute mean of Fisher info diagonal for each layer
        """
        layers: Dict[str, nn.Module] = {}
        for name in layer_names:
            try:
                layers[name] = model.get_submodule(name)
            except AttributeError:
                continue

        # init accum - one FP64 tensor per parameter, per layer
        fisher_accum: Dict[str, List[torch.Tensor]] = {}
        for name, layer in layers.items():
            fisher_accum[name] = [
                torch.zeros_like(p.data, dtype=torch.float64)
                for p in layer.parameters()
                if p.requires_grad
            ]

        n_batches_processed = 0
        dataloader_iter = iter(dataloader)

        for _ in range(self.num_fisher_batches):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Wrap around
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            batch_on_device = {k: v.to(device) for k, v in batch.items()}

            model.zero_grad()
            outputs = model(**batch_on_device)
            loss = outputs.loss
            loss.backward()

            # accum g^2
            for name, layer in layers.items():
                grad_params = [
                    p for p in layer.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                for accum, p in zip(fisher_accum[name], grad_params):
                    accum += p.grad.data.to(torch.float64) ** 2

            n_batches_processed += 1
            del loss, outputs, batch_on_device

        model.zero_grad()
        torch.cuda.empty_cache()

        # Fisher diagonal per layer
        fisher_means: Dict[str, float] = {}
        for name in layer_names:
            if name in fisher_accum and n_batches_processed > 0:
                # mean(F_ii) = mean over all parameters of (sum_batches(g^2) / n_batches)
                total_sum = sum(a.sum().item() for a in fisher_accum[name])
                total_params = sum(a.numel() for a in fisher_accum[name])
                if total_params > 0:
                    fisher_means[name] = total_sum / (n_batches_processed * total_params)
                else:
                    fisher_means[name] = 0.0
            else:
                fisher_means[name] = 0.0

        return fisher_means


    def _compute_gradient_norms(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute grad norm for each layer from a single forward + backward pass
        """
        grad_norms: Dict[str, float] = {}

        batch = next(iter(dataloader))
        batch_on_device = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch_on_device)
        loss = outputs.loss
        loss.backward()

        for name in layer_names:
            try:
                layer = model.get_submodule(name)
            except AttributeError:
                grad_norms[name] = 0.0
                continue

            layer_grad_norm_sq = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    layer_grad_norm_sq += p.grad.data.norm(2).item() ** 2

            grad_norms[name] = layer_grad_norm_sq ** 0.5

        # Cleanup
        model.zero_grad()
        del loss, outputs, batch_on_device
        torch.cuda.empty_cache()

        return grad_norms

    def _assemble_descriptors(
        self,
        model: nn.Module,
        layer_names: List[str],
        act_stats: Dict[str, Dict[str, torch.Tensor]],
        weight_ranges: Dict[str, float],
        weight_kurtoses: Dict[str, float],
        grad_norms: Dict[str, float],
        fisher_means: Dict[str, float],
        trace_results: Dict[str, Tuple[float, bool]],
    ) -> List[LayerDescriptor]:
        """
        Pack all computed signals into LayerDescriptor objects
        """
        descriptors: List[LayerDescriptor] = []
        n_layers = len(layer_names)

        for i, name in enumerate(layer_names):
            try:
                layer = model.get_submodule(name)
            except AttributeError:
                continue

            param_count = sum(p.numel() for p in layer.parameters())

            hessian_trace, hessian_is_psd = trace_results.get(name, (0.0, True))
            layer_act = act_stats.get(name, {})

            activation_kurtosis = _tensor_to_scalar(
                layer_act.get("kurtosis"), reduction="mean"
            )
            channel_outlier_rate = _tensor_to_scalar(
                layer_act.get("outlier_rate"), reduction="mean"
            )
            dynamic_range_ratio = _tensor_to_scalar(
                layer_act.get("dynamic_range"), reduction="mean"
            )
            activation_max_magnitude = _tensor_to_scalar(
                layer_act.get("max_abs"), reduction="max"
            )

            descriptors.append(
                LayerDescriptor(
                    # Metadata
                    layer_name=name,
                    layer_type=type(layer).__name__,
                    layer_index=i,
                    relative_depth=i / max(n_layers - 1, 1),
                    param_count=param_count,

                    # Sensitivity signals
                    hessian_trace=hessian_trace,
                    hessian_is_psd=hessian_is_psd,
                    gradient_norm=grad_norms.get(name, 0.0),
                    fisher_diagonal_mean=fisher_means.get(name, 0.0),
                    activation_kurtosis=activation_kurtosis,
                    channel_outlier_rate=channel_outlier_rate,
                    dynamic_range_ratio=dynamic_range_ratio,
                    activation_max_magnitude=activation_max_magnitude,
                    weight_range=weight_ranges.get(name, 0.0),
                    weight_kurtosis=weight_kurtoses.get(name, 0.0),

                    memory_bytes={},
                    latency_us={},
                    peak_memory_bytes={},
                    kernel_name={},
                    is_memory_bound={},
                )
            )

        return descriptors