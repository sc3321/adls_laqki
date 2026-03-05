"""
GuidedQuant-style output saliency computation.

Computes per-layer sensitivity by measuring how much the end-to-end loss cares about each layer's output channels 
This replaces the diagonal Fisher approximation (which ignores cross-weight interactions within output channels) with a block-diagonal Fisher approximation that preserves those interactions

    Output saliency :
        Sensitivity_layer = mean_j( ||dL/dz_j||^2 )
        -> z_j = X @ w_j, so dL/dz_j 
        -> captures how all weights contributing to output channel j jointly affect the loss via the shared input X

    The relationship (GuidedQuant):
        sum_j (dL/dz_j)^2 o (z_j - ẑ_j)^2  =  sum_j (w_j - \hat{w}_j)^T H_j (w_j - \hat{w}_j)
            -> where H_j = X^T Diag((dL/dz_j)^2) X  is a per-channel saliency-weighted Hessian

    For bit-width assignment (ILP), can collapse to a scalar per layer:
        saliency_scalar = mean over batches of  (1/d_out) sum_j ||dL/dz_j||^2

Reference:
    Kim et al., "GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance", ICML 2025. Algorithm 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle

log = logging.getLogger(__name__)

# TODO : 

@dataclass
class SaliencyConfig:
    """Configuration for GuidedQuant output saliency computation."""

    num_batches: int = 4
    """
        Number of calibration batches to average over.
        GuidedQuant uses 4 in all experiments. 
        The signal is stable because we're measuring per-output-channel gradient energy, which has much lower variance than per-weight gradient estimates
    """

    num_groups: int = 4
    """
        Number of groups g to partition output channels into.
        GuidedQuant uses g=4 for all experiments
        Groups reduce the cost of building per-group Hessians from O(d_out * d_in^2) to O(g * d_in^2)
            -> Only matters if we end up building H_k for modified GPTQ
            -> the ILP scalar is computed from the full per-channel saliency regardless of g."""

    gradient_scale: float = 1e3
    """
        Scale factor applied to gradients before squaring to prevent underflow
        GuidedQuant uses 1e3 in all experiments. 
        The squared result is divided by scale^2 before returning, so the final values are correct
    """

    save_dir: Optional[str] = None
    """
        If set, save per-group saliency tensors to this directory as .pt files
        These can later be loaded to build per-group Hessians H_k for modified GPTQ reconstruction 
    """

    channel_dim: int = -1
    """
    Which dimension of the layer output is the channel/feature dimension.
    For nn.Linear outputs in transformers: (batch, seq_len, d_out) -> dim=-1."""


class OutputSaliencyCalculator:
    """
    Computes GuidedQuant output saliency for target layers.
    """

    def __init__(self, config: SaliencyConfig = SaliencyConfig()):
        self.config = config

    def compute(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        """
        Run backward passes on calibration data, collect output saliency.

        Args:
            model: The LLM, in eval mode, on device.
            dataloader: Calibration data yielding dicts with at minimum input_ids and labels (standard HF causal LM format)
            layer_names: Fully qualified module names to score.
                         - Must be modules whose output is a Tensor (e.g. nn.Linear) or a tuple/list whose first Tensor element is the activation

        Returns:
            scalars:  {layer_name -> float}  One saliency scalar per layer.
                      This is mean-over-batches of (1/d_out) * ||dL/dZ||^2_F.
                        - replacement for fisher_diagonal_mean in LayerDescriptor.

            grouped:  {layer_name -> Tensor(num_groups,)}  Per-group saliency
                      grouped[name][k] = (1/|J_k|) * sum_{j in J_k} sum_i (dL/dz_ij)^2 averaged over batches
                        - used to build H_k if needed later.
        """
        model.eval()
        device = next(model.parameters()).device
        scale = self.config.gradient_scale
        num_groups = self.config.num_groups

        #Per-channel accums: (d_out,) per layer 
        # Accumulated over batches, then averaged at the end.
        channel_accum: Dict[str, Tensor] = {}
        channel_counts: Dict[str, int] = {}  # track d_out per layer for grouping

        hooks: List[RemovableHandle] = []

        # The hook fires during loss.backward() 
        # At that point grad_output[0] is dL/dZ for the module. 
        for name in layer_names:
            try:
                module = model.get_submodule(name)
            except AttributeError:
                log.warning(f"Module not found, skipping saliency: {name}")
                continue

            handle = module.register_full_backward_hook(
                self._make_backward_hook(name, channel_accum, channel_counts, scale)
            )
            hooks.append(handle)

        #Forward + backward passes 
        n_batches = 0
        dl_iter = iter(dataloader)

        for _ in range(self.config.num_batches):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                batch = next(dl_iter)

            batch_on_device = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad()

            outputs = model(**batch_on_device)
            loss = outputs.loss
            loss.backward()

            n_batches += 1
            del loss, outputs, batch_on_device

        #Cleanup hooks 
        for h in hooks:
            h.remove()
        model.zero_grad()
        torch.cuda.empty_cache()

        if n_batches == 0:
            log.warning("No batches processed for saliency computation")
            empty_scalar = {name: 0.0 for name in layer_names}
            empty_grouped = {
                name: torch.zeros(num_groups, dtype=torch.float64)
                for name in layer_names
            }
            return empty_scalar, empty_grouped

        #Reduce accumulators to scalars and grouped tensors 
        scalars: Dict[str, float] = {}
        grouped: Dict[str, Tensor] = {}

        scale_sq = scale ** 2

        for name in layer_names:
            if name not in channel_accum:
                scalars[name] = 0.0
                grouped[name] = torch.zeros(num_groups, dtype=torch.float64)
                continue

            # channel_accum[name] shape: (d_out,)
            # Contains sum over all (batches * batch_tokens) of (dL/dz_j * scale)^2
            per_channel = channel_accum[name] / (n_batches * scale_sq)
            d_out = per_channel.shape[0]

            # mean per-channel saliency (for ILP)
            scalars[name] = per_channel.mean().item()

            # partition into num_groups, mean within each group
            g = min(num_groups, d_out)
            usable = (d_out // g) * g
            grouped[name] = per_channel[:usable].reshape(g, -1).mean(dim=1)

        if self.config.save_dir is not None:
            #save per-group saliency to disk 
            save_path = Path(self.config.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            for name in layer_names:
                if name in grouped:
                    safe_name = name.replace(".", "_")
                    torch.save(grouped[name], save_path / f"saliency_{safe_name}.pt")
            log.info(f"Saved per-group saliency to {save_path}")

        return scalars, grouped

    def _make_backward_hook(
        self,
        layer_name: str,
        channel_accum: Dict[str, Tensor],
        channel_counts: Dict[str, int],
        scale: float,
    ):
        """
        Build a backward hook that accumulates squared output gradients.

        Scale the gradient by scale before squaring to avoid underflow (GuidedQuant uses 1e3) 
        The caller divides by scale^2 after accumulation
        """
        cdim = self.config.channel_dim

        def hook(
            module: nn.Module,
            grad_input: Tuple[Any, ...],
            grad_output: Tuple[Any, ...],
        ) -> None:
            # get first non-None tensor in grad_output
            dLdZ = None
            for g in grad_output:
                if g is not None and torch.is_tensor(g):
                    dLdZ = g
                    break

            if dLdZ is None:
                return

            # Scale, square, sum over all dims except channel dim.
            # dLdZ shape for nn.Linear: (batch, seq_len, d_out)
            # Result shape: (d_out,)
            dLdZ_f = dLdZ.float() * scale
            reduce_dims = tuple(d for d in range(dLdZ_f.ndim) if d != (cdim % dLdZ_f.ndim))
            per_channel = (dLdZ_f ** 2).sum(dim=reduce_dims)

            # Accumulate
            if layer_name not in channel_accum:
                channel_accum[layer_name] = per_channel.to(dtype=torch.float64).detach()
                channel_counts[layer_name] = per_channel.shape[0]
            else:
                channel_accum[layer_name] += per_channel.to(dtype=torch.float64).detach()

        return hook