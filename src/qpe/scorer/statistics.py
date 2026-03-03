from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from .models import (
    ActivationSpec,
    StatsConfig,
    LayerAccumulatorState,
    Tensor,
    ModuleOutput,
)


def compute_weight_stats(
    model: nn.Module, layer_names: List[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Returns (weight_ranges, weight_kurtoses) dicts."""
    eps = 1e-8
    ranges, kurtoses = {}, {}

    for name in layer_names:
        layer = model.get_submodule(name)
        w = layer.weight.data.flatten().float()
        ranges[name] = (w.max() - w.min()).item()

        mean = w.mean()
        var = w.var() + eps
        kurtoses[name] = (((w - mean) ** 4).mean() / (var**2)).item()

    return ranges, kurtoses


class ActivationStatsCollector:
    """
    Collects activation statistics for layers over many batches
    """

    def __init__(
        self,
        model: nn.Module,
        specs: Sequence[ActivationSpec],
        *,
        config: StatsConfig = StatsConfig(),
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> None:
        self._model = model
        self._specs = list(specs)
        self._config = config
        self._strict = strict
        self._device = device

        self._handles: List[RemovableHandle] = []
        self._states: Dict[str, LayerAccumulatorState] = {}

        self._validate_and_register()

    # context manager
    def __enter__(self) -> "ActivationStatsCollector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove_hooks()

    # public API
    @property
    def states(self) -> Mapping[str, LayerAccumulatorState]:
        """
        Exposes raw accumulators for debugging/testing.
        """
        return self._states

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """
        Clears accumulated statistics but keeps hooks registered.
        """
        self._states.clear()

    def finalize(self) -> Dict[str, Dict[str, Tensor]]:
        """
        Computes derived metrics (mean/var/kurtosis/etc) from raw accumulators

        Returns a dict:
          layer_name -> metrics dict
            - each metrics dict contains per-channel tensors (shape C,) and optionally layer-level scalars
        """
        results: Dict[str, Dict[str, Tensor]] = {}
        for layer_name, st in self._states.items():
            results[layer_name] = self._finalize_layer(st)
        return results

    # setup
    def _validate_and_register(self) -> None:
        """
        Resolves layers and registers hooks
        """
        for spec in self._specs:
            try:
                module = self._model.get_submodule(spec.layer_name)
            except AttributeError as e:
                if self._strict:
                    raise ValueError(f"Layer not found: {spec.layer_name}") from e
                else:
                    continue

            handle = module.register_forward_hook(self._make_forward_hook(spec))
            self._handles.append(handle)

    # hook construction
    def _make_forward_hook(
        self, spec: ActivationSpec
    ) -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
        """
        Returns a hook that:
          - extracts activation tensor
          - reduces over dims except channel
          - updates streaming accumulators
        """

        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            act = self._extract_activation(output, spec)

            # can move to a specific device for accumulation
            if self._device is not None and act.device != self._device:
                act = act.to(self._device)

            # Ensure numeric dtype for accumulation stability
            act_acc = act.to(dtype=self._config.accumulator_dtype)

            # Flatten/reduce into per-channel batch summaries
            reduced = self._reduce_for_stats(act_acc, spec)

            # Update state from reduced summaries
            self._update_state(spec.layer_name, reduced)

        return hook

    # extraction + reduction
    def _extract_activation(self, output: ModuleOutput, spec: ActivationSpec) -> Tensor:
        """
        Choose the tensor to treat as activation
        - If spec.output_extractor is provided, call it
        - Otherwise: choose the first tensor found in the output structure
        """
        if spec.output_extractor is not None:
            t = spec.output_extractor(output)
            if not torch.is_tensor(t):
                raise TypeError(
                    f"Extractor for {spec.layer_name} did not return Tensor."
                )
            return t

        # Default - find first tensor
        t = self._first_tensor(output)
        if t is None:
            raise TypeError(f"No Tensor output found for layer {spec.layer_name}.")
        return t

    def _reduce_for_stats(self, act: Tensor, spec: ActivationSpec) -> Dict[str, Tensor]:
        """
        Core per-batch computation.

        Input: act with arbitrary shape, with channel_dim specifying channels.
        Output: dict of per-channel vectors (shape C,).

        Computes:
          - sum1, sum2, sum4 per channel (for mean, variance, kurtosis)
          - max_abs per channel
          - batch_std per channel (needed for outlier test)
          - median_abs per channel
        """
        reduce_dims = spec.reduce_dims or tuple(
            d for d in range(act.ndim) if d != spec.channel_dim
        )

        C = act.shape[spec.channel_dim]
        count = act.numel() // C  # elements per channel

        sum1 = act.sum(dim=reduce_dims)
        sum2 = (act**2).sum(dim=reduce_dims)
        sum4 = (act**4).sum(dim=reduce_dims)
        max_abs = act.abs().amax(dim=reduce_dims)
        std = act.std(dim=reduce_dims)
        median_abs = act.abs().median(dim=reduce_dims).values

        return {
            "count": count,
            "sum1": sum1,
            "sum2": sum2,
            "sum4": sum4,
            "max_abs": max_abs,
            "std": std,
            "median_abs": median_abs,
        }

    #  state update
    def _update_state(self, layer_name: str, reduced: Dict[str, Tensor]) -> None:
        """
        Updates/initializes LayerAccumulatorState for a layer.
        """
        eps = self._config.eps
        sigma = self._config.outlier_sigma

        #  Count how many channels are outliers this batch
        outlier_mask = reduced["max_abs"] > sigma * (reduced["std"] + eps)
        batch_outlier_count = outlier_mask.sum().item()

        if layer_name not in self._states:
            # initialize state from scratch
            self._states[layer_name] = LayerAccumulatorState(
                count=reduced["count"],
                sum1=reduced["sum1"].clone(),
                sum2=reduced["sum2"].clone(),
                sum3=torch.zeros_like(
                    reduced["sum1"]
                ),  # not computed in reduce, zero-init
                sum4=reduced["sum4"].clone(),
                max_abs=reduced["max_abs"].clone(),
                outlier_channel_count=int(batch_outlier_count),
                outlier_batches_seen=1,
                median_abs_running=reduced["median_abs"].clone(),
            )
        else:
            # accumulate
            st = self._states[layer_name]

            st.count += reduced["count"]
            st.sum1 += reduced["sum1"]
            st.sum2 += reduced["sum2"]
            # sum3 stays zero (dont compute 3rd moment in reduce_for_stats)
            st.sum4 += reduced["sum4"]

            # element-wise maximum across all batches seen
            st.max_abs = torch.maximum(st.max_abs, reduced["max_abs"])

            # accumulate count of outlier channels across batches
            st.outlier_channel_count += int(batch_outlier_count)
            st.outlier_batches_seen += 1

            # use EMA with a decreasing learning rate alpha = 1/n since exact median requires all vals
            alpha = 1.0 / st.outlier_batches_seen
            if st.median_abs_running is not None:
                st.median_abs_running.lerp_(reduced["median_abs"], alpha)
            else:
                st.median_abs_running = reduced["median_abs"].clone()

    #  finalize
    def _finalize_layer(self, st: LayerAccumulatorState) -> Dict[str, Tensor]:
        """
        Converts raw sums into derived per-channel statistics:
        """
        eps = self._config.eps
        n = st.count

        # per-channel mean and variance from raw moments
        mean = st.sum1 / n
        var = (st.sum2 / n) - mean**2

        # Clamp variance to avoid negative values from numerical error
        var = var.clamp(min=0.0)

        raw_fourth = st.sum4 / n
        kurtosis = raw_fourth / (var**2 + eps)

        C = st.sum1.shape[0]
        total_channel_observations = C * st.outlier_batches_seen
        outlier_rate_scalar = st.outlier_channel_count / (
            total_channel_observations + eps
        )
        # Return as a tensor for consistency - broadcast to (C,) with uniform value
        outlier_rate = torch.full_like(st.sum1, outlier_rate_scalar)

        median_abs = (
            st.median_abs_running
            if st.median_abs_running is not None
            else torch.ones_like(st.max_abs)
        )
        dynamic_range = st.max_abs / (median_abs + eps)

        return {
            "mean": mean,
            "var": var,
            "kurtosis": kurtosis,
            "max_abs": st.max_abs,
            "outlier_rate": outlier_rate,
            "dynamic_range": dynamic_range,
        }

    # helpers
    @staticmethod
    def _first_tensor(x: Any) -> Optional[Tensor]:
        if torch.is_tensor(x):
            return x
        if isinstance(x, (tuple, list)):
            for item in x:
                t = ActivationStatsCollector._first_tensor(item)
                if t is not None:
                    return t
        if isinstance(x, dict):
            for item in x.values():
                t = ActivationStatsCollector._first_tensor(item)
                if t is not None:
                    return t
        return None


class HutchinsonTraceCalculator:

    def __init__(
        self, layers_per_group: Optional[int] = None, fix_batches: bool = False
    ) -> None:
        """
        Args :
            layers_per_group : Number of layers to analyze in one go.
                                - Do not want to analyze all layers in one go due to significant memory pressure of gradient and graph maintenance
            fix_batches      : Fix the calibration batches used across layer groups
                                - The Hessian trace varies across data batches
                                - Using different batches for different layer groups, you get a form of implicit averaging across calibration set
                                - Trace estimates are slightly noisier per-layer but more representative of the avg loss landscape
        """
        self.layers_per_group = layers_per_group or 1
        self.fix_batches = fix_batches

    def compute_trace(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
        n_samples: int = 200,
        group_size: Optional[int] = 5,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, Tuple[float, bool]]:

        model.eval()
        device = next(model.parameters()).device
        results = {}

        # Split layer names into groups of group_size
        if not group_size:
            group_size = len(layer_names) // self.layers_per_group

        groups = [
            layer_names[i : i + group_size]
            for i in range(0, len(layer_names), group_size)
        ]

        fixed_batch = {}
        if self.fix_batches:
            fixed_batch = next(iter(dataloader))

        for group_idx, group in enumerate(groups):
            # Fresh forward pass
            batch = fixed_batch if self.fix_batches else next(iter(dataloader))
            batch_samples = {k: v.to(device) for k, v in batch.items()}

            # build a brand-new computation graph
            outputs = model(**batch_samples)
            loss = outputs.loss

            # grads and HVPs for this group
            group_results = self._process_layer_group(
                model, loss, group, n_samples, dtype
            )
            results.update(group_results)

            # cleanup
            del loss, outputs, batch
            torch.cuda.empty_cache()

        return results

    def _process_layer_group(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        group: List[str],
        n_samples: int,
        dtype: torch.dtype,
    ) -> Dict[str, Tuple[float, bool]]:
        results = {}

        for layer_idx, name in enumerate(group):
            layer = model.get_submodule(name)
            params = [p for p in layer.parameters() if p.requires_grad]
            if not params:
                results[name] = (0.0, True)
                continue

            # create_graph=True: makes gradient itself differentiable
            # retain_graph=True: keeps forward graph for other layers
            n_params = sum(p.numel() for p in params)
            grads = torch.autograd.grad(
                loss, params, create_graph=True, retain_graph=True
            )

            # Hutchinson samples :
            trace_sum = 0.0
            for t in range(n_samples):

                # sample random rademacher vectors (z in [-1,1])
                z = [
                    2 * torch.randint(0, 2, size=p.shape, device=p.device, dtype=dtype)
                    - 1
                    for p in params
                ]

                # compute grad x rademacher scalar products
                gz = sum((g * _z).sum() for g, _z in zip(grads, z))

                # keep the graph after this grad call only if there are more samples or more layers.
                is_last_sample = t == n_samples - 1
                is_last_layer = layer_idx == len(group) - 1
                keep_graph = not (is_last_sample and is_last_layer)

                # Hv = d/dW (g.z) = H@z
                Hv = torch.autograd.grad(gz, params, retain_graph=keep_graph)

                # z^T H z
                sample = sum((_z * hv).sum().item() for _z, hv in zip(z, Hv))

                trace_sum += sample
                # free up
                del gz, Hv, z

            normalized = trace_sum / (n_samples * n_params)

            results[name] = (normalized, trace_sum > 0)
            # free up gradient tensors for this layer
            del grads

        return results
