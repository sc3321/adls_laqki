from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from .models import ActivationSpec, StatsConfig, LayerAccumulatorState, Tensor, ModuleOutput

class ActivationStatsCollector:
    """
    Collects activation statistics for specified layers over many batches.

    Intended flow:
      1) collector = ActivationStatsCollector(model, specs, config)
      2) with collector:
             for batch in dataloader:
                 outputs = model(...)
                 collector.on_batch_end()  # optional hook for batch-level updates
      3) results = collector.finalize()

    Notes:
    - Hooks do the per-batch reduction and update accumulator state.
    - No full activation tensors are stored across batches.
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
        where each metrics dict contains per-channel tensors (shape C,) and optionally layer-level scalars.
        """
        results: Dict[str, Dict[str, Tensor]] = {}
        for layer_name, st in self._states.items():
            results[layer_name] = self._finalize_layer(st)
        return results


    # setup
    def _validate_and_register(self) -> None:
        """
        Resolves layers and registers hooks
        Fail early if any names are invalid (strict=True)
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
        Choose the tensor to treat as activation.
        - If spec.output_extractor is provided, call it
        - Otherwise: choose the first tensor found in the output structure
        """
        if spec.output_extractor is not None:
            t = spec.output_extractor(output)
            if not torch.is_tensor(t):
                raise TypeError(f"Extractor for {spec.layer_name} did not return Tensor.")
            return t

        # Default extractor: find first tensor
        t = self._first_tensor(output)
        if t is None:
            raise TypeError(f"No Tensor output found for layer {spec.layer_name}.")
        return t

    def _reduce_for_stats(self, act: Tensor, spec: ActivationSpec) -> Dict[str, Tensor]:
        """
        Core per-batch computation.

        Input: act with arbitrary shape, with channel_dim specifying channels.
        Output: dict of per-channel vectors (shape C,).

        You will implement:
          - sum1, sum2, sum3, sum4 per channel (or means)
          - max_abs per channel
          - batch_std per channel (needed for outlier test)
          - median_abs per channel (exact/approx depending on config)
        """
        # Decide reduce_dims
        reduce_dims = spec.reduce_dims
        if reduce_dims is None:
            reduce_dims = tuple(d for d in range(act.ndim) if d != spec.channel_dim)

        # IMPORTANT: reduction should be per-channel.
        # compute these tensors:
        #   sum1_c, sum2_c, sum3_c, sum4_c, max_abs_c, std_c, median_abs_c
        #
        # Return a dict like:
        #   {
        #       "count": tensor_scalar_or_int,
        #       "sum1": (C,),
        #       "sum2": (C,),
        #       "sum3": (C,),
        #       "sum4": (C,),
        #       "max_abs": (C,),
        #       "std": (C,),
        #       "median_abs": (C,) or optional,
        #   }
        raise NotImplementedError

    # state update
    def _update_state(self, layer_name: str, reduced: Dict[str, Tensor]) -> None:
        """
        Updates/initializes LayerAccumulatorState for a layer.

        This method:
          - initializes state tensors on first batch
          - adds sums
          - updates running max
          - updates outlier counters using reduced["max_abs"] and reduced["std"]
          - updates median tracker (if you implement it)
        """
        raise NotImplementedError

    # finalize
    def _finalize_layer(self, st: LayerAccumulatorState) -> Dict[str, Tensor]:
        """
        Converts raw sums into:
          - mean per channel
          - var per channel
          - kurtosis per channel (excess or not; define it)
          - max_abs per channel
          - outlier_rate (scalar or per-channel; define it)
          - dynamic_range per channel (max_abs / (median_abs + eps))
        """
        raise NotImplementedError

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