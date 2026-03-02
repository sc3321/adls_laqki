from __future__ import annotations
import torch.nn as nn
from typing import List, Callable

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


Tensor = torch.Tensor

# Common module outputs (some layers return tuples/dicts)
ModuleOutput = Union[Tensor, Tuple[Any, ...], List[Any], Dict[str, Any]]


@dataclass(frozen=True)
class ActivationSpec:
    """
    Describes what to capture and how to interpret it.
    """
    layer_name: str
    # Which dimension is the channel dimension (hidden/features).
    # For transformer activations (B, S, H), channel_dim = -1
    channel_dim: int = -1

    # Reduce over all dims except channel_dim.
    # If None: infer as : all dims except channel_dim
    reduce_dims: Optional[Tuple[int, ...]] = None

    # If the module output is a structure, choose which tensor to extract.
    output_extractor: Optional[Callable[[ModuleOutput], Tensor]] = None


@dataclass(frozen=True)
class StatsConfig:
    """
    Controls numeric behavior / thresholds.
    """
    eps: float = 1e-8
    outlier_sigma: float = 6.0

    # Whether to compute exact median (expensive). If False, might sample/approx
    exact_median: bool = False

    # If sample for median/quantiles, define sampling strategy.
    median_sample_max_elements: int = 200_000

    accumulator_dtype: torch.dtype = torch.float64


@dataclass
class LayerAccumulatorState:
    """
    Streaming raw accumulators for one layer
    All tensors here should be shape: (C,) where C = num channels.
    """
    count: int  # number of elements per channel accumulated (tokens * batch, etc.)

    sum1: Tensor
    sum2: Tensor
    sum3: Tensor
    sum4: Tensor

    max_abs: Tensor

    # Outlier tracking (def: channels where max > sigma*std)
    outlier_channel_count: int
    outlier_batches_seen: int

    # Dynamic range ingredients.
    median_abs_running: Optional[Tensor] = None  # (C,) if implement per-channel

