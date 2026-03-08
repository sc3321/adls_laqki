import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizer

from .models import CalibrationConfig

log = logging.getLogger(__name__)


def _collate_classification(batch):
    """Rename SST-2's 'label' key to 'labels' expected by HuggingFace models."""
    from torch.utils.data.dataloader import default_collate
    collated = default_collate(batch)
    if "label" in collated:
        collated["labels"] = collated.pop("label")
    return collated


class CalibrationDataManager:
    """
    Manages calibration and validation data for the QPE pipeline.

    Responsibilities:
    - Load and cache calibration datasets from HuggingFace
    - Provide DataLoaders for scorer (calibration split) and validator
      (held-out validation split)
    - Compute importance matrices for llama.cpp imatrix export
    """

    def __init__(self, config: CalibrationConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self._dataset = None
        self._val_dataset = None

        # Implementation notes:
        # - Use datasets.load_dataset() with streaming=True for large corpora
        # - Tokenize with tokenizer(text, max_length=config.sequence_length,
        #     truncation=True, return_tensors="pt")
        # - Split into train/val using config.validation_split
        # - Cache tokenized data to config.cache_dir for reuse
        # - If multiple datasets specified, interleave with config.dataset_weights

    def _is_sst2(self) -> bool:
        return any("sst2" in d.lower() or "sst-2" in d.lower() for d in self.config.datasets)

    def _load_sst2(self, split: str):
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split=split)

        def tokenize(batch):
            return self.tokenizer(
                batch["sentence"],
                max_length=self.config.sequence_length,
                truncation=True,
                padding="max_length",
            )

        ds = ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return ds

    def get_dataloader(self, batch_size: int = 4) -> DataLoader:
        """
        Training split DataLoader for SensitivityScorer.

        For SST-2 (when config.datasets contains "sst2"), returns batches of
        {"input_ids", "attention_mask", "labels"} using the training split.

        Takes config.num_samples * (1 - config.validation_split) samples.
        """
        if self._is_sst2():
            ds = self._load_sst2("train")
            n = int(len(ds) * (1 - self.config.validation_split))
            n = min(n, self.config.num_samples)
            return DataLoader(
                ds.select(range(n)),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_classification,
            )
        raise NotImplementedError(
            f"CalibrationDataManager.get_dataloader() is not implemented for "
            f"datasets={self.config.datasets}. Currently supported: ['sst2']."
        )

    def get_validation_dataloader(self, batch_size: int = 4) -> DataLoader:
        """
        Held-out validation split DataLoader for ValidationEngine.

        For SST-2, uses the official GLUE validation split (no overlap with training).
        """
        if self._is_sst2():
            ds = self._load_sst2("validation")
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_classification,
            )
        raise NotImplementedError(
            f"CalibrationDataManager.get_validation_dataloader() is not implemented for "
            f"datasets={self.config.datasets}. Currently supported: ['sst2']."
        )

    def compute_importance_matrix(self, model: nn.Module) -> Dict[str, np.ndarray]:
        """
        Sum-of-squared activations per weight group (llama.cpp imatrix approach).

        Used by ConfigurationExporter for importance-weighted quantization.

        Algorithm:
        1. Register forward hooks on all quantizable Linear layers
        2. For each calibration batch:
           a. Forward pass through the model
           b. For each hooked layer, capture input activations A
           c. Reshape A into weight groups of size config.group_size
           d. Accumulate: importance[layer][group] += sum(A_group ** 2)
        3. Normalize by number of samples
        4. Return {layer_name: np.ndarray of shape (num_groups,)}
        """
        raise NotImplementedError(
            "CalibrationDataManager.compute_importance_matrix() requires "
            "a forward pass over calibration data."
        )
