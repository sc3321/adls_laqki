import logging
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizer

from .models import CalibrationConfig

log = logging.getLogger(__name__)


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

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Implementation notes:
        # - Use datasets.load_dataset() with streaming=True for large corpora
        # - Tokenize with tokenizer(text, max_length=config.sequence_length,
        #     truncation=True, return_tensors="pt")
        # - Split into train/val using config.validation_split
        # - Cache tokenized data to config.cache_dir for reuse
        # - If multiple datasets specified, interleave with config.dataset_weights

    def get_dataloader(self, batch_size: int = 4) -> DataLoader:
        """
        Training split DataLoader for SensitivityScorer.

        Returns tokenized calibration samples as a DataLoader.
        Each batch contains input_ids and attention_mask tensors.

        Implementation steps:
        1. Load datasets from config.datasets using HuggingFace datasets
        2. Interleave multiple datasets with config.dataset_weights
        3. Tokenize to config.sequence_length
        4. Take config.num_samples * (1 - config.validation_split) samples
        5. Return DataLoader with specified batch_size
        """
        self._ensure_prepared()
        return DataLoader(self._dataset, batch_size=batch_size, shuffle=False)

    def get_validation_dataloader(self, batch_size: int = 4) -> DataLoader:
        """
        Held-out split DataLoader for ValidationEngine perplexity screening.

        Same format as get_dataloader() but uses the validation split
        (config.validation_split fraction of total samples).
        """
        self._ensure_prepared()
        return DataLoader(self._val_dataset, batch_size=batch_size, shuffle=False)

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
        from qpe.utils.model_utils import get_quantizable_layers

        self._ensure_prepared()
        dataloader = self.get_dataloader(batch_size=4)

        group_size = int(getattr(self.config, "group_size", 128))
        layer_names = get_quantizable_layers(model)

        accum: Dict[str, torch.Tensor] = {}
        n_samples = 0

        handles = []

        def _make_hook(layer_name: str):
            def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if not inputs:
                    return
                activation = inputs[0]
                if not torch.is_tensor(activation):
                    return

                if activation.ndim == 1:
                    activation = activation.unsqueeze(0)
                flat = activation.detach().float().reshape(-1, activation.shape[-1])
                feature_energy = (flat**2).sum(dim=0)

                n_features = feature_energy.numel()
                n_groups = (n_features + group_size - 1) // group_size
                pad = n_groups * group_size - n_features
                if pad > 0:
                    feature_energy = torch.nn.functional.pad(feature_energy, (0, pad))

                grouped = feature_energy.view(n_groups, group_size).sum(dim=1)
                if layer_name not in accum:
                    accum[layer_name] = grouped.to(dtype=torch.float64, device="cpu")
                else:
                    accum[layer_name] += grouped.to(dtype=torch.float64, device="cpu")

            return _hook

        for layer_name in layer_names:
            try:
                module = model.get_submodule(layer_name)
            except AttributeError:
                continue
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(_make_hook(layer_name)))

        was_training = model.training
        model.eval()

        try:
            device = next(model.parameters()).device
            with torch.no_grad():
                for batch in dataloader:
                    batch_on_device = {
                        key: value.to(device)
                        for key, value in batch.items()
                        if torch.is_tensor(value)
                    }
                    model(**batch_on_device)
                    n_samples += int(batch_on_device["input_ids"].shape[0])
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                model.train()

        if n_samples == 0:
            return {name: np.zeros(0, dtype=np.float64) for name in layer_names}

        result: Dict[str, np.ndarray] = {}
        for name in layer_names:
            if name not in accum:
                result[name] = np.zeros(0, dtype=np.float64)
            else:
                result[name] = (accum[name] / float(n_samples)).numpy()
        return result

    def _ensure_prepared(self) -> None:
        if self._dataset is not None and self._val_dataset is not None:
            return
        train, val = self._build_tokenized_splits()
        self._dataset = train
        self._val_dataset = val

    def _build_tokenized_splits(self) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
        raw_texts = self._collect_raw_text_samples()
        tokenized = [self._tokenize_text(text) for text in raw_texts if text and text.strip()]

        if not tokenized:
            raise RuntimeError(
                "No calibration samples were produced from configured datasets. "
                "Check `calibration.datasets` and internet access."
            )

        n_total = len(tokenized)
        n_val = int(round(n_total * self.config.validation_split))
        n_val = max(1, min(n_val, n_total - 1)) if n_total > 1 else 0
        n_train = n_total - n_val

        train = tokenized[:n_train]
        val = tokenized[n_train:]
        if not val:
            val = train[:1]
        return train, val

    def _collect_raw_text_samples(self) -> list[str]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "CalibrationDataManager requires HuggingFace datasets. "
                "Install with `pip install datasets`."
            ) from exc

        names = list(self.config.datasets)
        if not names:
            raise ValueError("CalibrationConfig.datasets cannot be empty")

        weights = list(self.config.dataset_weights)
        if len(weights) != len(names):
            weights = [1.0 / len(names)] * len(names)

        total_target = int(self.config.num_samples)
        allocations = [max(1, int(round(total_target * w))) for w in weights]

        diff = total_target - sum(allocations)
        i = 0
        while diff != 0 and allocations:
            idx = i % len(allocations)
            if diff > 0:
                allocations[idx] += 1
                diff -= 1
            elif allocations[idx] > 1:
                allocations[idx] -= 1
                diff += 1
            i += 1

        collected: list[str] = []
        for dataset_name, target_n in zip(names, allocations):
            resolved_name, resolved_kwargs = self._resolve_dataset(dataset_name)
            try:
                ds = load_dataset(
                    resolved_name,
                    split="train",
                    streaming=True,
                    cache_dir=self.config.cache_dir,
                    **resolved_kwargs,
                )
            except Exception:
                log.warning("Failed to load dataset '%s'; skipping", dataset_name)
                continue

            sampled = ds.shuffle(seed=self.config.seed, buffer_size=max(1_000, target_n * 4))
            count = 0
            for record in sampled:
                text = self._extract_text(record)
                if text is None:
                    continue
                collected.append(text)
                count += 1
                if count >= target_n:
                    break

        return collected[: self.config.num_samples]

    def _resolve_dataset(self, name: str) -> tuple[str, dict[str, Any]]:
        lowered = name.lower()
        if lowered == "c4":
            return "allenai/c4", {"name": "en"}
        if lowered == "pile":
            return "monology/pile-uncopyrighted", {}
        return name, {}

    def _extract_text(self, record: dict[str, Any]) -> str | None:
        candidates = ["text", "content", "document", "ctx", "prompt"]
        for key in candidates:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in record.values():
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.sequence_length,
            padding="max_length",
            return_tensors="pt",
        )
        item: dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        item["labels"] = item["input_ids"].clone()
        return item
