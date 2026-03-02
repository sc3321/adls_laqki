"""Hierarchical disk cache for per-layer profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProfileCache:
    """Filesystem cache organized by version, GPU, model, layer, and batch."""

    def __init__(
        self,
        root_dir: str = ".qpe_cache/profiles",
        qpe_version: str = "2.0",
        gpu_name: str = "unknown",
        supported_precisions: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.qpe_version = str(qpe_version)
        self.supported_precisions = self._normalize_precisions(supported_precisions)

        self.version_dir = self.root_dir / f"v{self.qpe_version}"
        self.gpu_dir = self.version_dir / self._sanitize_gpu_name(gpu_name)
        self.gpu_dir.mkdir(parents=True, exist_ok=True)

        self._precisions_path = self.gpu_dir / "_precisions.json"
        self._write_json(self._precisions_path, self.supported_precisions)

    def get(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        precision: str,
    ) -> dict | None:
        """Return a cached profile payload or None on cache miss."""
        profile_path = self._profile_path(model_id, layer_name, batch_size, precision)
        if not profile_path.exists():
            return None

        try:
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        payload.pop("_layer_meta", None)
        return payload

    def put(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        precision: str,
        data: dict,
        layer_meta: dict[str, Any],
    ) -> None:
        """Write a profile payload and update completion status."""
        model_dir = self._model_dir(model_id)
        layer_dir = self._layer_dir(model_id, layer_name)
        batch_dir = self._batch_dir(model_id, layer_name, batch_size)

        model_dir.mkdir(parents=True, exist_ok=True)
        layer_dir.mkdir(parents=True, exist_ok=True)
        batch_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(layer_dir / "_meta.json", layer_meta)

        payload = dict(data)
        payload["_layer_meta"] = layer_meta
        self._write_json(
            self._profile_path(model_id, layer_name, batch_size, precision), payload
        )
        self._update_status(model_id, layer_name, batch_size)

    def is_batch_complete(self, model_id: str, layer_name: str, batch_size: int) -> bool:
        """Return True if a batch has all expected precision profiles."""
        status = self.verify(model_id)
        layer_entry = status.get("layers", {}).get(layer_name, {})
        batch_entry = layer_entry.get("batch_sizes", {}).get(str(batch_size), {})
        return bool(batch_entry.get("complete", False))

    def is_layer_complete(self, model_id: str, layer_name: str) -> bool:
        """Return True if all discovered batch directories are complete."""
        status = self.verify(model_id)
        layer_entry = status.get("layers", {}).get(layer_name, {})
        return bool(layer_entry.get("complete", False))

    def is_model_complete(self, model_id: str) -> bool:
        """Return True if every discovered layer is complete."""
        status = self.verify(model_id)
        return bool(status.get("model_complete", False))

    def verify(self, model_id: str) -> dict[str, Any]:
        """Rebuild model status by scanning the filesystem."""
        status = self._new_status()
        model_dir = self._model_dir(model_id)
        if not model_dir.exists():
            self._save_status(model_id, status)
            return status

        for layer_dir in sorted(model_dir.iterdir()):
            if not layer_dir.is_dir():
                continue

            layer_name = layer_dir.name
            batch_map: dict[str, dict[str, Any]] = {}

            for batch_dir in sorted(layer_dir.iterdir()):
                if not batch_dir.is_dir() or not batch_dir.name.startswith("bs_"):
                    continue

                batch_key = batch_dir.name.removeprefix("bs_")
                profiled_precisions = self._profiled_precisions(batch_dir)
                batch_complete = self._has_all_precisions(profiled_precisions)
                batch_map[batch_key] = {
                    "profiled": profiled_precisions,
                    "complete": batch_complete,
                }

            layer_complete = bool(batch_map) and all(
                entry.get("complete", False) for entry in batch_map.values()
            )
            status["layers"][layer_name] = {
                "batch_sizes": batch_map,
                "complete": layer_complete,
            }

        status["model_complete"] = bool(status["layers"]) and all(
            layer.get("complete", False) for layer in status["layers"].values()
        )
        self._save_status(model_id, status)
        return status

    @staticmethod
    def _sanitize_gpu_name(gpu_name: str) -> str:
        return (
            gpu_name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )

    @staticmethod
    def _sanitize_model_id(model_id: str) -> str:
        return (
            model_id.replace("/", "--")
            .replace("\\", "--")
            .replace(" ", "_")
            .replace(":", "_")
        )

    @staticmethod
    def _sanitize_layer_name(layer_name: str) -> str:
        return layer_name.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _model_dir(self, model_id: str) -> Path:
        return self.gpu_dir / self._sanitize_model_id(model_id)

    def _layer_dir(self, model_id: str, layer_name: str) -> Path:
        return self._model_dir(model_id) / self._sanitize_layer_name(layer_name)

    def _batch_dir(self, model_id: str, layer_name: str, batch_size: int) -> Path:
        return self._layer_dir(model_id, layer_name) / f"bs_{batch_size}"

    def _profile_path(
        self, model_id: str, layer_name: str, batch_size: int, precision: str
    ) -> Path:
        return self._batch_dir(model_id, layer_name, batch_size) / f"{precision}.json"

    def _status_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "_status.json"

    def _new_status(self) -> dict[str, Any]:
        return {
            "expected_precisions": list(self.supported_precisions),
            "layers": {},
            "model_complete": False,
        }

    def _load_status(self, model_id: str) -> dict[str, Any]:
        path = self._status_path(model_id)
        if not path.exists():
            return self._new_status()

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self.verify(model_id)

        if not isinstance(loaded, dict):
            return self.verify(model_id)

        loaded["expected_precisions"] = list(self.supported_precisions)
        loaded.setdefault("layers", {})
        loaded.setdefault("model_complete", False)
        return loaded

    def _save_status(self, model_id: str, status: dict[str, Any]) -> None:
        self._write_json(self._status_path(model_id), status)

    def _update_status(self, model_id: str, layer_name: str, batch_size: int) -> None:
        status = self._load_status(model_id)
        layer_entry = status["layers"].setdefault(
            layer_name,
            {"batch_sizes": {}, "complete": False},
        )

        batch_dir = self._batch_dir(model_id, layer_name, batch_size)
        profiled = self._profiled_precisions(batch_dir)
        batch_complete = self._has_all_precisions(profiled)
        layer_entry["batch_sizes"][str(batch_size)] = {
            "profiled": profiled,
            "complete": batch_complete,
        }
        layer_entry["complete"] = bool(layer_entry["batch_sizes"]) and all(
            entry.get("complete", False)
            for entry in layer_entry["batch_sizes"].values()
        )
        status["model_complete"] = bool(status["layers"]) and all(
            entry.get("complete", False) for entry in status["layers"].values()
        )
        self._save_status(model_id, status)

    def _has_all_precisions(self, profiled: list[str]) -> bool:
        return set(self.supported_precisions).issubset(set(profiled))

    def _profiled_precisions(self, batch_dir: Path) -> list[str]:
        if not batch_dir.exists():
            return []

        discovered: set[str] = set()
        for path in batch_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix != ".json":
                continue
            if path.name.startswith("_"):
                continue
            discovered.add(path.stem)

        ordered = [p for p in self.supported_precisions if p in discovered]
        extras = sorted(discovered.difference(self.supported_precisions))
        return ordered + extras

    @staticmethod
    def _normalize_precisions(precisions: list[str] | None) -> list[str]:
        if not precisions:
            return []
        deduped: list[str] = []
        for precision in precisions:
            if precision not in deduped:
                deduped.append(precision)
        return deduped

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

