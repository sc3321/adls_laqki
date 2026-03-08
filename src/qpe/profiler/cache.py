"""Hierarchical disk cache for per-layer profiling results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProfileCache:
    """
    Filesystem cache organized by version, GPU, model, layer, batch, seq_len, and regime.

    Directory layout:

        root_dir/
          v{qpe_version}/
            {gpu_name}/
              _precisions.json
              {model_id}/
                _status.json
                {layer_name}/
                  _meta.json
                  bs_{batch_size}/
                    seq_{seq_len}/
                      {regime}/
                        {precision}.json

    Status semantics:
      - A measurement point is identified by (batch_size, seq_len, regime).
      - A point is complete when all expected precision files exist.
      - A layer is complete when all discovered points are complete.
      - A model is complete when all discovered layers are complete.
    """

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
        seq_len: int,
        regime: str,
        precision: str,
    ) -> dict | None:
        """Return a cached profile payload or None on cache miss."""
        profile_path = self._profile_path(
            model_id=model_id,
            layer_name=layer_name,
            batch_size=batch_size,
            seq_len=seq_len,
            regime=regime,
            precision=precision,
        )
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
        seq_len: int,
        regime: str,
        precision: str,
        data: dict,
        layer_meta: dict[str, Any],
    ) -> None:
        """Write a profile payload and update completion status."""
        model_dir = self._model_dir(model_id)
        layer_dir = self._layer_dir(model_id, layer_name)
        point_dir = self._point_dir(
            model_id=model_id,
            layer_name=layer_name,
            batch_size=batch_size,
            seq_len=seq_len,
            regime=regime,
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        layer_dir.mkdir(parents=True, exist_ok=True)
        point_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(layer_dir / "_meta.json", layer_meta)

        payload = dict(data)
        payload["_layer_meta"] = layer_meta
        self._write_json(
            self._profile_path(
                model_id=model_id,
                layer_name=layer_name,
                batch_size=batch_size,
                seq_len=seq_len,
                regime=regime,
                precision=precision,
            ),
            payload,
        )

        self._update_status(
            model_id=model_id,
            layer_name=layer_name,
            batch_size=batch_size,
            seq_len=seq_len,
            regime=regime,
        )

    def is_batch_complete(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        seq_len: int,
        regime: str,
    ) -> bool:
        """
        Return True if the specific (batch_size, seq_len, regime) point
        has all expected precision profiles.
        """
        status = self.verify(model_id)
        layer_entry = status.get("layers", {}).get(
            self._sanitize_layer_name(layer_name), {}
        )
        point_key = self._point_key(batch_size=batch_size, seq_len=seq_len, regime=regime)
        point_entry = layer_entry.get("points", {}).get(point_key, {})
        return bool(point_entry.get("complete", False))

    def is_layer_complete(self, model_id: str, layer_name: str) -> bool:
        """Return True if all discovered measurement points for this layer are complete."""
        status = self.verify(model_id)
        layer_entry = status.get("layers", {}).get(
            self._sanitize_layer_name(layer_name), {}
        )
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
            if layer_dir.name.startswith("_"):
                continue

            sanitized_layer_name = layer_dir.name
            points: dict[str, dict[str, Any]] = {}

            for batch_dir in sorted(layer_dir.iterdir()):
                if not batch_dir.is_dir() or not batch_dir.name.startswith("bs_"):
                    continue

                batch_size = self._parse_prefixed_int(batch_dir.name, prefix="bs_")
                if batch_size is None:
                    continue

                for seq_dir in sorted(batch_dir.iterdir()):
                    if not seq_dir.is_dir() or not seq_dir.name.startswith("seq_"):
                        continue

                    seq_len = self._parse_prefixed_int(seq_dir.name, prefix="seq_")
                    if seq_len is None:
                        continue

                    for regime_dir in sorted(seq_dir.iterdir()):
                        if not regime_dir.is_dir():
                            continue

                        regime = regime_dir.name
                        profiled_precisions = self._profiled_precisions(regime_dir)
                        point_complete = self._has_all_precisions(profiled_precisions)
                        point_key = self._point_key(
                            batch_size=batch_size,
                            seq_len=seq_len,
                            regime=regime,
                        )

                        points[point_key] = {
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "regime": regime,
                            "profiled": profiled_precisions,
                            "complete": point_complete,
                        }

            layer_complete = bool(points) and all(
                entry.get("complete", False) for entry in points.values()
            )

            status["layers"][sanitized_layer_name] = {
                "points": points,
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

    @staticmethod
    def _sanitize_regime(regime: str) -> str:
        return regime.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")

    @staticmethod
    def _point_key(batch_size: int, seq_len: int, regime: str) -> str:
        return f"bs_{batch_size}|seq_{seq_len}|regime_{regime}"

    @staticmethod
    def _parse_prefixed_int(name: str, prefix: str) -> int | None:
        if not name.startswith(prefix):
            return None
        suffix = name[len(prefix) :]
        return int(suffix) if suffix.isdigit() else None

    def _model_dir(self, model_id: str) -> Path:
        return self.gpu_dir / self._sanitize_model_id(model_id)

    def _layer_dir(self, model_id: str, layer_name: str) -> Path:
        return self._model_dir(model_id) / self._sanitize_layer_name(layer_name)

    def _batch_dir(self, model_id: str, layer_name: str, batch_size: int) -> Path:
        return self._layer_dir(model_id, layer_name) / f"bs_{batch_size}"

    def _seq_dir(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        seq_len: int,
    ) -> Path:
        return self._batch_dir(model_id, layer_name, batch_size) / f"seq_{seq_len}"

    def _point_dir(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        seq_len: int,
        regime: str,
    ) -> Path:
        return (
            self._seq_dir(model_id, layer_name, batch_size, seq_len)
            / self._sanitize_regime(regime)
        )

    def _profile_path(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        seq_len: int,
        regime: str,
        precision: str,
    ) -> Path:
        return (
            self._point_dir(model_id, layer_name, batch_size, seq_len, regime)
            / f"{precision}.json"
        )

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

    def _update_status(
        self,
        model_id: str,
        layer_name: str,
        batch_size: int,
        seq_len: int,
        regime: str,
    ) -> None:
        status = self._load_status(model_id)
        sanitized_layer_name = self._sanitize_layer_name(layer_name)
        layer_entry = status["layers"].setdefault(
            sanitized_layer_name,
            {"points": {}, "complete": False},
        )

        point_dir = self._point_dir(
            model_id=model_id,
            layer_name=layer_name,
            batch_size=batch_size,
            seq_len=seq_len,
            regime=regime,
        )
        profiled = self._profiled_precisions(point_dir)
        point_complete = self._has_all_precisions(profiled)
        point_key = self._point_key(batch_size=batch_size, seq_len=seq_len, regime=regime)

        layer_entry["points"][point_key] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "regime": regime,
            "profiled": profiled,
            "complete": point_complete,
        }

        layer_entry["complete"] = bool(layer_entry["points"]) and all(
            entry.get("complete", False)
            for entry in layer_entry["points"].values()
        )
        status["model_complete"] = bool(status["layers"]) and all(
            entry.get("complete", False) for entry in status["layers"].values()
        )
        self._save_status(model_id, status)

    def _has_all_precisions(self, profiled: list[str]) -> bool:
        return set(self.supported_precisions).issubset(set(profiled))

    def _profiled_precisions(self, point_dir: Path) -> list[str]:
        if not point_dir.exists():
            return []

        discovered: set[str] = set()
        for path in point_dir.iterdir():
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
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
