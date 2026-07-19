from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


DEFAULT_NUMPY_WEIGHTS = Path(__file__).resolve().with_name("perception_model.npz")
_BATCH_NORM_EPS = 1e-5


class NumpyPerceptionModel:
    """NumPy inference for the submitted CNN when PyTorch is unavailable.

    The implementation mirrors the trained PyTorch network. It is a runtime
    backend, not a color/template recognizer, and consumes the same learned
    convolution weights as the PyTorch path.
    """

    is_numpy_backend = True

    def __init__(self, parameters: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
        self.parameters = parameters
        self.exit_type_head_available = bool(metadata.get("exit_type_head_available", True))
        self.has_trained_chest_state_head = bool(
            metadata.get("has_trained_chest_state_head", True)
        )
        self.has_trained_offset_head = bool(metadata.get("has_trained_offset_head", False))
        self.has_trained_player_offset_head = bool(
            metadata.get("has_trained_player_offset_head", self.has_trained_offset_head)
        )
        self.has_trained_refine_head = bool(metadata.get("has_trained_refine_head", True))
        self.has_trained_occupancy_head = bool(
            metadata.get("has_trained_occupancy_head", True)
        )

    def __call__(self, image_chw: np.ndarray) -> dict[str, np.ndarray]:
        image = np.asarray(image_chw, dtype=np.float32)
        if image.shape != (3, 128, 160):
            raise ValueError(f"NumPy CNN expects CHW=(3, 128, 160), got {image.shape}")

        high_resolution = self._encoder_block(image, "encoder.0", "encoder.1")
        high_resolution = self._encoder_block(
            high_resolution, "encoder.3", "encoder.4"
        )
        mid_resolution = self._max_pool_2x2(high_resolution)
        mid_resolution = self._encoder_block(
            mid_resolution, "encoder.7", "encoder.8"
        )
        features = self._max_pool_2x2(mid_resolution)
        features = self._encoder_block(features, "encoder.11", "encoder.12")
        features = self._encoder_block(features, "encoder.14", "encoder.15")

        pooled = self._adaptive_average_pool(features, 8, 10)
        tile_logits = self._head(pooled, "tile_head.1", "tile_head.3")
        exit_type_logits = self._head(
            pooled, "exit_type_head.1", "exit_type_head.3"
        )
        chest_state_logits = self._head(
            pooled, "chest_state_head.1", "chest_state_head.3"
        )
        occupancy_logits = self._head(
            pooled, "occupancy_head.1", "occupancy_head.3"
        )

        heatmap_logits = self._head(
            features, "heatmap_head.0", "heatmap_head.2"
        )
        heatmap_logits = self._resize_bilinear(heatmap_logits, 128, 160)
        if self.has_trained_refine_head:
            resized_mid = self._resize_bilinear(mid_resolution, 128, 160)
            refine_features = np.concatenate(
                (high_resolution, resized_mid, heatmap_logits), axis=0
            )
            player_refinement = self._refine_head(refine_features, "player_refine_head")
            monster_refinement = self._refine_head(
                refine_features, "monster_refine_head"
            )
            heatmap_logits = heatmap_logits + np.concatenate(
                (player_refinement, monster_refinement), axis=0
            )

        offset_logits = self._head(features, "offset_head.0", "offset_head.2")
        return {
            "tile_logits": tile_logits,
            "exit_type_logits": exit_type_logits,
            "chest_state_logits": chest_state_logits,
            "heatmap_logits": heatmap_logits,
            "offset_logits": offset_logits,
            "occupancy_logits": occupancy_logits,
        }

    def _parameter(self, name: str) -> np.ndarray:
        try:
            return self.parameters[name]
        except KeyError as exc:
            raise RuntimeError(f"NumPy perception weights missing parameter {name!r}") from exc

    def _encoder_block(self, value: np.ndarray, conv: str, batch_norm: str) -> np.ndarray:
        value = self._conv2d(value, conv)
        value = self._batch_norm(value, batch_norm)
        return np.maximum(value, 0.0, dtype=np.float32)

    def _head(self, value: np.ndarray, first: str, last: str) -> np.ndarray:
        value = np.maximum(self._conv2d(value, first), 0.0, dtype=np.float32)
        return self._conv2d(value, last)

    def _refine_head(self, value: np.ndarray, prefix: str) -> np.ndarray:
        value = np.maximum(
            self._conv2d(value, f"{prefix}.0"), 0.0, dtype=np.float32
        )
        value = np.maximum(
            self._conv2d(value, f"{prefix}.2"), 0.0, dtype=np.float32
        )
        return self._conv2d(value, f"{prefix}.4")

    def _conv2d(self, value: np.ndarray, prefix: str) -> np.ndarray:
        weight = self._parameter(f"{prefix}.weight")
        bias = self._parameter(f"{prefix}.bias")
        kernel_height, kernel_width = weight.shape[-2:]
        pad_height = kernel_height // 2 if kernel_height > 1 else 0
        pad_width = kernel_width // 2 if kernel_width > 1 else 0
        padded = np.pad(
            value,
            ((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
            mode="constant",
        )
        windows = sliding_window_view(
            padded, (kernel_height, kernel_width), axis=(1, 2)
        )
        columns = np.ascontiguousarray(
            windows.transpose(1, 2, 0, 3, 4)
        ).reshape(-1, weight.shape[1] * kernel_height * kernel_width)
        output = columns @ weight.reshape(weight.shape[0], -1).T
        output += bias
        return output.T.reshape(weight.shape[0], value.shape[1], value.shape[2]).astype(
            np.float32, copy=False
        )

    def _batch_norm(self, value: np.ndarray, prefix: str) -> np.ndarray:
        weight = self._parameter(f"{prefix}.weight")[:, None, None]
        bias = self._parameter(f"{prefix}.bias")[:, None, None]
        mean = self._parameter(f"{prefix}.running_mean")[:, None, None]
        variance = self._parameter(f"{prefix}.running_var")[:, None, None]
        scale = weight / np.sqrt(variance + _BATCH_NORM_EPS)
        return ((value - mean) * scale + bias).astype(np.float32, copy=False)

    @staticmethod
    def _max_pool_2x2(value: np.ndarray) -> np.ndarray:
        channels, height, width = value.shape
        return value.reshape(channels, height // 2, 2, width // 2, 2).max(
            axis=(2, 4)
        )

    @staticmethod
    def _adaptive_average_pool(
        value: np.ndarray, output_height: int, output_width: int
    ) -> np.ndarray:
        channels, height, width = value.shape
        if height % output_height or width % output_width:
            raise ValueError(
                "NumPy CNN only supports evenly divisible adaptive pooling shapes"
            )
        return value.reshape(
            channels,
            output_height,
            height // output_height,
            output_width,
            width // output_width,
        ).mean(axis=(2, 4), dtype=np.float32)

    @staticmethod
    def _resize_bilinear(
        value: np.ndarray, output_height: int, output_width: int
    ) -> np.ndarray:
        _, input_height, input_width = value.shape
        source_y = (
            (np.arange(output_height, dtype=np.float32) + 0.5)
            * (input_height / output_height)
            - 0.5
        )
        source_x = (
            (np.arange(output_width, dtype=np.float32) + 0.5)
            * (input_width / output_width)
            - 0.5
        )
        y0 = np.floor(source_y).astype(np.int64)
        x0 = np.floor(source_x).astype(np.int64)
        y1 = y0 + 1
        x1 = x0 + 1
        y_weight = (source_y - y0).astype(np.float32)
        x_weight = (source_x - x0).astype(np.float32)
        y0 = np.clip(y0, 0, input_height - 1)
        y1 = np.clip(y1, 0, input_height - 1)
        x0 = np.clip(x0, 0, input_width - 1)
        x1 = np.clip(x1, 0, input_width - 1)

        upper = value[:, y0, :] * (1.0 - y_weight)[None, :, None]
        upper += value[:, y1, :] * y_weight[None, :, None]
        output = upper[:, :, x0] * (1.0 - x_weight)[None, None, :]
        output += upper[:, :, x1] * x_weight[None, None, :]
        return output.astype(np.float32, copy=False)


def load_numpy_model(weights_path: str | Path = DEFAULT_NUMPY_WEIGHTS) -> NumpyPerceptionModel:
    path = Path(weights_path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    if not path.exists():
        raise ModuleNotFoundError(
            "PyTorch is unavailable and the NumPy perception weights are missing: "
            f"{path}"
        )
    with np.load(path, allow_pickle=False) as archive:
        parameters = {
            name.removeprefix("parameter__"): np.asarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name.startswith("parameter__")
        }
        metadata = {
            name.removeprefix("metadata__"): archive[name].item()
            for name in archive.files
            if name.startswith("metadata__")
        }
    return NumpyPerceptionModel(parameters, metadata)


def sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    output = np.empty_like(value, dtype=np.float32)
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def softmax(value: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=axis, keepdims=True)


__all__ = [
    "DEFAULT_NUMPY_WEIGHTS",
    "NumpyPerceptionModel",
    "load_numpy_model",
    "sigmoid",
    "softmax",
]
