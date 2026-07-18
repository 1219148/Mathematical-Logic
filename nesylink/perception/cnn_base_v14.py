from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH, TILE_SIZE
from nesylink.core.observation import TILE_BRIDGE, TILE_EXIT, TILE_MONSTER, TILE_PLAYER


NUM_TILE_CLASSES = 12
EXIT_TYPE_NAMES = ("none", "normal", "locked_key", "conditional")
EXIT_TYPE_TO_INDEX = {name: index for index, name in enumerate(EXIT_TYPE_NAMES)}
NUM_EXIT_TYPE_CLASSES = len(EXIT_TYPE_NAMES)
CHEST_STATE_NAMES = ("none", "closed", "opened")
CHEST_STATE_TO_INDEX = {name: index for index, name in enumerate(CHEST_STATE_NAMES)}
NUM_CHEST_STATE_CLASSES = len(CHEST_STATE_NAMES)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET = DEFAULT_DATA_DIR / "perception_dataset.npz"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "perception_model.pt"
BUILTIN_TASKS = tuple(f"mathematical_logic/task_{idx}" for idx in range(1, 6))
COLOR_IMAGE_VARIANTS = ("default", "grayscale", "dark", "bright", "high_contrast", "inverted")
REDRAW_IMAGE_VARIANTS = ("redraw_geometric", "redraw_symbols")
IMAGE_VARIANTS = COLOR_IMAGE_VARIANTS + REDRAW_IMAGE_VARIANTS
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
PLAYER_COORDINATE_LOSS_WEIGHT = 0.05
PLAYER_TILE_LOSS_WEIGHT = 0.15
MONSTER_COORDINATE_LOSS_WEIGHT = 0.08
MONSTER_TILE_LOSS_WEIGHT = 0.15
MONSTER_CENTER_WINDOW_RADIUS = 4
MONSTER_CENTER_POWER = 8
ENTITY_OFFSET_LOSS_WEIGHT = 0.20
ENTITY_OFFSET_STRIDE = 4
ENTITY_OFFSET_SCALE = 6.0
REFINE_HEATMAP_LOSS_WEIGHT = 0.35
REFINE_PLAYER_COORDINATE_LOSS_WEIGHT = 0.20
REFINE_PLAYER_TILE_LOSS_WEIGHT = 0.45
REFINE_MONSTER_COORDINATE_LOSS_WEIGHT = 0.20
REFINE_MONSTER_TILE_LOSS_WEIGHT = 0.35
PLAYER_COORDINATE_WINDOW_RADIUS = 4
PLAYER_COORDINATE_TEMPERATURE = 1.5
EXIT_OCCLUSION_CONFIDENCE = 0.70
EXIT_SIDE_PAIR_CONFIDENCE = 0.80
EXIT_BRIDGE_OVERLAP_CONFIDENCE = 0.60
MAX_TRAINING_VALIDATION_SAMPLES = 1024
REFINE_TRAIN_MODE = "auto"


@dataclass(frozen=True)
class DatasetConfig:
    output: Path = DEFAULT_DATASET
    samples: int = 2400
    builtin_ratio: float = 0.45
    exit_overlap_ratio: float = 0.15
    max_monsters: int = 6
    seed: int = 0


def _torch_modules() -> tuple[Any, Any, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset

    return torch, nn, Dataset


def _BaseTorchModule() -> type:
    try:
        return _torch_modules()[1].Module
    except ModuleNotFoundError:
        class _MissingTorchModule:
            pass

        return _MissingTorchModule


def _BaseTorchDataset() -> type:
    try:
        return _torch_modules()[2]
    except ModuleNotFoundError:
        class _MissingTorchDataset:
            pass

        return _MissingTorchDataset


class TinyPerceptionCNN(_BaseTorchModule()):
    """小型感知网络。

    tile_logits 还原 8x10 语义图；heatmap_logits 预测玩家/怪物中心点。
    训练时使用 info/structured obs 自动标注，推理时只输入 raw pixels。
    """

    def __init__(
        self,
        num_classes: int = NUM_TILE_CLASSES,
        num_exit_type_classes: int = NUM_EXIT_TYPE_CLASSES,
        num_chest_state_classes: int = NUM_CHEST_STATE_CLASSES,
    ) -> None:
        torch, nn, _ = _torch_modules()
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 80
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 40
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )
        self.tile_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((GRID_HEIGHT, GRID_WIDTH)),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, num_classes, kernel_size=1),
        )
        self.exit_type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((GRID_HEIGHT, GRID_WIDTH)),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_exit_type_classes, kernel_size=1),
        )
        self.exit_type_head_available = True
        self.chest_state_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((GRID_HEIGHT, GRID_WIDTH)),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_chest_state_classes, kernel_size=1),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
        )
        self.player_refine_head = nn.Sequential(
            nn.Conv2d(98, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.monster_refine_head = nn.Sequential(
            nn.Conv2d(98, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        for refine_head in (self.player_refine_head, self.monster_refine_head):
            nn.init.zeros_(refine_head[-1].weight)
            nn.init.zeros_(refine_head[-1].bias)
        self._interpolate = torch.nn.functional.interpolate
        self._concat = torch.cat
        self.has_trained_offset_head = True
        self.has_trained_refine_head = True

    def forward(self, images: Any) -> dict[str, Any]:
        high_resolution_features = self.encoder[:6](images)
        mid_resolution_features = self.encoder[6:10](high_resolution_features)
        features = self.encoder[10:](mid_resolution_features)
        tile_logits = self.tile_head(features)
        exit_type_logits = self.exit_type_head(features)
        chest_state_logits = self.chest_state_head(features)
        heatmap_logits = self.heatmap_head(features)
        heatmap_logits = self._interpolate(
            heatmap_logits,
            size=(MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        if getattr(self, "has_trained_refine_head", True):
            mid_resolution_features = self._interpolate(
                mid_resolution_features,
                size=(MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH),
                mode="bilinear",
                align_corners=False,
            )
            refine_features = self._concat(
                (
                    high_resolution_features,
                    mid_resolution_features,
                    heatmap_logits,
                ),
                dim=1,
            )
            refinement = self._concat(
                (
                    self.player_refine_head(refine_features),
                    self.monster_refine_head(refine_features),
                ),
                dim=1,
            )
            heatmap_logits = heatmap_logits + refinement
        offset_logits = self.offset_head(features)
        return {
            "tile_logits": tile_logits,
            "exit_type_logits": exit_type_logits,
            "chest_state_logits": chest_state_logits,
            "heatmap_logits": heatmap_logits,
            "offset_logits": offset_logits,
        }


class PerceptionDataset(_BaseTorchDataset()):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        indices: np.ndarray,
        augment: bool = False,
        variants: Sequence[str] = ("default",),
        randomize_variant: bool = False,
    ) -> None:
        torch, _, Dataset = _torch_modules()
        super().__init__()
        del Dataset
        data = np.load(dataset_path)
        self.images = data["images"][indices]
        self.redraw_images = {
            variant: data[f"images_{variant}"][indices]
            for variant in REDRAW_IMAGE_VARIANTS
            if f"images_{variant}" in data.files
        }
        self.grids = data["grids"][indices]
        if "exit_type_grids" in data.files:
            self.exit_type_grids = data["exit_type_grids"][indices]
        else:
            self.exit_type_grids = _fallback_exit_type_grids(self.grids)
        if "chest_state_grids" in data.files:
            self.chest_state_grids = data["chest_state_grids"][indices]
        else:
            self.chest_state_grids = _fallback_chest_state_grids(self.grids)
        self.player_centers = data["player_centers"][indices]
        self.monster_centers = data["monster_centers"][indices]
        self.monster_masks = data["monster_masks"][indices]
        self.augment = bool(augment)
        self.variants = tuple(variants)
        self.randomize_variant = bool(randomize_variant)
        if not self.variants:
            raise ValueError("variants must contain at least one image variant")
        unknown_variants = sorted(set(self.variants) - set(IMAGE_VARIANTS))
        if unknown_variants:
            raise ValueError(f"unsupported image variants: {unknown_variants}")
        missing_redraw = sorted(
            (set(self.variants) & set(REDRAW_IMAGE_VARIANTS)) - set(self.redraw_images)
        )
        if missing_redraw:
            raise ValueError(
                "dataset is missing redraw images for variants: " + ", ".join(missing_redraw)
            )
        self.torch = torch

    def __len__(self) -> int:
        multiplier = 1 if self.randomize_variant else len(self.variants)
        return int(self.images.shape[0]) * multiplier

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.randomize_variant:
            sample_index = index
            variant = random.choice(self.variants)
        else:
            sample_index = index // len(self.variants)
            variant = self.variants[index % len(self.variants)]
        if variant in self.redraw_images:
            image = self.redraw_images[variant][sample_index].astype(np.float32) / 255.0
        else:
            image = apply_image_variant(self.images[sample_index], variant)
        if self.augment:
            image = _augment_image(image)
        image_tensor = self.torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        grid_tensor = self.torch.from_numpy(self.grids[sample_index].astype(np.int64))
        exit_type_tensor = self.torch.from_numpy(self.exit_type_grids[sample_index].astype(np.int64))
        chest_state_tensor = self.torch.from_numpy(self.chest_state_grids[sample_index].astype(np.int64))
        heatmap = _make_heatmaps(
            self.player_centers[sample_index],
            self.monster_centers[sample_index],
            self.monster_masks[sample_index],
        )
        return {
            "image": image_tensor,
            "grid": grid_tensor,
            "exit_type_grid": exit_type_tensor,
            "chest_state_grid": chest_state_tensor,
            "heatmap": self.torch.from_numpy(heatmap).float(),
            "player_center": self.torch.from_numpy(
                self.player_centers[sample_index].astype(np.float32)
            ),
            "monster_centers": self.torch.from_numpy(
                self.monster_centers[sample_index].astype(np.float32)
            ),
            "monster_masks": self.torch.from_numpy(
                self.monster_masks[sample_index].astype(np.bool_)
            ),
        }


def collect_dataset(config: DatasetConfig) -> Path:
    """采集 perception 数据。

    数据来源包含公开任务 rollout 和随机单房间地图。标签来自 structured obs，
    只用于训练阶段；最终 PerceptionEngine.extract 不会读取 info。
    """
    if config.samples < 1:
        raise ValueError("samples must be >= 1")
    if config.builtin_ratio < 0 or config.exit_overlap_ratio < 0:
        raise ValueError("dataset ratios must be non-negative")
    if config.builtin_ratio + config.exit_overlap_ratio > 1:
        raise ValueError("builtin_ratio + exit_overlap_ratio must be <= 1")

    rng = random.Random(config.seed)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    images: list[np.ndarray] = []
    redraw_images: dict[str, list[np.ndarray]] = {variant: [] for variant in REDRAW_IMAGE_VARIANTS}
    grids: list[np.ndarray] = []
    exit_type_grids: list[np.ndarray] = []
    chest_state_grids: list[np.ndarray] = []
    player_centers: list[np.ndarray] = []
    monster_centers: list[np.ndarray] = []
    monster_masks: list[np.ndarray] = []

    builtin_samples = int(config.samples * config.builtin_ratio)
    exit_overlap_samples = int(config.samples * config.exit_overlap_ratio)
    random_samples = max(0, config.samples - builtin_samples - exit_overlap_samples)
    _collect_builtin_samples(
        target_count=builtin_samples,
        rng=rng,
        images=images,
        redraw_images=redraw_images,
        grids=grids,
        exit_type_grids=exit_type_grids,
        chest_state_grids=chest_state_grids,
        player_centers=player_centers,
        monster_centers=monster_centers,
        monster_masks=monster_masks,
        max_monsters=config.max_monsters,
    )
    _collect_exit_overlap_samples(
        target_count=exit_overlap_samples,
        rng=rng,
        images=images,
        redraw_images=redraw_images,
        grids=grids,
        exit_type_grids=exit_type_grids,
        chest_state_grids=chest_state_grids,
        player_centers=player_centers,
        monster_centers=monster_centers,
        monster_masks=monster_masks,
        max_monsters=config.max_monsters,
    )
    _collect_random_room_samples(
        target_count=random_samples,
        rng=rng,
        images=images,
        redraw_images=redraw_images,
        grids=grids,
        exit_type_grids=exit_type_grids,
        chest_state_grids=chest_state_grids,
        player_centers=player_centers,
        monster_centers=monster_centers,
        monster_masks=monster_masks,
        max_monsters=config.max_monsters,
        data_dir=output.parent,
    )

    np.savez_compressed(
        output,
        images=np.stack(images).astype(np.uint8),
        **{
            f"images_{variant}": np.stack(variant_images).astype(np.uint8)
            for variant, variant_images in redraw_images.items()
        },
        grids=np.stack(grids).astype(np.uint8),
        exit_type_grids=np.stack(exit_type_grids).astype(np.uint8),
        chest_state_grids=np.stack(chest_state_grids).astype(np.uint8),
        player_centers=np.stack(player_centers).astype(np.float32),
        monster_centers=np.stack(monster_centers).astype(np.float32),
        monster_masks=np.stack(monster_masks).astype(np.bool_),
    )
    return output


def train_model(
    dataset_path: str | Path = DEFAULT_DATASET,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    *,
    epochs: int = 14,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    chest_head_only: bool = False,
) -> dict[str, float]:
    torch, nn, _ = _torch_modules()
    DataLoader = torch.utils.data.DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    resolved_device = _resolve_device(torch, device)
    dataset_path = Path(dataset_path)
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path)
    sample_count = int(data["images"].shape[0])
    rng = np.random.default_rng(seed)
    train_indices, val_indices = _grouped_train_val_split(sample_count, rng)
    if len(val_indices) > MAX_TRAINING_VALIDATION_SAMPLES:
        val_indices = val_indices[:MAX_TRAINING_VALIDATION_SAMPLES]

    train_ds = PerceptionDataset(
        dataset_path,
        indices=train_indices,
        augment=True,
        variants=IMAGE_VARIANTS,
        randomize_variant=True,
    )
    val_ds = PerceptionDataset(dataset_path, indices=val_indices, augment=False, variants=IMAGE_VARIANTS)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyPerceptionCNN().to(resolved_device)
    warm_started = _warm_start_model(model, weights_path, torch, resolved_device)
    if warm_started:
        model.has_trained_offset_head = bool(
            getattr(model, "offset_head_warm_started", False)
        )
        model.has_trained_refine_head = bool(
            getattr(model, "refine_head_warm_started", False)
        )
    if REFINE_TRAIN_MODE not in {"auto", "both", "player", "monster"}:
        raise ValueError(f"unsupported REFINE_TRAIN_MODE={REFINE_TRAIN_MODE!r}")
    refine_target = "both" if REFINE_TRAIN_MODE == "auto" else REFINE_TRAIN_MODE
    refine_head_only = (
        not chest_head_only
        and warm_started
        and (
            REFINE_TRAIN_MODE != "auto"
            or not getattr(model, "refine_head_warm_started", False)
        )
    )
    offset_head_only = (
        not chest_head_only
        and not refine_head_only
        and warm_started
        and not getattr(model, "offset_head_warm_started", False)
    )
    if chest_head_only:
        if not warm_started:
            raise ValueError("--chest-head-only requires an existing compatible checkpoint")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.chest_state_head.parameters():
            parameter.requires_grad_(True)
        trainable_parameters = model.chest_state_head.parameters()
    elif refine_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        refine_parameters = []
        if refine_target in {"both", "player"}:
            refine_parameters.extend(model.player_refine_head.parameters())
        if refine_target in {"both", "monster"}:
            refine_parameters.extend(model.monster_refine_head.parameters())
        for parameter in refine_parameters:
            parameter.requires_grad_(True)
        model.has_trained_refine_head = True
        trainable_parameters = refine_parameters
        print(
            f"training_refine_head_only=true target={refine_target}",
            flush=True,
        )
    elif offset_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.offset_head.parameters():
            parameter.requires_grad_(True)
        # Validation and checkpoint metadata must evaluate the freshly trained
        # continuous-coordinate head during this warm-start phase.  Leaving the
        # flag false silently measured the old heatmap-only centres and could
        # never select the best offset checkpoint.
        model.has_trained_offset_head = True
        trainable_parameters = model.offset_head.parameters()
        print("training_offset_head_only=true", flush=True)
    else:
        localization_parameters = (
            list(model.heatmap_head.parameters())
            + list(model.offset_head.parameters())
            + list(model.player_refine_head.parameters())
            + list(model.monster_refine_head.parameters())
        )
        localization_ids = {id(parameter) for parameter in localization_parameters}
        shared_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in localization_ids
        ]
        trainable_parameters = [
            {"params": shared_parameters, "lr": lr * 0.25},
            {"params": localization_parameters, "lr": lr},
        ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=1e-4)
    if chest_head_only:
        train_loader = _cache_chest_feature_loader(
            model,
            train_loader,
            device=resolved_device,
            torch=torch,
            batch_size=max(batch_size, 256),
            shuffle=True,
        )
        val_loader = _cache_chest_feature_loader(
            model,
            val_loader,
            device=resolved_device,
            torch=torch,
            batch_size=max(batch_size, 256),
            shuffle=False,
        )
    tile_weights = _tile_class_weights(data["grids"]).to(resolved_device)
    exit_type_labels = (
        data["exit_type_grids"]
        if "exit_type_grids" in data.files
        else _fallback_exit_type_grids(data["grids"])
    )
    exit_type_weights = _exit_type_class_weights(exit_type_labels).to(resolved_device)
    chest_state_labels = (
        data["chest_state_grids"]
        if "chest_state_grids" in data.files
        else _fallback_chest_state_grids(data["grids"])
    )
    chest_state_weights = _chest_state_class_weights(chest_state_labels).to(resolved_device)
    tile_loss_fn = nn.CrossEntropyLoss(weight=tile_weights)
    exit_type_loss_fn = nn.CrossEntropyLoss(weight=exit_type_weights)
    chest_state_loss_fn = nn.CrossEntropyLoss(weight=chest_state_weights)
    heatmap_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(12.0, device=resolved_device))

    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    patience = 6
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        if chest_head_only:
            model.eval()
            model.chest_state_head.train()
        elif refine_head_only:
            model.eval()
            if refine_target in {"both", "player"}:
                model.player_refine_head.train()
            if refine_target in {"both", "monster"}:
                model.monster_refine_head.train()
        elif offset_head_only:
            model.eval()
            model.offset_head.train()
        else:
            model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            if chest_head_only:
                features, chest_state_grids = batch
                features = features.to(resolved_device)
                chest_state_grids = chest_state_grids.to(resolved_device)
                chest_state_logits = model.chest_state_head(features)
                loss = chest_state_loss_fn(chest_state_logits, chest_state_grids)
            else:
                images = batch["image"].to(resolved_device)
                chest_state_grids = batch["chest_state_grid"].to(resolved_device)
                grids = batch["grid"].to(resolved_device)
                exit_type_grids = batch["exit_type_grid"].to(resolved_device)
                heatmaps = batch["heatmap"].to(resolved_device)
                player_centers = batch["player_center"].to(resolved_device)
                monster_centers = batch["monster_centers"].to(resolved_device)
                monster_masks = batch["monster_masks"].to(resolved_device)
                output = model(images)
                tile_loss = tile_loss_fn(output["tile_logits"], grids)
                exit_type_loss = exit_type_loss_fn(output["exit_type_logits"], exit_type_grids)
                chest_state_loss = chest_state_loss_fn(output["chest_state_logits"], chest_state_grids)
                heatmap_loss = heatmap_loss_fn(output["heatmap_logits"], heatmaps)
                player_heatmap_loss = heatmap_loss_fn(
                    output["heatmap_logits"][:, 0:1], heatmaps[:, 0:1]
                )
                monster_heatmap_loss = heatmap_loss_fn(
                    output["heatmap_logits"][:, 1:2], heatmaps[:, 1:2]
                )
                player_coordinate_loss, _ = _player_coordinate_loss(
                    output["heatmap_logits"],
                    player_centers,
                    torch=torch,
                )
                player_tile_loss = _player_tile_classification_loss(
                    output["heatmap_logits"],
                    player_centers,
                    torch=torch,
                )
                monster_coordinate_loss = _monster_coordinate_loss(
                    output["heatmap_logits"],
                    monster_centers,
                    monster_masks,
                    torch=torch,
                )
                monster_tile_loss = _monster_tile_classification_loss(
                    output["heatmap_logits"],
                    monster_centers,
                    monster_masks,
                    torch=torch,
                )
                entity_offset_loss = _entity_offset_loss(
                    output["offset_logits"],
                    player_centers,
                    monster_centers,
                    monster_masks,
                    torch=torch,
                )
                loss = (
                    tile_loss
                    + 0.30 * exit_type_loss
                    + 0.55 * chest_state_loss
                    + 0.35 * heatmap_loss
                    + PLAYER_COORDINATE_LOSS_WEIGHT * player_coordinate_loss
                    + PLAYER_TILE_LOSS_WEIGHT * player_tile_loss
                    + MONSTER_COORDINATE_LOSS_WEIGHT * monster_coordinate_loss
                    + MONSTER_TILE_LOSS_WEIGHT * monster_tile_loss
                    + ENTITY_OFFSET_LOSS_WEIGHT * entity_offset_loss
                )
                if offset_head_only:
                    loss = entity_offset_loss
                elif refine_head_only:
                    player_refine_loss = (
                        REFINE_HEATMAP_LOSS_WEIGHT * player_heatmap_loss
                        + REFINE_PLAYER_COORDINATE_LOSS_WEIGHT * player_coordinate_loss
                        + REFINE_PLAYER_TILE_LOSS_WEIGHT * player_tile_loss
                    )
                    monster_refine_loss = (
                        REFINE_HEATMAP_LOSS_WEIGHT * monster_heatmap_loss
                        + REFINE_MONSTER_COORDINATE_LOSS_WEIGHT * monster_coordinate_loss
                        + REFINE_MONSTER_TILE_LOSS_WEIGHT * monster_tile_loss
                    )
                    if refine_target == "player":
                        loss = player_refine_loss
                    elif refine_target == "monster":
                        loss = monster_refine_loss
                    else:
                        loss = player_refine_loss + monster_refine_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1

        metrics = (
            evaluate_chest_head(model, val_loader, device=resolved_device)
            if chest_head_only
            else evaluate_model(model, val_loader, device=resolved_device)
        )
        metrics["train_loss"] = train_loss / max(1, train_batches)
        if chest_head_only:
            print(
                f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} chest_acc={metrics['chest_state_acc']:.4f} "
                f"chest_closed_recall={metrics['chest_closed_recall']:.4f} "
                f"chest_open_recall={metrics['chest_open_recall']:.4f} "
                f"chest_fpr={metrics['chest_false_positive_rate']:.6f}",
                flush=True,
            )
        else:
            print(
                f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} tile_acc={metrics['tile_acc']:.4f} "
                f"exit_type_acc={metrics['exit_type_acc']:.4f} "
                f"chest_closed_recall={metrics['chest_closed_recall']:.4f} "
                f"chest_open_recall={metrics['chest_open_recall']:.4f} "
                f"player_err={metrics['player_center_error_px']:.2f}px "
                f"monster_err={metrics['monster_center_error_px']:.2f}px "
                f"monster_recall={metrics['monster_tile_recall']:.4f}",
                flush=True,
            )
        selection_score = metrics["val_loss"]
        if refine_head_only:
            player_selection_score = metrics["player_center_error_px"] + 8.0 * (
                1.0 - metrics["player_tile_acc"]
            )
            monster_selection_score = metrics["monster_center_error_px"] + 8.0 * (
                1.0 - metrics["monster_tile_recall"]
            )
            if refine_target == "player":
                selection_score = player_selection_score
            elif refine_target == "monster":
                selection_score = monster_selection_score
            else:
                selection_score = player_selection_score + monster_selection_score
        elif offset_head_only:
            selection_score = (
                metrics["player_center_error_px"]
                + metrics["monster_center_error_px"]
                + 5.0 * (1.0 - metrics["monster_tile_recall"])
            )
        metrics["selection_score"] = float(selection_score)
        if selection_score < best_val:
            best_val = selection_score
            best_metrics = dict(metrics)
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_tile_classes": NUM_TILE_CLASSES,
                    "num_exit_type_classes": NUM_EXIT_TYPE_CLASSES,
                    "num_chest_state_classes": NUM_CHEST_STATE_CLASSES,
                    "exit_type_names": EXIT_TYPE_NAMES,
                    "chest_state_names": CHEST_STATE_NAMES,
                    "metrics": best_metrics,
                    "dataset": str(dataset_path),
                    "has_trained_offset_head": bool(
                        getattr(model, "has_trained_offset_head", False)
                    ),
                    "has_trained_refine_head": bool(
                        getattr(model, "has_trained_refine_head", False)
                    ),
                },
                weights_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience and (chest_head_only or metrics["tile_acc"] > 0.97):
                break
    return best_metrics


def _grouped_train_val_split(
    sample_count: int,
    rng: np.random.Generator,
    *,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep neighboring rollout frames together to avoid validation leakage."""

    blocks = [
        np.arange(start, min(start + block_size, sample_count), dtype=np.int64)
        for start in range(0, sample_count, block_size)
    ]
    if len(blocks) == 1:
        indices = blocks[0].copy()
        rng.shuffle(indices)
        split = max(1, int(len(indices) * 0.85))
        return indices[:split], indices[split:] if split < len(indices) else indices[:1]
    rng.shuffle(blocks)
    train_block_count = min(len(blocks) - 1, max(1, int(len(blocks) * 0.85)))
    train_indices = np.concatenate(blocks[:train_block_count])
    val_indices = np.concatenate(blocks[train_block_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _player_coordinate_loss(
    heatmap_logits: Any,
    player_centers: Any,
    *,
    torch: Any,
    radius: int = PLAYER_COORDINATE_WINDOW_RADIUS,
) -> tuple[Any, Any]:
    """Supervise the sub-pixel center around the detached predicted peak."""
    player_logits = heatmap_logits[:, 0]
    height, width = player_logits.shape[-2:]
    ys = torch.arange(height, device=player_logits.device, dtype=player_logits.dtype)
    xs = torch.arange(width, device=player_logits.device, dtype=player_logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    peak_indices = player_logits.detach().flatten(1).argmax(dim=1)
    peak_x = (peak_indices % width)[:, None, None]
    peak_y = (peak_indices // width)[:, None, None]
    local_mask = (torch.abs(grid_x - peak_x) <= radius) & (
        torch.abs(grid_y - peak_y) <= radius
    )
    probabilities = torch.sigmoid(player_logits)
    weights = torch.relu(probabilities - 0.05).pow(MONSTER_CENTER_POWER)
    weights = weights * local_mask.to(weights.dtype)
    weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    predicted_centers = torch.stack(
        (
            (weights * grid_x).sum(dim=(1, 2)),
            (weights * grid_y).sum(dim=(1, 2)),
        ),
        dim=1,
    )
    loss = torch.nn.functional.smooth_l1_loss(
        predicted_centers,
        player_centers,
        beta=0.5,
    )
    return loss, predicted_centers


def _player_tile_classification_loss(
    heatmap_logits: Any,
    player_centers: Any,
    *,
    torch: Any,
) -> Any:
    """Make the localization heatmap select the correct tile at boundaries."""

    player_logits = heatmap_logits[:, 0]
    batch_size = player_logits.shape[0]
    tile_scores = player_logits.reshape(
        batch_size,
        GRID_HEIGHT,
        TILE_SIZE,
        GRID_WIDTH,
        TILE_SIZE,
    ).permute(0, 1, 3, 2, 4)
    tile_scores = torch.logsumexp(tile_scores.flatten(3), dim=3).flatten(1)
    target_x = torch.floor(player_centers[:, 0] / TILE_SIZE).long()
    target_y = torch.floor(player_centers[:, 1] / TILE_SIZE).long()
    target_x = target_x.clamp(0, GRID_WIDTH - 1)
    target_y = target_y.clamp(0, GRID_HEIGHT - 1)
    target_tiles = target_y * GRID_WIDTH + target_x
    return torch.nn.functional.cross_entropy(tile_scores, target_tiles)


def _monster_coordinate_loss(
    heatmap_logits: Any,
    monster_centers: Any,
    monster_masks: Any,
    *,
    torch: Any,
    radius: int = PLAYER_COORDINATE_WINDOW_RADIUS,
) -> Any:
    """Supervise every visible monster center, including moving sub-pixel poses."""

    valid = monster_masks.bool()
    if not bool(valid.any()):
        return heatmap_logits[:, 1].sum() * 0.0
    batch_indices = torch.arange(
        heatmap_logits.shape[0], device=heatmap_logits.device
    )[:, None].expand_as(valid)[valid]
    targets = monster_centers[valid]
    logits = heatmap_logits[batch_indices, 1]
    height, width = logits.shape[-2:]
    offsets = torch.arange(-radius, radius + 1, device=logits.device)
    center_x = torch.round(targets[:, 0]).long()
    center_y = torch.round(targets[:, 1]).long()
    xs = (center_x[:, None] + offsets[None, :]).clamp(0, width - 1)
    ys = (center_y[:, None] + offsets[None, :]).clamp(0, height - 1)
    target_indices = torch.arange(logits.shape[0], device=logits.device)
    local_logits = logits[
        target_indices[:, None, None],
        ys[:, :, None],
        xs[:, None, :],
    ]
    probabilities = torch.softmax(
        local_logits.flatten(1) / PLAYER_COORDINATE_TEMPERATURE,
        dim=1,
    ).view_as(local_logits)
    predicted = torch.stack(
        (
            (probabilities * xs[:, None, :].to(logits.dtype)).sum(dim=(1, 2)),
            (probabilities * ys[:, :, None].to(logits.dtype)).sum(dim=(1, 2)),
        ),
        dim=1,
    )
    return torch.nn.functional.smooth_l1_loss(predicted, targets, beta=0.5)


def _monster_tile_classification_loss(
    heatmap_logits: Any,
    monster_centers: Any,
    monster_masks: Any,
    *,
    torch: Any,
) -> Any:
    """Use a task-independent multi-label objective for monster tile stability."""

    logits = heatmap_logits[:, 1]
    batch_size = logits.shape[0]
    tile_scores = logits.reshape(
        batch_size,
        GRID_HEIGHT,
        TILE_SIZE,
        GRID_WIDTH,
        TILE_SIZE,
    ).permute(0, 1, 3, 2, 4)
    tile_scores = torch.logsumexp(tile_scores.flatten(3), dim=3) - math.log(
        TILE_SIZE * TILE_SIZE
    )
    targets = torch.zeros_like(tile_scores)
    valid = monster_masks.bool()
    if bool(valid.any()):
        target_x = torch.floor(monster_centers[..., 0] / TILE_SIZE).long()
        target_y = torch.floor(monster_centers[..., 1] / TILE_SIZE).long()
        target_x = target_x.clamp(0, GRID_WIDTH - 1)
        target_y = target_y.clamp(0, GRID_HEIGHT - 1)
        batch_indices = torch.arange(batch_size, device=logits.device)[:, None]
        targets[batch_indices.expand_as(valid)[valid], target_y[valid], target_x[valid]] = 1.0
    return torch.nn.functional.binary_cross_entropy_with_logits(
        tile_scores,
        targets,
        pos_weight=torch.tensor(12.0, device=logits.device, dtype=logits.dtype),
    )


def _entity_offset_loss(
    offset_logits: Any,
    player_centers: Any,
    monster_centers: Any,
    monster_masks: Any,
    *,
    torch: Any,
) -> Any:
    """Regress a continuous center from any neighboring low-resolution cell."""

    player_masks = torch.ones(
        player_centers.shape[0],
        1,
        dtype=torch.bool,
        device=player_centers.device,
    )
    player_loss = _offset_loss_for_entities(
        offset_logits,
        player_centers[:, None, :],
        player_masks,
        channel_offset=0,
        torch=torch,
    )
    monster_loss = _offset_loss_for_entities(
        offset_logits,
        monster_centers,
        monster_masks.bool(),
        channel_offset=2,
        torch=torch,
    )
    return player_loss + monster_loss


def _offset_loss_for_entities(
    offset_logits: Any,
    centers: Any,
    masks: Any,
    *,
    channel_offset: int,
    torch: Any,
) -> Any:
    valid = masks.bool()
    if not bool(valid.any()):
        return offset_logits.sum() * 0.0
    batch_indices = torch.arange(
        offset_logits.shape[0], device=offset_logits.device
    )[:, None].expand_as(valid)[valid]
    targets = centers[valid]
    center_x = torch.floor(targets[:, 0] / ENTITY_OFFSET_STRIDE).long()
    center_y = torch.floor(targets[:, 1] / ENTITY_OFFSET_STRIDE).long()
    neighbor_x, neighbor_y = torch.meshgrid(
        torch.arange(-1, 2, device=offset_logits.device),
        torch.arange(-1, 2, device=offset_logits.device),
        indexing="xy",
    )
    cells_x = center_x[:, None] + neighbor_x.flatten()[None, :]
    cells_y = center_y[:, None] + neighbor_y.flatten()[None, :]
    in_bounds = (
        (cells_x >= 0)
        & (cells_x < offset_logits.shape[-1])
        & (cells_y >= 0)
        & (cells_y < offset_logits.shape[-2])
    )
    entity_indices, neighbor_indices = torch.where(in_bounds)
    selected_batch = batch_indices[entity_indices]
    selected_x = cells_x[entity_indices, neighbor_indices]
    selected_y = cells_y[entity_indices, neighbor_indices]
    predicted = torch.tanh(
        offset_logits[
            selected_batch,
            channel_offset : channel_offset + 2,
            selected_y,
            selected_x,
        ]
    ) * ENTITY_OFFSET_SCALE
    bases = torch.stack(
        (
            selected_x.to(targets.dtype) * ENTITY_OFFSET_STRIDE
            + ENTITY_OFFSET_STRIDE * 0.5,
            selected_y.to(targets.dtype) * ENTITY_OFFSET_STRIDE
            + ENTITY_OFFSET_STRIDE * 0.5,
        ),
        dim=1,
    )
    target_offsets = targets[entity_indices] - bases
    return torch.nn.functional.smooth_l1_loss(
        predicted,
        target_offsets,
        beta=0.25,
    )


def evaluate_model(model: Any, loader: Any, *, device: Any) -> dict[str, float]:
    torch, nn, _ = _torch_modules()
    tile_loss_fn = nn.CrossEntropyLoss()
    exit_type_loss_fn = nn.CrossEntropyLoss()
    chest_state_loss_fn = nn.CrossEntropyLoss()
    heatmap_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(12.0, device=device))
    model.eval()
    total_loss = 0.0
    total_tiles = 0
    correct_tiles = 0
    exit_type_total = 0
    correct_exit_types = 0
    chest_state_total = 0
    correct_chest_states = 0
    chest_closed_hits = 0
    chest_closed_total = 0
    chest_open_hits = 0
    chest_open_total = 0
    chest_false_positives = 0
    chest_none_total = 0
    player_errors: list[float] = []
    player_tile_hits = 0
    player_tile_total = 0
    monster_errors: list[float] = []
    monster_hits = 0
    monster_total = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            grids = batch["grid"].to(device)
            exit_type_grids = batch["exit_type_grid"].to(device)
            chest_state_grids = batch["chest_state_grid"].to(device)
            heatmaps = batch["heatmap"].to(device)
            player_centers = batch["player_center"].to(device)
            monster_centers = batch["monster_centers"].to(device)
            monster_masks = batch["monster_masks"].to(device)
            output = model(images)
            player_coordinate_loss, _ = _player_coordinate_loss(
                output["heatmap_logits"],
                player_centers,
                torch=torch,
            )
            player_tile_loss = _player_tile_classification_loss(
                output["heatmap_logits"],
                player_centers,
                torch=torch,
            )
            monster_coordinate_loss = _monster_coordinate_loss(
                output["heatmap_logits"],
                monster_centers,
                monster_masks,
                torch=torch,
            )
            monster_tile_loss = _monster_tile_classification_loss(
                output["heatmap_logits"],
                monster_centers,
                monster_masks,
                torch=torch,
            )
            entity_offset_loss = _entity_offset_loss(
                output["offset_logits"],
                player_centers,
                monster_centers,
                monster_masks,
                torch=torch,
            )
            loss = (
                tile_loss_fn(output["tile_logits"], grids)
                + 0.30 * exit_type_loss_fn(output["exit_type_logits"], exit_type_grids)
                + 0.55 * chest_state_loss_fn(output["chest_state_logits"], chest_state_grids)
                + 0.35
                * heatmap_loss_fn(
                    output["heatmap_logits"],
                    heatmaps,
                )
                + PLAYER_COORDINATE_LOSS_WEIGHT * player_coordinate_loss
                + PLAYER_TILE_LOSS_WEIGHT * player_tile_loss
                + MONSTER_COORDINATE_LOSS_WEIGHT * monster_coordinate_loss
                + MONSTER_TILE_LOSS_WEIGHT * monster_tile_loss
                + ENTITY_OFFSET_LOSS_WEIGHT * entity_offset_loss
            )
            total_loss += float(loss.cpu())
            batches += 1

            pred_grid = output["tile_logits"].argmax(dim=1)
            correct_tiles += int((pred_grid == grids).sum().cpu())
            total_tiles += int(grids.numel())
            pred_exit_type_grid = output["exit_type_logits"].argmax(dim=1)
            exit_mask = exit_type_grids != EXIT_TYPE_TO_INDEX["none"]
            correct_exit_types += int((pred_exit_type_grid[exit_mask] == exit_type_grids[exit_mask]).sum().cpu())
            exit_type_total += int(exit_mask.sum().cpu())

            pred_chest_state_grid = output["chest_state_logits"].argmax(dim=1)
            correct_chest_states += int((pred_chest_state_grid == chest_state_grids).sum().cpu())
            chest_state_total += int(chest_state_grids.numel())
            closed_mask = chest_state_grids == CHEST_STATE_TO_INDEX["closed"]
            open_mask = chest_state_grids == CHEST_STATE_TO_INDEX["opened"]
            none_mask = chest_state_grids == CHEST_STATE_TO_INDEX["none"]
            chest_closed_hits += int((pred_chest_state_grid[closed_mask] == CHEST_STATE_TO_INDEX["closed"]).sum().cpu())
            chest_closed_total += int(closed_mask.sum().cpu())
            chest_open_hits += int((pred_chest_state_grid[open_mask] == CHEST_STATE_TO_INDEX["opened"]).sum().cpu())
            chest_open_total += int(open_mask.sum().cpu())
            chest_false_positives += int((pred_chest_state_grid[none_mask] != CHEST_STATE_TO_INDEX["none"]).sum().cpu())
            chest_none_total += int(none_mask.sum().cpu())

            probs = torch.sigmoid(output["heatmap_logits"]).cpu().numpy()
            offset_maps = (
                torch.tanh(output["offset_logits"]).cpu().numpy()
                * ENTITY_OFFSET_SCALE
                if getattr(model, "has_trained_offset_head", False)
                else None
            )
            true_player_centers = player_centers.cpu().numpy()
            true_monster_centers = monster_centers.cpu().numpy()
            true_monster_masks = monster_masks.cpu().numpy()
            for sample_idx in range(probs.shape[0]):
                pred_player_y, pred_player_x = np.unravel_index(
                    int(np.argmax(probs[sample_idx, 0])),
                    probs[sample_idx, 0].shape,
                )
                pred_player_x, pred_player_y = _refine_peak_center(
                    probs[sample_idx, 0],
                    int(pred_player_x),
                    int(pred_player_y),
                )
                if offset_maps is not None:
                    pred_player_x, pred_player_y = _center_from_offset_prediction(
                        (pred_player_x, pred_player_y),
                        offset_maps[sample_idx],
                        channel_offset=0,
                    )
                true_player_x, true_player_y = true_player_centers[sample_idx]
                player_errors.append(
                    math.hypot(pred_player_x - true_player_x, pred_player_y - true_player_y)
                )
                player_tile_hits += int(
                    int(pred_player_x // TILE_SIZE) == int(true_player_x // TILE_SIZE)
                    and int(pred_player_y // TILE_SIZE) == int(true_player_y // TILE_SIZE)
                )
                player_tile_total += 1

                pred_monster_tiles = _tiles_from_heatmap(
                    probs[sample_idx, 1],
                    threshold=0.35,
                    max_peaks=8,
                    offset_map=(
                        offset_maps[sample_idx]
                        if offset_maps is not None
                        else None
                    ),
                )
                pred_monster_centers = _centers_from_heatmap(
                    probs[sample_idx, 1],
                    threshold=0.35,
                    max_peaks=8,
                    offset_map=(
                        offset_maps[sample_idx]
                        if offset_maps is not None
                        else None
                    ),
                )
                monster_errors.extend(
                    _greedy_center_errors(
                        pred_monster_centers,
                        true_monster_centers[sample_idx][
                            true_monster_masks[sample_idx]
                        ],
                    )
                )
                true_monster_tiles = set(zip(*np.where(grids[sample_idx].cpu().numpy() == TILE_MONSTER)[::-1]))
                monster_total += len(true_monster_tiles)
                monster_hits += len(set(pred_monster_tiles) & true_monster_tiles)

    return {
        "val_loss": total_loss / max(1, batches),
        "tile_acc": correct_tiles / max(1, total_tiles),
        "exit_type_acc": correct_exit_types / max(1, exit_type_total),
        "chest_state_acc": correct_chest_states / max(1, chest_state_total),
        "chest_closed_recall": chest_closed_hits / max(1, chest_closed_total),
        "chest_open_recall": chest_open_hits / max(1, chest_open_total),
        "chest_false_positive_rate": chest_false_positives / max(1, chest_none_total),
        "player_center_error_px": float(np.mean(player_errors)) if player_errors else 0.0,
        "player_tile_acc": player_tile_hits / max(1, player_tile_total),
        "monster_center_error_px": (
            float(np.mean(monster_errors)) if monster_errors else 0.0
        ),
        "monster_tile_recall": monster_hits / max(1, monster_total),
    }


def _cache_chest_feature_loader(
    model: Any,
    loader: Any,
    *,
    device: Any,
    torch: Any,
    batch_size: int,
    shuffle: bool,
) -> Any:
    """Run the frozen encoder once and cache compact per-tile features on CPU."""
    model.encoder.eval()
    feature_batches: list[Any] = []
    label_batches: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features = model.chest_state_head[0](model.encoder(images))
            feature_batches.append(features.cpu())
            label_batches.append(batch["chest_state_grid"].cpu())
    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    dataset = torch.utils.data.TensorDataset(features, labels)
    print(f"cached_chest_features={len(dataset)}", flush=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate_chest_head(model: Any, loader: Any, *, device: Any) -> dict[str, float]:
    """Evaluate the independently trainable chest head without running unrelated heads."""
    torch, nn, _ = _torch_modules()
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    batches = 0
    total = 0
    correct = 0
    closed_hits = 0
    closed_total = 0
    open_hits = 0
    open_total = 0
    false_positives = 0
    none_total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                features, labels = batch
                logits = model.chest_state_head(features.to(device))
                labels = labels.to(device)
            else:
                images = batch["image"].to(device)
                labels = batch["chest_state_grid"].to(device)
                logits = model.chest_state_head(model.encoder(images))
            predictions = logits.argmax(dim=1)
            total_loss += float(loss_fn(logits, labels).cpu())
            batches += 1
            correct += int((predictions == labels).sum().cpu())
            total += int(labels.numel())
            closed_mask = labels == CHEST_STATE_TO_INDEX["closed"]
            open_mask = labels == CHEST_STATE_TO_INDEX["opened"]
            none_mask = labels == CHEST_STATE_TO_INDEX["none"]
            closed_hits += int((predictions[closed_mask] == CHEST_STATE_TO_INDEX["closed"]).sum().cpu())
            closed_total += int(closed_mask.sum().cpu())
            open_hits += int((predictions[open_mask] == CHEST_STATE_TO_INDEX["opened"]).sum().cpu())
            open_total += int(open_mask.sum().cpu())
            false_positives += int((predictions[none_mask] != CHEST_STATE_TO_INDEX["none"]).sum().cpu())
            none_total += int(none_mask.sum().cpu())

    return {
        "val_loss": total_loss / max(1, batches),
        "chest_state_acc": correct / max(1, total),
        "chest_closed_recall": closed_hits / max(1, closed_total),
        "chest_open_recall": open_hits / max(1, open_total),
        "chest_false_positive_rate": false_positives / max(1, none_total),
    }


def evaluate_model_variants(
    model: Any,
    dataset_path: str | Path,
    *,
    device: Any,
    batch_size: int,
    variants: Sequence[str] = IMAGE_VARIANTS,
) -> dict[str, dict[str, float]]:
    torch, _, _ = _torch_modules()
    data = np.load(dataset_path)
    indices = np.arange(data["images"].shape[0])
    results: dict[str, dict[str, float]] = {}
    for variant in variants:
        ds = PerceptionDataset(dataset_path, indices=indices, augment=False, variants=(variant,))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        results[variant] = evaluate_model(model, loader, device=device)
    return results


def _tile_class_weights(grids: np.ndarray) -> Any:
    torch, _, _ = _torch_modules()
    counts = np.bincount(grids.reshape(-1).astype(np.int64), minlength=NUM_TILE_CLASSES).astype(np.float32)
    # 频率越低权重越高；sqrt 可以避免极少类权重过大导致训练不稳定。
    counts += 1.0
    weights = np.sqrt(counts.sum() / counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def _exit_type_class_weights(exit_type_grids: np.ndarray) -> Any:
    torch, _, _ = _torch_modules()
    counts = np.bincount(
        exit_type_grids.reshape(-1).astype(np.int64),
        minlength=NUM_EXIT_TYPE_CLASSES,
    ).astype(np.float32)
    counts += 1.0
    weights = np.sqrt(counts.sum() / counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def _chest_state_class_weights(chest_state_grids: np.ndarray) -> Any:
    torch, _, _ = _torch_modules()
    counts = np.bincount(
        chest_state_grids.reshape(-1).astype(np.int64),
        minlength=NUM_CHEST_STATE_CLASSES,
    ).astype(np.float32)
    counts += 1.0
    weights = np.sqrt(counts.sum() / counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def load_model(weights_path: str | Path = DEFAULT_WEIGHTS, *, device: str | None = None) -> tuple[Any, Any]:
    torch, _, _ = _torch_modules()
    resolved_device = _resolve_device(torch, device)
    try:
        checkpoint = torch.load(weights_path, map_location=resolved_device, weights_only=True)
    except TypeError:
        # 兼容旧版 PyTorch：旧版本 torch.load 还没有 weights_only 参数。
        checkpoint = torch.load(weights_path, map_location=resolved_device)
    model = TinyPerceptionCNN(
        num_classes=int(checkpoint.get("num_tile_classes", NUM_TILE_CLASSES)),
        num_exit_type_classes=int(checkpoint.get("num_exit_type_classes", NUM_EXIT_TYPE_CLASSES)),
        num_chest_state_classes=int(checkpoint.get("num_chest_state_classes", NUM_CHEST_STATE_CLASSES)),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    allowed_missing = {
        name
        for name in incompatible.missing_keys
        if name.startswith("exit_type_head.")
        or name.startswith("chest_state_head.")
        or name.startswith("offset_head.")
        or name.startswith("player_refine_head.")
        or name.startswith("monster_refine_head.")
    }
    model.exit_type_head_available = not any(
        name.startswith("exit_type_head.") for name in incompatible.missing_keys
    )
    model.has_trained_chest_state_head = not any(
        name.startswith("chest_state_head.") for name in incompatible.missing_keys
    )
    offset_weights_available = not any(
        name.startswith("offset_head.") for name in incompatible.missing_keys
    )
    refine_weights_available = not any(
        name.startswith("player_refine_head.")
        or name.startswith("monster_refine_head.")
        for name in incompatible.missing_keys
    )
    model.has_trained_offset_head = bool(
        checkpoint.get("has_trained_offset_head", offset_weights_available)
    ) and offset_weights_available
    model.has_trained_refine_head = bool(
        checkpoint.get("has_trained_refine_head", refine_weights_available)
    ) and refine_weights_available
    unexpected = list(incompatible.unexpected_keys)
    missing = [name for name in incompatible.missing_keys if name not in allowed_missing]
    if unexpected or missing:
        raise RuntimeError(
            f"perception checkpoint incompatible: missing={missing}, unexpected={unexpected}"
        )
    model.to(resolved_device)
    model.eval()
    return model, torch


def _warm_start_model(model: Any, weights_path: Path, torch: Any, device: Any) -> bool:
    if not weights_path.exists():
        return False
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        return False

    current_state = model.state_dict()
    compatible_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and tuple(tensor.shape) == tuple(current_state[name].shape)
    }
    if not compatible_state:
        return False
    model.load_state_dict(compatible_state, strict=False)
    model.offset_head_warm_started = bool(
        checkpoint.get("has_trained_offset_head", True)
    ) and all(
        name in compatible_state
        for name in current_state
        if name.startswith("offset_head.")
    )
    model.refine_head_warm_started = bool(
        checkpoint.get("has_trained_refine_head", True)
    ) and all(
        name in compatible_state
        for name in current_state
        if name.startswith("player_refine_head.")
        or name.startswith("monster_refine_head.")
    )
    print(f"warm_start_params={len(compatible_state)}", flush=True)
    return True


def predict_frame(
    model: Any,
    frame: np.ndarray,
    *,
    torch_module: Any | None = None,
    device: str | None = None,
    confidence_threshold: float = 0.35,
) -> dict[str, Any]:
    torch = torch_module if torch_module is not None else _torch_modules()[0]
    resolved_device = _resolve_device(torch, device)
    image = apply_image_variant(frame, "default")
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).float().to(resolved_device)
    with torch.no_grad():
        output = model(tensor)
        grid = output["tile_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if model.exit_type_head_available and "exit_type_logits" in output:
            exit_type_probabilities = (
                torch.softmax(output["exit_type_logits"], dim=1)[0].cpu().numpy()
            )
            exit_type_grid = exit_type_probabilities.argmax(axis=0).astype(np.uint8)
        else:
            exit_type_probabilities = None
            exit_type_grid = _fallback_exit_type_grids(grid[None, ...])[0]
        if getattr(model, "has_trained_chest_state_head", True) and "chest_state_logits" in output:
            chest_state_grid = output["chest_state_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        else:
            chest_state_grid = _fallback_chest_state_grids(grid[None, ...])[0]
        heatmaps = torch.sigmoid(output["heatmap_logits"])[0].cpu().numpy()
        offset_map = (
            torch.tanh(output["offset_logits"])[0].cpu().numpy()
            * ENTITY_OFFSET_SCALE
            if getattr(model, "has_trained_offset_head", False)
            else None
        )

    player_conf = float(heatmaps[0].max())
    player = None
    if player_conf >= confidence_threshold:
        y, x = np.unravel_index(int(np.argmax(heatmaps[0])), heatmaps[0].shape)
        center_x, center_y = _refine_peak_center(heatmaps[0], int(x), int(y))
        if offset_map is not None:
            center_x, center_y = _center_from_offset_prediction(
                (center_x, center_y), offset_map, channel_offset=0
            )
        player = {"center_px": (center_x, center_y), "confidence": player_conf}
    elif np.any(grid == TILE_PLAYER):
        y, x = np.argwhere(grid == TILE_PLAYER)[0]
        player = {
            "center_px": (float(x * TILE_SIZE + TILE_SIZE * 0.5), float(y * TILE_SIZE + TILE_SIZE * 0.5)),
            "confidence": 0.0,
        }

    player_tile = None
    if player is not None:
        player_tile = (
            int(player["center_px"][0] // TILE_SIZE),
            int(player["center_px"][1] // TILE_SIZE),
        )

    monsters = []
    for x, y, conf in _peaks_from_heatmap(heatmaps[1], threshold=confidence_threshold, max_peaks=8):
        center_x, center_y = _refine_peak_center(
            heatmaps[1],
            x,
            y,
            radius=MONSTER_CENTER_WINDOW_RADIUS,
            power=MONSTER_CENTER_POWER,
        )
        if offset_map is not None:
            center_x, center_y = _center_from_offset_prediction(
                (center_x, center_y), offset_map, channel_offset=2
            )
        monsters.append(
            {
                "center_px": (center_x, center_y),
                "confidence": float(conf),
                "entity_type": "unknown",
            }
        )
    if not monsters:
        for y, x in np.argwhere(grid == TILE_MONSTER):
            monsters.append(
                {
                    "center_px": (float(x * TILE_SIZE + TILE_SIZE * 0.5), float(y * TILE_SIZE + TILE_SIZE * 0.5)),
                    "confidence": 0.0,
                    "entity_type": "unknown",
                }
            )

    return {
        "grid": grid,
        "exit_type_grid": exit_type_grid,
        "exit_types": _exit_types_from_prediction(
            grid,
            exit_type_grid,
            exit_type_probabilities=exit_type_probabilities,
            player_tile=player_tile,
        ),
        "chest_state_grid": chest_state_grid,
        "closed_chests": _tiles_from_class_grid(chest_state_grid, CHEST_STATE_TO_INDEX["closed"]),
        "opened_chests": _tiles_from_class_grid(chest_state_grid, CHEST_STATE_TO_INDEX["opened"]),
        "player": player,
        "monsters": tuple(monsters),
        "heatmaps": heatmaps,
    }


def _tiles_from_class_grid(grid: np.ndarray, class_index: int) -> set[tuple[int, int]]:
    rows, cols = np.where(grid == class_index)
    return {(int(col), int(row)) for row, col in zip(rows, cols)}


def _collect_builtin_samples(
    *,
    target_count: int,
    rng: random.Random,
    images: list[np.ndarray],
    redraw_images: dict[str, list[np.ndarray]],
    grids: list[np.ndarray],
    exit_type_grids: list[np.ndarray],
    chest_state_grids: list[np.ndarray],
    player_centers: list[np.ndarray],
    monster_centers: list[np.ndarray],
    monster_masks: list[np.ndarray],
    max_monsters: int,
) -> None:
    """Collect generic short rollouts so dynamic entities are not only seen at rest."""

    from nesylink.env import make_env

    if target_count <= 0:
        return
    made = 0
    while made < target_count:
        task_ids = list(BUILTIN_TASKS)
        rng.shuffle(task_ids)
        for task_id in task_ids:
            env = make_env(task_id=task_id, observation_mode="full", render_mode="rgb_array")
            env.reset(seed=rng.randrange(10_000_000))
            runtime = env.engine.runtime
            room_ids = list(runtime.room_manager.room_ids)
            monster_rooms = [
                room_id
                for room_id in room_ids
                if runtime.room_manager.get_room(
                    runtime.room_manager.coord_for_room_id(room_id)
                ).monsters
            ]
            room_id = rng.choice(monster_rooms if monster_rooms and rng.random() < 0.75 else room_ids)
            runtime.room_coord = runtime.room_manager.coord_for_room_id(room_id)
            runtime.room = runtime.room_manager.get_room(runtime.room_coord)
            _randomize_monster_poses(env, rng)
            if runtime.room.monsters and rng.random() < 0.85:
                _position_player_near_monster(env, rng)
            else:
                _randomize_player_pose(env, rng)
            _randomize_visible_chest_states(env, rng)
            burst = rng.randint(8, 28) if runtime.room.monsters else rng.randint(2, 6)
            for _ in range(burst):
                _append_sample(
                    env,
                    {},
                    images,
                    redraw_images,
                    grids,
                    exit_type_grids,
                    chest_state_grids,
                    player_centers,
                    monster_centers,
                    monster_masks,
                    max_monsters,
                )
                made += 1
                if made >= target_count:
                    env.close()
                    return
                actions = list(range(env.action_space.n))
                base_weights = [0.06, 0.13, 0.13, 0.13, 0.13, 0.20, 0.22]
                weights = base_weights[: len(actions)] + [0.10] * max(
                    0, len(actions) - len(base_weights)
                )
                action = int(rng.choices(actions, weights=weights, k=1)[0])
                _label, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            env.close()


def _collect_exit_overlap_samples(
    *,
    target_count: int,
    rng: random.Random,
    images: list[np.ndarray],
    redraw_images: dict[str, list[np.ndarray]],
    grids: list[np.ndarray],
    exit_type_grids: list[np.ndarray],
    chest_state_grids: list[np.ndarray],
    player_centers: list[np.ndarray],
    monster_centers: list[np.ndarray],
    monster_masks: list[np.ndarray],
    max_monsters: int,
) -> None:
    """Collect hard negatives where the moving player visually overlaps an exit."""
    from nesylink.core.observation import build_observation
    from nesylink.env import make_env

    made = 0
    while made < target_count:
        task_ids = list(BUILTIN_TASKS)
        rng.shuffle(task_ids)
        for task_id in task_ids:
            env = make_env(task_id=task_id, observation_mode="full", render_mode="rgb_array")
            env.reset(seed=rng.randrange(10_000_000))
            runtime = env.engine.runtime
            room_ids = list(runtime.room_manager.room_ids)
            rng.shuffle(room_ids)
            for room_id in room_ids:
                coord = runtime.room_manager.coord_for_room_id(room_id)
                runtime.room_coord = coord
                runtime.room = runtime.room_manager.get_room(coord)
                exit_tiles = [tile for exit_config in runtime.room.exits for tile in exit_config.tiles]
                rng.shuffle(exit_tiles)
                for col, row in exit_tiles:
                    offset_x = rng.uniform(-7.95, 7.95)
                    offset_y = rng.uniform(-7.95, 7.95)
                    runtime.player.position_px = (
                        float(np.clip(col * TILE_SIZE + offset_x, 0, MAP_PIXEL_WIDTH - TILE_SIZE)),
                        float(np.clip(row * TILE_SIZE + offset_y, 0, MAP_PIXEL_HEIGHT - TILE_SIZE)),
                    )
                    runtime.player.facing = rng.choice(("up", "down", "left", "right"))
                    _randomize_player_action(runtime.player, rng)
                    _randomize_visible_chest_states(env, rng)
                    label = build_observation(runtime.room, runtime.player, max_monsters)
                    _append_sample(
                        env,
                        label,
                        images,
                        redraw_images,
                        grids,
                        exit_type_grids,
                        chest_state_grids,
                        player_centers,
                        monster_centers,
                        monster_masks,
                        max_monsters,
                    )
                    made += 1
                    if made >= target_count:
                        env.close()
                        return
            env.close()


def _collect_random_room_samples(
    *,
    target_count: int,
    rng: random.Random,
    images: list[np.ndarray],
    redraw_images: dict[str, list[np.ndarray]],
    grids: list[np.ndarray],
    exit_type_grids: list[np.ndarray],
    chest_state_grids: list[np.ndarray],
    player_centers: list[np.ndarray],
    monster_centers: list[np.ndarray],
    monster_masks: list[np.ndarray],
    max_monsters: int,
    data_dir: Path,
) -> None:
    from nesylink.env import make_env

    if target_count <= 0:
        return
    map_dir = data_dir / "generated_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    while made < target_count:
        room_path = map_dir / f"random_room_{made:05d}.json"
        room_path.write_text(json.dumps(_random_room_payload(rng), indent=2), encoding="utf-8")
        env = make_env(map_path=room_path, observation_mode="full", render_mode="rgb_array", max_monsters=max_monsters)
        env.reset(seed=rng.randrange(10_000_000))
        _randomize_monster_poses(env, rng)
        if env.engine.runtime.room.monsters and rng.random() < 0.80:
            _position_player_near_monster(env, rng)
        else:
            _randomize_player_pose(env, rng)
        _randomize_visible_chest_states(env, rng)
        for _ in range(rng.randint(8, 24)):
            _randomize_visible_chest_states(env, rng)
            _append_sample(
                env,
                {},
                images,
                redraw_images,
                grids,
                exit_type_grids,
                chest_state_grids,
                player_centers,
                monster_centers,
                monster_masks,
                max_monsters,
            )
            made += 1
            if made >= target_count:
                break
            actions = list(range(env.action_space.n))
            base_weights = [0.06, 0.13, 0.13, 0.13, 0.13, 0.20, 0.22]
            weights = base_weights[: len(actions)] + [0.10] * max(0, len(actions) - len(base_weights))
            action = int(rng.choices(actions, weights=weights, k=1)[0])
            _label, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        env.close()


def _append_sample(
    env: Any,
    label: dict[str, np.ndarray],
    images: list[np.ndarray],
    redraw_images: dict[str, list[np.ndarray]],
    grids: list[np.ndarray],
    exit_type_grids: list[np.ndarray],
    chest_state_grids: list[np.ndarray],
    player_centers: list[np.ndarray],
    monster_centers: list[np.ndarray],
    monster_masks: list[np.ndarray],
    max_monsters: int,
) -> None:
    del label
    frame = env.render()[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH].astype(np.uint8)
    from nesylink.core.observation import room_observation
    from utils.evaluate_policy import redraw_obs_from_state

    runtime = env.engine.runtime
    grid = room_observation(runtime.room, runtime.player).astype(np.uint8)
    exit_type_grid = _exit_type_grid_from_env(env)
    chest_state_grid = _chest_state_grid_from_env(env)
    player_center = np.asarray(runtime.player.position_px, dtype=np.float32) + runtime.player.size_px * 0.5
    centers = np.full((max_monsters, 2), -1.0, dtype=np.float32)
    masks = np.zeros((max_monsters,), dtype=np.bool_)
    for index, monster in enumerate(runtime.room.monsters.values()):
        if index >= max_monsters:
            break
        centers[index] = np.asarray(monster.position_px, dtype=np.float32) + monster.size_px * 0.5
        masks[index] = True

    images.append(frame)
    for variant, variant_images in redraw_images.items():
        preset = variant.removeprefix("redraw_")
        variant_images.append(redraw_obs_from_state(env, preset=preset, shape=frame.shape))
    grids.append(grid)
    exit_type_grids.append(exit_type_grid)
    chest_state_grids.append(chest_state_grid)
    player_centers.append(player_center)
    monster_centers.append(centers)
    monster_masks.append(masks)


def _random_room_payload(rng: random.Random) -> dict[str, Any]:
    occupied: set[tuple[int, int]] = set()
    spawn = (rng.randrange(1, GRID_WIDTH - 1), rng.randrange(1, GRID_HEIGHT - 1))
    occupied.add(spawn)
    layout: list[str] = []
    exit_floor = {
        (4, 0),
        (5, 0),
        (4, GRID_HEIGHT - 1),
        (5, GRID_HEIGHT - 1),
        (0, 3),
        (0, 4),
        (GRID_WIDTH - 1, 3),
        (GRID_WIDTH - 1, 4),
    }
    for y in range(GRID_HEIGHT):
        row = []
        for x in range(GRID_WIDTH):
            tile = (x, y)
            wall = rng.random() < 0.14 and tile not in exit_floor and tile != spawn
            row.append("#" if wall else ".")
            if wall:
                occupied.add(tile)
        layout.append("".join(row))

    objects: list[dict[str, Any]] = []
    free_tiles = [(x, y) for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH) if (x, y) not in occupied]
    rng.shuffle(free_tiles)

    def take_tile() -> tuple[int, int] | None:
        while free_tiles:
            tile = free_tiles.pop()
            if tile not in occupied:
                occupied.add(tile)
                return tile
        return None

    chest_positions: list[tuple[int, int]] = []
    for idx in range(rng.randint(1, 4)):
        if (pos := take_tile()) is not None:
            objects.append(
                {
                    "id": f"chest_{idx}",
                    "kind": "chest",
                    "pos": list(pos),
                    "loot": {"kind": rng.choice(("key", "gold", "heal"))},
                }
            )
            chest_positions.append(pos)
    for idx in range(rng.randint(2, 8)):
        if (pos := take_tile()) is not None:
            objects.append({"id": f"trap_{idx}", "kind": "trap", "pos": list(pos), "damage": 1})
    for idx in range(rng.randint(0, 2)):
        if (pos := take_tile()) is not None:
            objects.append({"id": f"button_{idx}", "kind": "button", "pos": list(pos)})
    for idx in range(rng.randint(1, 4)):
        if (pos := take_tile()) is not None:
            objects.append({"id": f"npc_{idx}", "kind": "npc", "pos": list(pos), "text": "..."})
    for idx in range(rng.randint(0, 3)):
        if (pos := take_tile()) is not None:
            objects.append(
                {
                    "id": f"monster_{idx}",
                    "kind": "monster",
                    "pos": list(pos),
                    "monster_type": rng.choice(["chaser", "ambusher", "patroller"]),
                    "hp": rng.choice([1, 2, 3]),
                }
            )

    exits = []
    if rng.random() < 0.65:
        for direction in rng.sample(["north", "south", "west", "east"], rng.randint(1, 2)):
            exit_type = rng.choice(["normal", "locked_key", "conditional"])
            exit_payload = {
                "id": f"{direction}_exit",
                "direction": direction,
                "target_room": "random_room",
                "target_entry": "default",
                "type": exit_type,
            }
            if exit_type == "locked_key":
                exit_payload["requires"] = {"key_count": 1, "consume_key": False}
            elif exit_type == "conditional":
                exit_payload["requires"] = {"all_monsters_defeated": True}
            exits.append(exit_payload)

    dynamic_objects = []
    if rng.random() < 0.55:
        dynamic_tiles = []
        for _ in range(rng.randint(2, 8)):
            if (pos := take_tile()) is not None:
                dynamic_tiles.append(list(pos))
        if chest_positions and rng.random() < 0.75:
            # 独立 chest head 必须学习同一 tile 可同时包含 bridge 和 chest。
            dynamic_tiles.append(list(rng.choice(chest_positions)))
        if dynamic_tiles:
            dynamic_objects.append(
                {
                    "id": "random_bridge",
                    "kind": "rotating_bridge",
                    "initial_state": "active",
                    "background_tile": "gap",
                    "active_tile": "bridge",
                    "states": {"active": {"tiles": dynamic_tiles}},
                }
            )

    return {
        "id": "random_room",
        "coord": [0, 0],
        "layout": layout,
        "spawns": {"default": list(spawn)},
        "default_spawn": "default",
        "objects": objects,
        "dynamic_objects": dynamic_objects,
        "exits": exits,
    }


def _exit_type_grid_from_env(env: Any) -> np.ndarray:
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    room = env.engine.runtime.room
    for exit_config in room.exits:
        exit_type = str(getattr(exit_config, "exit_type", "normal")).lower()
        type_index = EXIT_TYPE_TO_INDEX.get(exit_type, EXIT_TYPE_TO_INDEX["normal"])
        for col, row in exit_config.tiles:
            grid[row, col] = type_index
    return grid


def _chest_state_grid_from_env(env: Any) -> np.ndarray:
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
    room = env.engine.runtime.room
    for chest in room.chests.values():
        if not chest.is_visible:
            continue
        state = "opened" if chest.is_open else "closed"
        col, row = chest.pos
        grid[row, col] = CHEST_STATE_TO_INDEX[state]
    return grid


def _randomize_visible_chest_states(env: Any, rng: random.Random) -> None:
    """为训练采集生成关闭/打开宝箱帧；不在推理阶段使用。"""

    for chest in env.engine.runtime.room.chests.values():
        if not chest.is_visible and chest.reveal_on:
            chest.is_visible = True
        if chest.is_visible:
            chest.is_open = rng.random() < 0.45


def _position_player_near_monster(env: Any, rng: random.Random) -> None:
    """Start a generic physics rollout near a moving entity, without map assumptions."""

    from nesylink.core.state import move_with_tile_collisions, tile_to_top_left_px

    runtime = env.engine.runtime
    room = runtime.room
    if not room.monsters:
        _randomize_player_pose(env, rng)
        return
    monster = rng.choice(tuple(room.monsters.values()))
    blocked = set(room.runtime_blocking_tiles()) | {
        item.tile_pos for item in room.monsters.values()
    }
    candidates = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) not in blocked
        and 1 <= abs(x - monster.tile_pos[0]) + abs(y - monster.tile_pos[1]) <= 4
    ]
    if not candidates:
        _randomize_player_pose(env, rng)
        return
    tile = rng.choice(candidates)
    base_position = tile_to_top_left_px(tile)
    runtime.player.position_px = move_with_tile_collisions(
        base_position,
        runtime.player.size_px,
        (_sample_entity_offset(rng), _sample_entity_offset(rng)),
        room.runtime_blocking_tiles(),
    )
    delta_x = monster.tile_pos[0] - tile[0]
    delta_y = monster.tile_pos[1] - tile[1]
    if abs(delta_x) > abs(delta_y):
        runtime.player.facing = "right" if delta_x > 0 else "left"
    elif delta_y:
        runtime.player.facing = "down" if delta_y > 0 else "up"
    else:
        runtime.player.facing = rng.choice(("up", "down", "left", "right"))
    _randomize_player_action(runtime.player, rng)


def _center_from_offset_prediction(
    coarse_center: tuple[float, float],
    offset_map: np.ndarray,
    *,
    channel_offset: int,
) -> tuple[float, float]:
    cell_x = int(
        np.clip(
            coarse_center[0] // ENTITY_OFFSET_STRIDE,
            0,
            offset_map.shape[2] - 1,
        )
    )
    cell_y = int(
        np.clip(
            coarse_center[1] // ENTITY_OFFSET_STRIDE,
            0,
            offset_map.shape[1] - 1,
        )
    )
    base_x = cell_x * ENTITY_OFFSET_STRIDE + ENTITY_OFFSET_STRIDE * 0.5
    base_y = cell_y * ENTITY_OFFSET_STRIDE + ENTITY_OFFSET_STRIDE * 0.5
    return (
        float(base_x + offset_map[channel_offset, cell_y, cell_x]),
        float(base_y + offset_map[channel_offset + 1, cell_y, cell_x]),
    )


def _randomize_player_pose(env: Any, rng: random.Random) -> None:
    """Cover every within-tile pixel offset plus sword/shield render poses."""
    from nesylink.core.state import move_with_tile_collisions, tile_to_top_left_px

    runtime = env.engine.runtime
    room = runtime.room
    occupied = set(room.runtime_blocking_tiles()) | {monster.tile_pos for monster in room.monsters.values()}
    candidates = [
        (x, y)
        for y in range(GRID_HEIGHT)
        for x in range(GRID_WIDTH)
        if (x, y) not in occupied
    ]
    if not candidates:
        return
    tile = rng.choice(candidates)
    base_position = tile_to_top_left_px(tile)
    runtime.player.position_px = move_with_tile_collisions(
        base_position,
        runtime.player.size_px,
        (_sample_entity_offset(rng), _sample_entity_offset(rng)),
        room.runtime_blocking_tiles(),
    )
    runtime.player.facing = rng.choice(("up", "down", "left", "right"))
    _randomize_player_action(runtime.player, rng)


def _sample_entity_offset(rng: random.Random) -> float:
    """Mix exact integer phases, near-boundary phases, and continuous positions."""

    draw = rng.random()
    if draw < 0.50:
        return float(rng.randint(-8, 7))
    if draw < 0.75:
        return float(rng.choice((-7.95, -7.5, -7.0, 6.0, 7.0, 7.5, 7.95)))
    return float(rng.uniform(-7.95, 7.95))


def _randomize_monster_poses(env: Any, rng: random.Random) -> None:
    """Expose the detector to legal monster phases around tile boundaries."""

    from nesylink.core.state import move_with_tile_collisions, tile_to_top_left_px

    room = env.engine.runtime.room
    monsters = tuple(room.monsters.values())
    occupied = {monster.tile_pos for monster in monsters}
    for monster in monsters:
        if rng.random() >= 0.80:
            continue
        origin_tile = monster.tile_pos
        blockers = set(room.runtime_blocking_tiles()) | occupied
        blockers.discard(origin_tile)
        monster.position_px = move_with_tile_collisions(
            tile_to_top_left_px(origin_tile),
            monster.size_px,
            (_sample_entity_offset(rng), _sample_entity_offset(rng)),
            blockers,
        )
        occupied.discard(origin_tile)
        occupied.add(monster.tile_pos)


def _randomize_player_action(player: Any, rng: random.Random) -> None:
    if rng.random() < 0.45:
        item_name = rng.choice(("sword", "shield"))
        player.start_action(
            item_name=item_name,
            pose=item_name,
            facing=player.facing,
            ticks=rng.randint(1, 8),
        )
    else:
        player.clear_action()


def _fallback_exit_type_grids(grids: np.ndarray) -> np.ndarray:
    exit_type_grids = np.zeros_like(grids, dtype=np.uint8)
    exit_type_grids[np.asarray(grids) == TILE_EXIT] = EXIT_TYPE_TO_INDEX["normal"]
    return exit_type_grids


def _fallback_chest_state_grids(grids: np.ndarray) -> np.ndarray:
    chest_state_grids = np.zeros_like(grids, dtype=np.uint8)
    chest_state_grids[np.asarray(grids) == TILE_CHEST] = CHEST_STATE_TO_INDEX["closed"]
    return chest_state_grids


def _exit_types_from_prediction(
    grid: np.ndarray,
    exit_type_grid: np.ndarray,
    *,
    exit_type_probabilities: np.ndarray | None = None,
    player_tile: tuple[int, int] | None = None,
) -> dict[tuple[int, int], str]:
    """Fuse the tile and exit heads so a player cannot visually erase an exit."""

    if exit_type_probabilities is None:
        exit_tiles = {
            (int(col), int(row))
            for row, col in np.argwhere(grid == TILE_EXIT)
        }
    else:
        exit_confidences = 1.0 - exit_type_probabilities[0]
        bridge_rows, bridge_cols = np.where(grid == TILE_BRIDGE)
        bridge_tiles = {
            (int(col), int(row)) for row, col in zip(bridge_rows, bridge_cols)
        }
        if bridge_tiles:
            exit_tiles = _decode_exit_side_pairs(exit_confidences, bridge_tiles)
        else:
            exit_tiles = {
                (int(col), int(row))
                for row, col in np.argwhere(grid == TILE_EXIT)
            }
        if bridge_tiles and player_tile is not None:
            _restore_player_occluded_exit_pair(
                exit_tiles,
                exit_confidences,
                player_tile,
            )
    if exit_type_probabilities is not None and player_tile is not None:
        col, row = player_tile
        inside = 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        if inside and int(grid[row, col]) == TILE_PLAYER:
            exit_confidence = 1.0 - float(exit_type_probabilities[0, row, col])
            if exit_confidence >= EXIT_OCCLUSION_CONFIDENCE:
                exit_tiles.add((col, row))

    exit_types: dict[tuple[int, int], str] = {}
    for col, row in exit_tiles:
        if exit_type_probabilities is None:
            type_index = int(exit_type_grid[row, col])
        else:
            type_index = 1 + int(np.argmax(exit_type_probabilities[1:, row, col]))
        if type_index <= EXIT_TYPE_TO_INDEX["none"] or type_index >= len(EXIT_TYPE_NAMES):
            exit_type = "normal"
        else:
            exit_type = EXIT_TYPE_NAMES[type_index]
        exit_types[(col, row)] = exit_type
    return exit_types


def _decode_exit_side_pairs(
    confidences: np.ndarray,
    bridge_tiles: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Decode the learned edge scores into one two-cell doorway per side."""

    height, width = confidences.shape
    sides = (
        [(x, 0) for x in range(width)],
        [(x, height - 1) for x in range(width)],
        [(0, y) for y in range(height)],
        [(width - 1, y) for y in range(height)],
    )
    result: set[tuple[int, int]] = set()
    for positions in sides:
        middle = (len(positions) - 1) * 0.5
        candidates = []
        for index in range(len(positions) - 1):
            first, second = positions[index : index + 2]
            score = 0.5 * (
                float(confidences[first[1], first[0]])
                + float(confidences[second[1], second[0]])
            )
            candidates.append((score, -abs(index + 0.5 - middle), first, second))
        score, _center_bias, first, second = max(candidates)
        bridge_reaches_side = any(
            abs(edge[0] - bridge[0]) + abs(edge[1] - bridge[1]) <= 1
            for edge in positions
            for bridge in bridge_tiles
        )
        threshold = (
            EXIT_BRIDGE_OVERLAP_CONFIDENCE
            if bridge_reaches_side
            else EXIT_SIDE_PAIR_CONFIDENCE
        )
        if score >= threshold:
            result.update((first, second))
    return result


def _restore_player_occluded_exit_pair(
    exit_tiles: set[tuple[int, int]],
    confidences: np.ndarray,
    player_tile: tuple[int, int],
) -> None:
    """Keep a learned two-cell doorway stable near an occluding player."""

    height, width = confidences.shape
    x, y = player_tile
    candidates: list[
        tuple[float, float, tuple[int, int], tuple[int, int]]
    ] = []

    def add_vertical(side_x: int) -> None:
        middle = (height - 1) * 0.5
        projected = (side_x, y)
        for candidate_y in (y - 1, y + 1):
            if 0 <= candidate_y < height:
                companion = (side_x, candidate_y)
                candidates.append(
                    (
                        float(confidences[candidate_y, side_x]),
                        -abs(candidate_y - middle),
                        projected,
                        companion,
                    )
                )

    def add_horizontal(side_y: int) -> None:
        middle = (width - 1) * 0.5
        projected = (x, side_y)
        for candidate_x in (x - 1, x + 1):
            if 0 <= candidate_x < width:
                companion = (candidate_x, side_y)
                candidates.append(
                    (
                        float(confidences[side_y, candidate_x]),
                        -abs(candidate_x - middle),
                        projected,
                        companion,
                    )
                )

    if x <= 1:
        add_vertical(0)
    if x >= width - 2:
        add_vertical(width - 1)
    if y <= 1:
        add_horizontal(0)
    if y >= height - 2:
        add_horizontal(height - 1)
    if not candidates:
        return
    confidence, _center_bias, projected, companion = max(candidates)
    if (
        projected == player_tile
        or confidence >= EXIT_BRIDGE_OVERLAP_CONFIDENCE
    ):
        exit_tiles.update((projected, companion))


def _make_heatmaps(
    player_center: np.ndarray,
    monster_centers: np.ndarray,
    monster_mask: np.ndarray,
    *,
    sigma: float = 3.0,
) -> np.ndarray:
    heatmaps = np.zeros((2, MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH), dtype=np.float32)
    _draw_gaussian(heatmaps[0], player_center, sigma=sigma)
    for center, active in zip(monster_centers, monster_mask):
        if active and center[0] >= 0 and center[1] >= 0:
            _draw_gaussian(heatmaps[1], center, sigma=sigma)
    return heatmaps


def _draw_gaussian(heatmap: np.ndarray, center: np.ndarray, *, sigma: float) -> None:
    cx, cy = float(center[0]), float(center[1])
    radius = int(max(2, sigma * 3))
    x0, x1 = max(0, int(cx) - radius), min(MAP_PIXEL_WIDTH, int(cx) + radius + 1)
    y0, y1 = max(0, int(cy) - radius), min(MAP_PIXEL_HEIGHT, int(cy) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    patch = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], patch)


def apply_image_variant(image: np.ndarray, variant: str) -> np.ndarray:
    """Apply one of the supported color/brightness variants and return float RGB."""
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image variant input must be HxWx3 RGB, got shape={arr.shape}")
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0

    if variant == "default":
        out = arr
    elif variant == "grayscale":
        gray = arr.mean(axis=2)
        out = np.repeat(gray[..., None], 3, axis=2)
    elif variant == "dark":
        out = arr * 0.55
    elif variant == "bright":
        out = arr * 1.35
    elif variant == "high_contrast":
        out = (arr > (127.0 / 255.0)).astype(np.float32)
    elif variant == "inverted":
        out = 1.0 - arr
    else:
        raise ValueError(f"unsupported image variant: {variant}")
    out = np.clip(out, 0.0, 1.0)
    if variant in {"grayscale", "dark", "bright"}:
        out = np.floor(out * 255.0 + 1e-5) / 255.0
    return out.astype(np.float32)


def _rgb_to_luma(image: np.ndarray) -> np.ndarray:
    return np.tensordot(image[..., :3], _LUMA_WEIGHTS, axes=([-1], [0])).astype(np.float32)


def _augment_image(image: np.ndarray) -> np.ndarray:
    # 颜色增强用于避免模型只记住固定 RGB；测试渲染细节变化时会更稳。
    rng = np.random.default_rng()
    out = image.copy()
    if rng.random() < 0.20:
        return out.astype(np.float32, copy=False)
    brightness = rng.uniform(-0.12, 0.12)
    contrast = rng.uniform(0.75, 1.35)
    out = (out - 0.5) * contrast + 0.5 + brightness
    if rng.random() < 0.20:
        gray = out.mean(axis=2, keepdims=True)
        out = gray * rng.uniform(0.6, 1.0) + out * rng.uniform(0.0, 0.4)
    channel_scale = rng.uniform(0.75, 1.25, size=(1, 1, 3))
    out = out * channel_scale
    out += rng.normal(0.0, 0.025, size=out.shape)
    if rng.random() < 0.15:
        height, width = out.shape[:2]
        cut_height = int(rng.integers(2, min(9, height) + 1))
        cut_width = int(rng.integers(4, min(21, width) + 1))
        top = int(rng.integers(0, height - cut_height + 1))
        left = int(rng.integers(0, width - cut_width + 1))
        fill = rng.uniform(0.0, 1.0, size=(1, 1, 3))
        out[top : top + cut_height, left : left + cut_width] = fill
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _peaks_from_heatmap(
    heatmap: np.ndarray,
    *,
    threshold: float,
    max_peaks: int,
    min_distance_px: float = 12.0,
) -> list[tuple[int, int, float]]:
    flat_indices = np.argsort(heatmap.ravel())[::-1]
    peaks: list[tuple[int, int, float]] = []
    for flat in flat_indices:
        conf = float(heatmap.ravel()[flat])
        if conf < threshold or len(peaks) >= max_peaks:
            break
        y, x = np.unravel_index(int(flat), heatmap.shape)
        if all(math.hypot(x - px, y - py) >= min_distance_px for px, py, _ in peaks):
            peaks.append((int(x), int(y), conf))
    return peaks


def _refine_peak_center(
    heatmap: np.ndarray,
    peak_x: int,
    peak_y: int,
    *,
    radius: int = 4,
    power: int = 8,
) -> tuple[float, float]:
    """Refine an integer heatmap maximum to a stable sub-pixel center."""
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)
    patch = np.maximum(heatmap[y0:y1, x0:x1] - 0.05, 0.0) ** power
    total = float(patch.sum())
    if total <= 1e-12:
        return float(peak_x), float(peak_y)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((xs * patch).sum() / total), float((ys * patch).sum() / total)


def _tiles_from_heatmap(
    heatmap: np.ndarray,
    *,
    threshold: float,
    max_peaks: int,
    offset_map: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    return [
        (int(center_x // TILE_SIZE), int(center_y // TILE_SIZE))
        for center_x, center_y in _centers_from_heatmap(
            heatmap,
            threshold=threshold,
            max_peaks=max_peaks,
            offset_map=offset_map,
        )
    ]


def _centers_from_heatmap(
    heatmap: np.ndarray,
    *,
    threshold: float,
    max_peaks: int,
    offset_map: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    centers = []
    for x, y, _confidence in _peaks_from_heatmap(
        heatmap, threshold=threshold, max_peaks=max_peaks
    ):
        center_x, center_y = _refine_peak_center(
            heatmap,
            x,
            y,
            radius=MONSTER_CENTER_WINDOW_RADIUS,
            power=MONSTER_CENTER_POWER,
        )
        if offset_map is not None:
            center_x, center_y = _center_from_offset_prediction(
                (center_x, center_y), offset_map, channel_offset=2
            )
        centers.append((center_x, center_y))
    return centers


def _greedy_center_errors(
    predicted: Sequence[tuple[float, float]],
    truth: Sequence[Sequence[float]],
) -> list[float]:
    candidates = sorted(
        (
            math.hypot(pred_x - float(true_center[0]), pred_y - float(true_center[1])),
            pred_index,
            truth_index,
        )
        for pred_index, (pred_x, pred_y) in enumerate(predicted)
        for truth_index, true_center in enumerate(truth)
    )
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    errors = []
    for error, pred_index, truth_index in candidates:
        if pred_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(pred_index)
        used_truth.add(truth_index)
        errors.append(error)
    return errors


def _resolve_device(torch: Any, device: str | None) -> Any:
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train NesyLink pixel perception CNN.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output", default=str(DEFAULT_DATASET))
    collect_parser.add_argument("--samples", type=int, default=2400)
    collect_parser.add_argument("--builtin-ratio", type=float, default=0.45)
    collect_parser.add_argument("--exit-overlap-ratio", type=float, default=0.15)
    collect_parser.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default=str(DEFAULT_DATASET))
    train_parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    train_parser.add_argument("--epochs", type=int, default=14)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--chest-head-only", action="store_true")

    combo_parser = subparsers.add_parser("collect-train")
    combo_parser.add_argument("--data", default=str(DEFAULT_DATASET))
    combo_parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    combo_parser.add_argument("--samples", type=int, default=2400)
    combo_parser.add_argument("--builtin-ratio", type=float, default=0.45)
    combo_parser.add_argument("--exit-overlap-ratio", type=float, default=0.15)
    combo_parser.add_argument("--epochs", type=int, default=14)
    combo_parser.add_argument("--batch-size", type=int, default=64)
    combo_parser.add_argument("--lr", type=float, default=1e-3)
    combo_parser.add_argument("--seed", type=int, default=0)
    combo_parser.add_argument("--device", default="auto")
    combo_parser.add_argument("--chest-head-only", action="store_true")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--data", default=str(DEFAULT_DATASET))
    eval_parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--variant", choices=(*IMAGE_VARIANTS, "all"), default="default")

    args = parser.parse_args(argv)
    if args.command == "collect":
        path = collect_dataset(
            DatasetConfig(
                output=Path(args.output),
                samples=args.samples,
                builtin_ratio=args.builtin_ratio,
                exit_overlap_ratio=args.exit_overlap_ratio,
                seed=args.seed,
            )
        )
        print(f"saved dataset: {path}")
    elif args.command == "train":
        metrics = train_model(
            args.data,
            args.weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            chest_head_only=args.chest_head_only,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "collect-train":
        path = collect_dataset(
            DatasetConfig(
                output=Path(args.data),
                samples=args.samples,
                builtin_ratio=args.builtin_ratio,
                exit_overlap_ratio=args.exit_overlap_ratio,
                seed=args.seed,
            )
        )
        metrics = train_model(
            path,
            args.weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            chest_head_only=args.chest_head_only,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "eval":
        torch, _, _ = _torch_modules()
        model, _torch = load_model(args.weights, device=args.device)
        resolved_device = _resolve_device(torch, args.device)
        if args.variant == "all":
            metrics = evaluate_model_variants(
                model,
                args.data,
                device=resolved_device,
                batch_size=args.batch_size,
            )
        else:
            data = np.load(args.data)
            indices = np.arange(data["images"].shape[0])
            ds = PerceptionDataset(args.data, indices=indices, augment=False, variants=(args.variant,))
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            metrics = evaluate_model(model, loader, device=resolved_device)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
