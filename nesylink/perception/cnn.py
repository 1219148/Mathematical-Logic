from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nesylink.core.constants import (
    GRID_HEIGHT,
    GRID_WIDTH,
    MAP_PIXEL_HEIGHT,
    MAP_PIXEL_WIDTH,
    TILE_SIZE,
)
from nesylink.core.observation import TILE_EMPTY, TILE_MONSTER, TILE_PLAYER
from nesylink.perception import cnn_base_v14 as _v14
from nesylink.perception.cnn_base_v14 import *  # noqa: F403


# The public API mirrors the original perception module while an independent
# occupancy head removes the single-label ambiguity for moving entities.
NUM_TILE_CLASSES = _v14.NUM_TILE_CLASSES
EXIT_TYPE_NAMES = _v14.EXIT_TYPE_NAMES
EXIT_TYPE_TO_INDEX = _v14.EXIT_TYPE_TO_INDEX
NUM_EXIT_TYPE_CLASSES = _v14.NUM_EXIT_TYPE_CLASSES
CHEST_STATE_NAMES = _v14.CHEST_STATE_NAMES
CHEST_STATE_TO_INDEX = _v14.CHEST_STATE_TO_INDEX
NUM_CHEST_STATE_CLASSES = _v14.NUM_CHEST_STATE_CLASSES
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET = DEFAULT_DATA_DIR / "perception_dataset.npz"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "perception_model.pt"
IMAGE_VARIANTS = _v14.IMAGE_VARIANTS
REDRAW_IMAGE_VARIANTS = _v14.REDRAW_IMAGE_VARIANTS
MAX_TRAINING_VALIDATION_SAMPLES = _v14.MAX_TRAINING_VALIDATION_SAMPLES
ENTITY_OFFSET_SCALE = _v14.ENTITY_OFFSET_SCALE

OCCUPANCY_CHANNELS = ("player", "monster")
NUM_OCCUPANCY_CHANNELS = len(OCCUPANCY_CHANNELS)
OCCUPANCY_LOSS_WEIGHT = 0.75
OCCUPANCY_CARDINALITY_LOSS_WEIGHT = 0.10
OCCUPANCY_MONSTER_THRESHOLD = 0.50
OCCUPANCY_MAX_MONSTERS = 8
MAX_HEAT_OCCUPANCY_MATCH_DISTANCE_PX = TILE_SIZE * 1.25
# Precision-first gates for hypotheses emitted by only one localization head;
# agreement between heatmap and occupancy does not require these fallbacks.
HEAT_ONLY_MONSTER_CONFIDENCE = 0.9999
INDEPENDENT_OCCUPANCY_CONFIDENCE = 0.75
PLAYER_BOUNDARY_CORRECTION_WINDOW_PX = 0.35


@dataclass(frozen=True)
class DatasetConfig:
    output: Path = DEFAULT_DATASET
    samples: int = 2400
    builtin_ratio: float = 0.45
    exit_overlap_ratio: float = 0.15
    max_monsters: int = 6
    seed: int = 0


def _torch_modules() -> tuple[Any, Any, Any]:
    return _v14._torch_modules()


class TinyPerceptionCNN(_v14.TinyPerceptionCNN):
    """Split-V14 plus an independent logical entity-occupancy head.

    Heatmaps still estimate sub-pixel centers.  The new two-channel 8x10 head
    independently predicts whether each logical tile contains a player and/or a
    monster.  It does not share parameters with the semantic tile classifier.
    """

    def __init__(
        self,
        num_classes: int = NUM_TILE_CLASSES,
        num_exit_type_classes: int = NUM_EXIT_TYPE_CLASSES,
        num_chest_state_classes: int = NUM_CHEST_STATE_CLASSES,
    ) -> None:
        torch, nn, _ = _torch_modules()
        super().__init__(
            num_classes=num_classes,
            num_exit_type_classes=num_exit_type_classes,
            num_chest_state_classes=num_chest_state_classes,
        )
        self.occupancy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((GRID_HEIGHT, GRID_WIDTH)),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, NUM_OCCUPANCY_CHANNELS, kernel_size=1),
        )
        self._interpolate = torch.nn.functional.interpolate
        self._concat = torch.cat
        self.has_trained_occupancy_head = False
        self.has_trained_player_offset_head = False
        self.occupancy_head_warm_started = False
        self.initialize_occupancy_from_tile_head()

    def initialize_occupancy_from_tile_head(self) -> None:
        """Give old V14/V15 checkpoints a useful, shape-compatible head start."""

        torch, _, _ = _torch_modules()
        with torch.no_grad():
            self.occupancy_head[1].weight.copy_(self.tile_head[1].weight)
            self.occupancy_head[1].bias.copy_(self.tile_head[1].bias)
            self.occupancy_head[3].weight[0].copy_(self.tile_head[3].weight[TILE_PLAYER])
            self.occupancy_head[3].bias[0].copy_(self.tile_head[3].bias[TILE_PLAYER])
            self.occupancy_head[3].weight[1].copy_(self.tile_head[3].weight[TILE_MONSTER])
            self.occupancy_head[3].bias[1].copy_(self.tile_head[3].bias[TILE_MONSTER])

    def forward(self, images: Any) -> dict[str, Any]:
        # This is the V14 forward path, kept intact so the occupancy branch does
        # not perturb heatmap/offset/static outputs before training.
        high_resolution_features = self.encoder[:6](images)
        mid_resolution_features = self.encoder[6:10](high_resolution_features)
        features = self.encoder[10:](mid_resolution_features)
        tile_logits = self.tile_head(features)
        exit_type_logits = self.exit_type_head(features)
        chest_state_logits = self.chest_state_head(features)
        occupancy_logits = self.occupancy_head(features)
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
                (high_resolution_features, mid_resolution_features, heatmap_logits),
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
            "occupancy_logits": occupancy_logits,
        }


class PerceptionDataset(_v14.PerceptionDataset):
    """V14 dataset view with logical occupancy labels derived from true centers."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        occupancy = self.torch.zeros(
            (NUM_OCCUPANCY_CHANNELS, GRID_HEIGHT, GRID_WIDTH),
            dtype=self.torch.float32,
        )

        player_center = sample["player_center"]
        _mark_center_tile(occupancy[0], player_center)
        for center, present in zip(sample["monster_centers"], sample["monster_masks"]):
            if bool(present):
                _mark_center_tile(occupancy[1], center)

        sample["occupancy"] = occupancy
        return sample


def _mark_center_tile(channel: Any, center: Any) -> None:
    x = int(float(center[0]) // TILE_SIZE)
    y = int(float(center[1]) // TILE_SIZE)
    if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
        channel[y, x] = 1.0


def collect_dataset(config: DatasetConfig) -> Path:
    """Collect the unchanged V14 NPZ schema; occupancy labels are derived lazily."""

    return _v14.collect_dataset(
        _v14.DatasetConfig(
            output=Path(config.output),
            samples=config.samples,
            builtin_ratio=config.builtin_ratio,
            exit_overlap_ratio=config.exit_overlap_ratio,
            max_monsters=config.max_monsters,
            seed=config.seed,
        )
    )


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
    """Train V16 with automatic occupancy-head-only warm start.

    If ``weights_path`` already contains V14/V15 weights, compatible parameters
    are loaded, frozen, and only the new occupancy head is trained.  A subsequent
    call on a V16 checkpoint resumes normal joint training.  This preserves the
    public V14 signature and avoids degrading a strong localization checkpoint.
    """

    torch, nn, _ = _torch_modules()
    DataLoader = torch.utils.data.DataLoader
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    resolved_device = _v14._resolve_device(torch, device)
    dataset_path = Path(dataset_path)
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path)
    sample_count = int(data["images"].shape[0])
    rng = np.random.default_rng(seed)
    train_indices, val_indices = _v14._grouped_train_val_split(sample_count, rng)
    if len(val_indices) > MAX_TRAINING_VALIDATION_SAMPLES:
        val_indices = val_indices[:MAX_TRAINING_VALIDATION_SAMPLES]

    train_ds = PerceptionDataset(
        dataset_path,
        indices=train_indices,
        augment=True,
        variants=IMAGE_VARIANTS,
        randomize_variant=True,
    )
    val_ds = PerceptionDataset(
        dataset_path,
        indices=val_indices,
        augment=False,
        variants=IMAGE_VARIANTS,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyPerceptionCNN().to(resolved_device)
    warm_started = _warm_start_model(model, weights_path, torch, resolved_device)
    occupancy_head_only = (
        warm_started
        and not getattr(model, "occupancy_head_warm_started", False)
        and not chest_head_only
    )

    if chest_head_only:
        if not warm_started:
            raise ValueError("--chest-head-only requires an existing compatible checkpoint")
        _freeze_all(model)
        _unfreeze(model.chest_state_head)
        trainable_parameters: Any = model.chest_state_head.parameters()
        print("training_chest_head_only=true", flush=True)
    elif occupancy_head_only:
        _freeze_all(model)
        _unfreeze(model.occupancy_head)
        trainable_parameters = model.occupancy_head.parameters()
        print("training_occupancy_head_only=true", flush=True)
    else:
        occupancy_parameters = list(model.occupancy_head.parameters())
        localization_parameters = (
            list(model.heatmap_head.parameters())
            + list(model.offset_head.parameters())
            + list(model.player_refine_head.parameters())
            + list(model.monster_refine_head.parameters())
            + occupancy_parameters
        )
        localization_ids = {id(parameter) for parameter in localization_parameters}
        shared_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in localization_ids
        ]
        trainable_parameters = [
            {"params": shared_parameters, "lr": lr * 0.25},
            {"params": localization_parameters, "lr": lr},
        ]

    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=1e-4)
    tile_weights = _v14._tile_class_weights(data["grids"]).to(resolved_device)
    exit_type_labels = (
        data["exit_type_grids"]
        if "exit_type_grids" in data.files
        else _v14._fallback_exit_type_grids(data["grids"])
    )
    exit_type_weights = _v14._exit_type_class_weights(exit_type_labels).to(resolved_device)
    chest_state_labels = (
        data["chest_state_grids"]
        if "chest_state_grids" in data.files
        else _v14._fallback_chest_state_grids(data["grids"])
    )
    chest_state_weights = _v14._chest_state_class_weights(chest_state_labels).to(resolved_device)
    occupancy_pos_weight = _occupancy_pos_weight(data).to(resolved_device)

    tile_loss_fn = nn.CrossEntropyLoss(weight=tile_weights)
    exit_type_loss_fn = nn.CrossEntropyLoss(weight=exit_type_weights)
    chest_state_loss_fn = nn.CrossEntropyLoss(weight=chest_state_weights)
    heatmap_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(12.0, device=resolved_device)
    )
    occupancy_loss_fn = nn.BCEWithLogitsLoss(pos_weight=occupancy_pos_weight)

    best_score = float("inf")
    best_metrics: dict[str, float] = {}
    patience = 6
    stale_epochs = 0
    if not chest_head_only:
        model.has_trained_occupancy_head = True

    for epoch in range(1, epochs + 1):
        if chest_head_only:
            model.eval()
            model.chest_state_head.train()
        elif occupancy_head_only:
            model.eval()
            model.occupancy_head.train()
        else:
            model.train()

        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            images = batch["image"].to(resolved_device)
            if chest_head_only:
                chest_state_grids = batch["chest_state_grid"].to(resolved_device)
                features = model.encoder(images)
                loss = chest_state_loss_fn(model.chest_state_head(features), chest_state_grids)
            elif occupancy_head_only:
                occupancy = batch["occupancy"].to(resolved_device)
                with torch.no_grad():
                    features = model.encoder(images)
                occupancy_logits = model.occupancy_head(features)
                occupancy_bce = occupancy_loss_fn(occupancy_logits, occupancy)
                occupancy_cardinality = _occupancy_cardinality_loss(
                    occupancy_logits, occupancy, torch=torch
                )
                loss = (
                    occupancy_bce
                    + OCCUPANCY_CARDINALITY_LOSS_WEIGHT * occupancy_cardinality
                )
            else:
                grids = batch["grid"].to(resolved_device)
                exit_type_grids = batch["exit_type_grid"].to(resolved_device)
                chest_state_grids = batch["chest_state_grid"].to(resolved_device)
                heatmaps = batch["heatmap"].to(resolved_device)
                player_centers = batch["player_center"].to(resolved_device)
                monster_centers = batch["monster_centers"].to(resolved_device)
                monster_masks = batch["monster_masks"].to(resolved_device)
                occupancy = batch["occupancy"].to(resolved_device)
                output = model(images)

                player_coordinate_loss, _ = _v14._player_coordinate_loss(
                    output["heatmap_logits"], player_centers, torch=torch
                )
                player_tile_loss = _v14._player_tile_classification_loss(
                    output["heatmap_logits"], player_centers, torch=torch
                )
                monster_coordinate_loss = _v14._monster_coordinate_loss(
                    output["heatmap_logits"],
                    monster_centers,
                    monster_masks,
                    torch=torch,
                )
                monster_tile_loss = _v14._monster_tile_classification_loss(
                    output["heatmap_logits"],
                    monster_centers,
                    monster_masks,
                    torch=torch,
                )
                entity_offset_loss = _v14._entity_offset_loss(
                    output["offset_logits"],
                    player_centers,
                    monster_centers,
                    monster_masks,
                    torch=torch,
                )
                occupancy_bce = occupancy_loss_fn(output["occupancy_logits"], occupancy)
                occupancy_cardinality = _occupancy_cardinality_loss(
                    output["occupancy_logits"], occupancy, torch=torch
                )
                loss = (
                    tile_loss_fn(output["tile_logits"], grids)
                    + 0.30 * exit_type_loss_fn(output["exit_type_logits"], exit_type_grids)
                    + 0.55 * chest_state_loss_fn(
                        output["chest_state_logits"], chest_state_grids
                    )
                    + 0.35 * heatmap_loss_fn(output["heatmap_logits"], heatmaps)
                    + _v14.PLAYER_COORDINATE_LOSS_WEIGHT * player_coordinate_loss
                    + _v14.PLAYER_TILE_LOSS_WEIGHT * player_tile_loss
                    + _v14.MONSTER_COORDINATE_LOSS_WEIGHT * monster_coordinate_loss
                    + _v14.MONSTER_TILE_LOSS_WEIGHT * monster_tile_loss
                    + _v14.ENTITY_OFFSET_LOSS_WEIGHT * entity_offset_loss
                    + OCCUPANCY_LOSS_WEIGHT * occupancy_bce
                    + OCCUPANCY_CARDINALITY_LOSS_WEIGHT * occupancy_cardinality
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1

        if chest_head_only:
            metrics = _v14.evaluate_chest_head(model, val_loader, device=resolved_device)
        else:
            metrics = evaluate_model(
                model,
                val_loader,
                device=resolved_device,
                occupancy_pos_weight=occupancy_pos_weight,
            )
        metrics["train_loss"] = train_loss / max(1, train_batches)

        if chest_head_only:
            selection_score = metrics["val_loss"]
            print(
                f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"chest_acc={metrics['chest_state_acc']:.4f}",
                flush=True,
            )
        elif occupancy_head_only:
            selection_score = (
                1.0 - metrics["occupancy_player_tile_acc"]
                + 1.0 - metrics["occupancy_monster_f1"]
                + 0.25 * (1.0 - metrics["occupancy_monster_exact"])
            )
            print(
                f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
                f"occ_loss={metrics['occupancy_val_loss']:.4f} "
                f"player_tile={metrics['occupancy_player_tile_acc']:.4f} "
                f"monster_f1={metrics['occupancy_monster_f1']:.4f} "
                f"monster_exact={metrics['occupancy_monster_exact']:.4f}",
                flush=True,
            )
        else:
            selection_score = metrics["val_loss"]
            print(
                f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
                f"val_loss={metrics['val_loss']:.4f} tile_acc={metrics['tile_acc']:.4f} "
                f"player_occ={metrics['occupancy_player_tile_acc']:.4f} "
                f"monster_occ_f1={metrics['occupancy_monster_f1']:.4f} "
                f"player_err={metrics['player_center_error_px']:.2f}px "
                f"monster_err={metrics['monster_center_error_px']:.2f}px",
                flush=True,
            )

        metrics["selection_score"] = float(selection_score)
        if selection_score < best_score:
            best_score = float(selection_score)
            best_metrics = dict(metrics)
            stale_epochs = 0
            _save_checkpoint(model, weights_path, dataset_path, best_metrics, torch)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return best_metrics


def _freeze_all(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _unfreeze(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def _occupancy_pos_weight(data: Any) -> Any:
    torch, _, _ = _torch_modules()
    samples = int(data["images"].shape[0])
    cells = max(1, samples * GRID_HEIGHT * GRID_WIDTH)
    player_positives = max(1, samples)
    monster_positives = max(1, int(np.asarray(data["monster_masks"]).sum()))
    positives = np.asarray((player_positives, monster_positives), dtype=np.float32)
    # sqrt balancing is intentionally conservative: full inverse frequency tends
    # to improve recall at the cost of many false-positive monster tiles.
    weights = np.sqrt(np.maximum(1.0, (cells - positives) / positives))
    weights = np.clip(weights, 1.0, 32.0).astype(np.float32)
    return torch.from_numpy(weights).reshape(NUM_OCCUPANCY_CHANNELS, 1, 1)


def _occupancy_cardinality_loss(logits: Any, targets: Any, *, torch: Any) -> Any:
    predicted_counts = torch.sigmoid(logits).sum(dim=(2, 3))
    true_counts = targets.sum(dim=(2, 3))
    return torch.nn.functional.smooth_l1_loss(
        predicted_counts,
        true_counts,
        beta=0.5,
    ) / math.sqrt(float(GRID_HEIGHT * GRID_WIDTH))


def evaluate_model(
    model: Any,
    loader: Any,
    *,
    device: Any,
    occupancy_pos_weight: Any | None = None,
) -> dict[str, float]:
    metrics = _v14.evaluate_model(model, loader, device=device)
    occupancy_metrics = _evaluate_occupancy(
        model,
        loader,
        device=device,
        occupancy_pos_weight=occupancy_pos_weight,
    )
    metrics.update(occupancy_metrics)
    metrics["val_loss"] += (
        OCCUPANCY_LOSS_WEIGHT * occupancy_metrics["occupancy_val_loss"]
        + OCCUPANCY_CARDINALITY_LOSS_WEIGHT
        * occupancy_metrics["occupancy_cardinality_loss"]
    )
    return metrics


def _evaluate_occupancy(
    model: Any,
    loader: Any,
    *,
    device: Any,
    occupancy_pos_weight: Any | None,
) -> dict[str, float]:
    torch, nn, _ = _torch_modules()
    if occupancy_pos_weight is None:
        occupancy_pos_weight = torch.ones(
            (NUM_OCCUPANCY_CHANNELS, 1, 1), device=device
        )
    else:
        occupancy_pos_weight = occupancy_pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=occupancy_pos_weight)

    model.eval()
    loss_total = 0.0
    cardinality_total = 0.0
    batches = 0
    samples = 0
    player_hits = 0
    monster_tp = 0
    monster_fp = 0
    monster_fn = 0
    monster_exact = 0
    joint_exact = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["occupancy"].to(device)
            logits = model(images)["occupancy_logits"]
            loss_total += float(loss_fn(logits, targets).cpu())
            cardinality_total += float(
                _occupancy_cardinality_loss(logits, targets, torch=torch).cpu()
            )
            batches += 1

            probabilities = torch.sigmoid(logits)
            player_pred = probabilities[:, 0].flatten(1).argmax(dim=1)
            player_true = targets[:, 0].flatten(1).argmax(dim=1)
            player_correct = player_pred == player_true
            player_hits += int(player_correct.sum().cpu())

            monster_pred = probabilities[:, 1] >= OCCUPANCY_MONSTER_THRESHOLD
            monster_true = targets[:, 1] >= 0.5
            monster_tp += int((monster_pred & monster_true).sum().cpu())
            monster_fp += int((monster_pred & ~monster_true).sum().cpu())
            monster_fn += int((~monster_pred & monster_true).sum().cpu())
            monster_frame_exact = (monster_pred == monster_true).flatten(1).all(dim=1)
            monster_exact += int(monster_frame_exact.sum().cpu())
            joint_exact += int((monster_frame_exact & player_correct).sum().cpu())
            samples += int(targets.shape[0])

    precision = monster_tp / max(1, monster_tp + monster_fp)
    recall = monster_tp / max(1, monster_tp + monster_fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "occupancy_val_loss": loss_total / max(1, batches),
        "occupancy_cardinality_loss": cardinality_total / max(1, batches),
        "occupancy_player_tile_acc": player_hits / max(1, samples),
        "occupancy_monster_precision": precision,
        "occupancy_monster_recall": recall,
        "occupancy_monster_f1": f1,
        "occupancy_monster_exact": monster_exact / max(1, samples),
        "occupancy_joint_exact": joint_exact / max(1, samples),
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
    pos_weight = _occupancy_pos_weight(data).to(device)
    results: dict[str, dict[str, float]] = {}
    for variant in variants:
        dataset = PerceptionDataset(
            dataset_path,
            indices=indices,
            augment=False,
            variants=(variant,),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        results[variant] = evaluate_model(
            model,
            loader,
            device=device,
            occupancy_pos_weight=pos_weight,
        )
    return results


def _save_checkpoint(
    model: Any,
    weights_path: Path,
    dataset_path: Path,
    metrics: dict[str, float],
    torch: Any,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_tile_classes": NUM_TILE_CLASSES,
            "num_exit_type_classes": NUM_EXIT_TYPE_CLASSES,
            "num_chest_state_classes": NUM_CHEST_STATE_CLASSES,
            "num_occupancy_channels": NUM_OCCUPANCY_CHANNELS,
            "occupancy_channels": OCCUPANCY_CHANNELS,
            "exit_type_names": EXIT_TYPE_NAMES,
            "chest_state_names": CHEST_STATE_NAMES,
            "metrics": metrics,
            "dataset": str(dataset_path),
            "has_trained_offset_head": bool(
                getattr(model, "has_trained_offset_head", False)
            ),
            "has_trained_player_offset_head": bool(
                getattr(model, "has_trained_player_offset_head", False)
            ),
            "has_trained_refine_head": bool(
                getattr(model, "has_trained_refine_head", False)
            ),
            "has_trained_occupancy_head": bool(
                getattr(model, "has_trained_occupancy_head", False)
            ),
        },
        weights_path,
    )


def _load_checkpoint(weights_path: str | Path, torch: Any, device: Any) -> dict[str, Any]:
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weights_path, map_location=device)


def load_model(
    weights_path: str | Path = DEFAULT_WEIGHTS,
    *,
    device: str | None = None,
) -> tuple[Any, Any]:
    torch, _, _ = _torch_modules()
    resolved_device = _v14._resolve_device(torch, device)
    checkpoint = _load_checkpoint(weights_path, torch, resolved_device)
    model = TinyPerceptionCNN(
        num_classes=int(checkpoint.get("num_tile_classes", NUM_TILE_CLASSES)),
        num_exit_type_classes=int(
            checkpoint.get("num_exit_type_classes", NUM_EXIT_TYPE_CLASSES)
        ),
        num_chest_state_classes=int(
            checkpoint.get("num_chest_state_classes", NUM_CHEST_STATE_CLASSES)
        ),
    )
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)

    allowed_missing_prefixes = (
        "exit_type_head.",
        "chest_state_head.",
        "offset_head.",
        "player_refine_head.",
        "monster_refine_head.",
        "occupancy_head.",
    )
    missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"perception checkpoint incompatible: missing={missing}, unexpected={unexpected}"
        )

    _restore_head_flags(model, checkpoint, incompatible.missing_keys)
    occupancy_available = not any(
        name.startswith("occupancy_head.") for name in incompatible.missing_keys
    )
    model.has_trained_occupancy_head = bool(
        checkpoint.get("has_trained_occupancy_head", occupancy_available)
    ) and occupancy_available
    model.occupancy_head_warm_started = model.has_trained_occupancy_head
    if not occupancy_available:
        model.initialize_occupancy_from_tile_head()

    model.to(resolved_device)
    model.eval()
    return model, torch


def _restore_head_flags(
    model: Any,
    checkpoint: dict[str, Any],
    missing_keys: Sequence[str],
) -> None:
    model.exit_type_head_available = not any(
        name.startswith("exit_type_head.") for name in missing_keys
    )
    model.has_trained_chest_state_head = not any(
        name.startswith("chest_state_head.") for name in missing_keys
    )
    offset_available = not any(name.startswith("offset_head.") for name in missing_keys)
    refine_available = not any(
        name.startswith(("player_refine_head.", "monster_refine_head."))
        for name in missing_keys
    )
    model.has_trained_offset_head = bool(
        checkpoint.get("has_trained_offset_head", offset_available)
    ) and offset_available
    model.has_trained_player_offset_head = bool(
        checkpoint.get(
            "has_trained_player_offset_head",
            model.has_trained_offset_head,
        )
    ) and offset_available
    model.has_trained_refine_head = bool(
        checkpoint.get("has_trained_refine_head", refine_available)
    ) and refine_available


def _warm_start_model(model: Any, weights_path: Path, torch: Any, device: Any) -> bool:
    if not weights_path.exists():
        return False
    checkpoint = _load_checkpoint(weights_path, torch, device)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        return False

    current_state = model.state_dict()
    compatible_state = {
        name: tensor
        for name, tensor in state_dict.items()
        if name in current_state and tuple(tensor.shape) == tuple(current_state[name].shape)
    }
    required_prefixes = [
        "encoder.",
        "tile_head.",
        "exit_type_head.",
        "chest_state_head.",
        "heatmap_head.",
    ]
    if bool(checkpoint.get("has_trained_refine_head", False)):
        required_prefixes.extend(("player_refine_head.", "monster_refine_head."))
    if bool(checkpoint.get("has_trained_offset_head", False)):
        required_prefixes.append("offset_head.")
    required_keys = {
        name
        for name in current_state
        if name.startswith(tuple(required_prefixes))
    }
    if not compatible_state or not required_keys.issubset(compatible_state):
        return False
    model.load_state_dict(compatible_state, strict=False)

    offset_keys = [name for name in current_state if name.startswith("offset_head.")]
    refine_keys = [
        name
        for name in current_state
        if name.startswith(("player_refine_head.", "monster_refine_head."))
    ]
    occupancy_keys = [name for name in current_state if name.startswith("occupancy_head.")]
    model.offset_head_warm_started = bool(
        checkpoint.get("has_trained_offset_head", True)
    ) and all(name in compatible_state for name in offset_keys)
    model.refine_head_warm_started = bool(
        checkpoint.get("has_trained_refine_head", True)
    ) and all(name in compatible_state for name in refine_keys)
    model.occupancy_head_warm_started = bool(
        checkpoint.get("has_trained_occupancy_head", False)
    ) and all(name in compatible_state for name in occupancy_keys)

    model.has_trained_offset_head = model.offset_head_warm_started
    model.has_trained_refine_head = model.refine_head_warm_started
    model.has_trained_occupancy_head = model.occupancy_head_warm_started
    if not model.occupancy_head_warm_started:
        model.initialize_occupancy_from_tile_head()
    print(
        f"warm_start_params={len(compatible_state)} "
        f"occupancy_warm_started={str(model.occupancy_head_warm_started).lower()}",
        flush=True,
    )
    return True


def predict_frame(
    model: Any,
    frame: np.ndarray,
    *,
    torch_module: Any | None = None,
    device: str | None = None,
    confidence_threshold: float = 0.35,
) -> dict[str, Any]:
    """Predict with occupancy-authoritative tiles and heatmap-derived centers.

    For a trained V16 checkpoint, ``player_tile``, ``monster_tiles``, and each
    detection's ``tile`` field are the logical locations.  ``center_px`` remains
    the heatmap/refinement result.  A consuming engine must honor the explicit
    ``tile`` field instead of recomputing it from ``center_px``.
    """

    # Old V14/V15 checkpoints load safely but must not use an untrained new head.
    if not getattr(model, "has_trained_occupancy_head", False):
        return _v14.predict_frame(
            model,
            frame,
            torch_module=torch_module,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    torch = torch_module if torch_module is not None else _torch_modules()[0]
    resolved_device = _v14._resolve_device(torch, device)
    image = _v14.apply_image_variant(frame, "default")
    tensor = (
        torch.from_numpy(np.transpose(image, (2, 0, 1)))
        .unsqueeze(0)
        .float()
        .to(resolved_device)
    )
    with torch.no_grad():
        output = model(tensor)
        raw_grid = output["tile_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        if model.exit_type_head_available and "exit_type_logits" in output:
            exit_type_probabilities = (
                torch.softmax(output["exit_type_logits"], dim=1)[0].cpu().numpy()
            )
            exit_type_grid = exit_type_probabilities.argmax(axis=0).astype(np.uint8)
        else:
            exit_type_probabilities = None
            exit_type_grid = _v14._fallback_exit_type_grids(raw_grid[None, ...])[0]
        if (
            getattr(model, "has_trained_chest_state_head", True)
            and "chest_state_logits" in output
        ):
            chest_state_probabilities = (
                torch.softmax(output["chest_state_logits"], dim=1)[0].cpu().numpy()
            )
            chest_state_grid = chest_state_probabilities.argmax(axis=0).astype(np.uint8)
        else:
            chest_state_probabilities = None
            chest_state_grid = _v14._fallback_chest_state_grids(raw_grid[None, ...])[0]
        heatmaps = torch.sigmoid(output["heatmap_logits"])[0].cpu().numpy()
        occupancy_probabilities = (
            torch.sigmoid(output["occupancy_logits"])[0].cpu().numpy()
        )
        player_offset_available = bool(
            getattr(model, "has_trained_player_offset_head", False)
            or getattr(model, "has_trained_offset_head", False)
        )
        offset_map = (
            torch.tanh(output["offset_logits"])[0].cpu().numpy() * ENTITY_OFFSET_SCALE
            if player_offset_available
            else None
        )

    player_tile = _player_tile_from_occupancy(occupancy_probabilities[0])
    monster_tiles = _monster_tiles_from_occupancy(
        occupancy_probabilities[1], threshold=confidence_threshold
    )

    grid = raw_grid.copy()
    grid[(grid == TILE_PLAYER) | (grid == TILE_MONSTER)] = TILE_EMPTY
    for x, y in monster_tiles:
        grid[y, x] = TILE_MONSTER
    grid[player_tile[1], player_tile[0]] = TILE_PLAYER

    player_confidence = float(heatmaps[0].max())
    if player_confidence >= confidence_threshold:
        peak_y, peak_x = np.unravel_index(int(np.argmax(heatmaps[0])), heatmaps[0].shape)
        center_x, center_y = _v14._refine_peak_center(
            heatmaps[0], int(peak_x), int(peak_y)
        )
        if offset_map is not None:
            center_x, center_y = _v14._center_from_offset_prediction(
                (center_x, center_y), offset_map, channel_offset=0
            )
    else:
        center_x, center_y = _tile_center(player_tile)
    player_tile = _fuse_player_tile(
        player_tile,
        occupancy_probabilities[0],
        (center_x, center_y),
    )
    player = {
        "center_px": (center_x, center_y),
        "confidence": player_confidence,
        "tile": player_tile,
        "tile_confidence": float(occupancy_probabilities[0, player_tile[1], player_tile[0]]),
    }

    heat_monsters: list[dict[str, Any]] = []
    for peak_x, peak_y, confidence in _v14._peaks_from_heatmap(
        heatmaps[1], threshold=confidence_threshold, max_peaks=OCCUPANCY_MAX_MONSTERS
    ):
        center_x, center_y = _v14._refine_peak_center(
            heatmaps[1],
            peak_x,
            peak_y,
            radius=_v14.MONSTER_CENTER_WINDOW_RADIUS,
            power=_v14.MONSTER_CENTER_POWER,
        )
        if offset_map is not None and getattr(model, "has_trained_offset_head", False):
            center_x, center_y = _v14._center_from_offset_prediction(
                (center_x, center_y), offset_map, channel_offset=2
            )
        heat_monsters.append(
            {
                "center_px": (center_x, center_y),
                "confidence": float(confidence),
            }
        )
    monster_tiles = _fuse_monster_tiles(
        heat_monsters,
        monster_tiles,
        occupancy_probabilities[1],
    )
    grid[(grid == TILE_MONSTER) | (grid == TILE_PLAYER)] = TILE_EMPTY
    for x, y in monster_tiles:
        grid[y, x] = TILE_MONSTER
    grid[player_tile[1], player_tile[0]] = TILE_PLAYER
    monsters = _pair_heat_centers_with_occupancy(
        heat_monsters, monster_tiles, occupancy_probabilities[1]
    )

    return {
        "grid": grid,
        "exit_type_grid": exit_type_grid,
        "exit_type_probabilities": exit_type_probabilities,
        "exit_types": _v14._exit_types_from_prediction(
            grid,
            exit_type_grid,
            exit_type_probabilities=exit_type_probabilities,
            player_tile=player_tile,
        ),
        "chest_state_grid": chest_state_grid,
        "chest_state_probabilities": chest_state_probabilities,
        "closed_chests": _v14._tiles_from_class_grid(
            chest_state_grid, CHEST_STATE_TO_INDEX["closed"]
        ),
        "opened_chests": _v14._tiles_from_class_grid(
            chest_state_grid, CHEST_STATE_TO_INDEX["opened"]
        ),
        "player": player,
        "monsters": tuple(monsters),
        "heatmaps": heatmaps,
        "occupancy_probabilities": occupancy_probabilities,
        "player_tile": player_tile,
        "monster_tiles": frozenset(monster_tiles),
    }


def _player_tile_from_occupancy(probabilities: np.ndarray) -> tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(probabilities)), probabilities.shape)
    return int(x), int(y)


def _fuse_player_tile(
    occupancy_tile: tuple[int, int],
    probabilities: np.ndarray,
    center_px: tuple[float, float],
) -> tuple[int, int]:
    """Resolve a one-pixel grid-line ambiguity using both localization heads."""

    order = np.argsort(probabilities, axis=None)[::-1][:2]
    if len(order) < 2:
        return occupancy_tile
    candidates = [
        (int(index % GRID_WIDTH), int(index // GRID_WIDTH))
        for index in order
    ]
    first, second = candidates
    if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
        return occupancy_tile
    axis = 0 if first[0] != second[0] else 1
    coordinate = float(center_px[axis])
    boundary = round(coordinate / TILE_SIZE) * TILE_SIZE
    signed_distance = coordinate - boundary
    lower = min(first[axis], second[axis])
    higher = max(first[axis], second[axis])
    if (
        abs(signed_distance) <= PLAYER_BOUNDARY_CORRECTION_WINDOW_PX
        and first[axis] == lower
        and second[axis] == higher
    ):
        return second
    return occupancy_tile


def _monster_tiles_from_occupancy(
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[int, int]]:
    effective_threshold = max(OCCUPANCY_MONSTER_THRESHOLD, float(threshold))
    rows, cols = np.where(probabilities >= effective_threshold)
    candidates = sorted(
        (
            (float(probabilities[row, col]), int(col), int(row))
            for row, col in zip(rows, cols)
        ),
        reverse=True,
    )[:OCCUPANCY_MAX_MONSTERS]
    return [(x, y) for _probability, x, y in candidates]


def _pair_heat_centers_with_occupancy(
    heat_monsters: Sequence[dict[str, Any]],
    monster_tiles: Sequence[tuple[int, int]],
    occupancy_probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    assignments: dict[int, int] = {}
    candidates = sorted(
        (
            (
                math.hypot(
                    float(heat["center_px"][0]) - _tile_center(tile)[0],
                    float(heat["center_px"][1]) - _tile_center(tile)[1],
                ),
                tile_index,
                heat_index,
            )
            for tile_index, tile in enumerate(monster_tiles)
            for heat_index, heat in enumerate(heat_monsters)
        )
    )
    used_heat: set[int] = set()
    for distance, tile_index, heat_index in candidates:
        if distance > MAX_HEAT_OCCUPANCY_MATCH_DISTANCE_PX:
            break
        if tile_index in assignments or heat_index in used_heat:
            continue
        assignments[tile_index] = heat_index
        used_heat.add(heat_index)

    monsters: list[dict[str, Any]] = []
    for tile_index, tile in enumerate(monster_tiles):
        heat_index = assignments.get(tile_index)
        if heat_index is None:
            center_px = _tile_center(tile)
            confidence = 0.0
            center_source = "occupancy_fallback"
        else:
            center_px = tuple(float(value) for value in heat_monsters[heat_index]["center_px"])
            confidence = float(heat_monsters[heat_index]["confidence"])
            center_source = "heatmap"
        monsters.append(
            {
                "center_px": center_px,
                "confidence": confidence,
                "entity_type": "unknown",
                "tile": tile,
                "tile_confidence": float(occupancy_probabilities[tile[1], tile[0]]),
                "center_source": center_source,
            }
        )
    return monsters


def _fuse_monster_tiles(
    heat_monsters: Sequence[dict[str, Any]],
    occupancy_tiles: Sequence[tuple[int, int]],
    occupancy_probabilities: np.ndarray,
) -> list[tuple[int, int]]:
    """Use precise heat centers while retaining independent strong detections.

    A moving sprite can straddle two cells, causing a sigmoid occupancy head to
    activate both neighbors.  The heatmap head predicts one continuous center
    per visible entity, so each of its centers contributes exactly one tile.
    Occupancy detections remain useful as fallbacks when the heat head misses an
    isolated entity, but adjacent cells already represented by a heat center are
    duplicate hypotheses rather than additional monsters.
    """

    heat_tiles = {
        (
            int(np.clip(float(item["center_px"][0]) // TILE_SIZE, 0, GRID_WIDTH - 1)),
            int(np.clip(float(item["center_px"][1]) // TILE_SIZE, 0, GRID_HEIGHT - 1)),
        )
        for item in heat_monsters
        if float(item.get("confidence", 0.0)) >= HEAT_ONLY_MONSTER_CONFIDENCE
    }
    fused = set(heat_tiles)
    for tile in occupancy_tiles:
        if any(abs(tile[0] - other[0]) + abs(tile[1] - other[1]) <= 1 for other in heat_tiles):
            continue
        if (
            float(occupancy_probabilities[tile[1], tile[0]])
            >= INDEPENDENT_OCCUPANCY_CONFIDENCE
        ):
            fused.add(tile)
    return sorted(fused)


def _tile_center(tile: tuple[int, int]) -> tuple[float, float]:
    return (
        float(tile[0] * TILE_SIZE + TILE_SIZE * 0.5),
        float(tile[1] * TILE_SIZE + TILE_SIZE * 0.5),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train experimental NesyLink V16 occupancy perception CNN."
    )
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
    eval_parser.add_argument(
        "--variant", choices=(*IMAGE_VARIANTS, "all"), default="default"
    )

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
        resolved_device = _v14._resolve_device(torch, args.device)
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
            dataset = PerceptionDataset(
                args.data,
                indices=indices,
                augment=False,
                variants=(args.variant,),
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            metrics = evaluate_model(
                model,
                loader,
                device=resolved_device,
                occupancy_pos_weight=_occupancy_pos_weight(data).to(resolved_device),
            )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
