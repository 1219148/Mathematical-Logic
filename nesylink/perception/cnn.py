from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nesylink.core.constants import GRID_HEIGHT, GRID_WIDTH, MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH, TILE_SIZE
from nesylink.core.observation import TILE_MONSTER, TILE_PLAYER


NUM_TILE_CLASSES = 12
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATASET = DEFAULT_DATA_DIR / "perception_dataset.npz"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "perception_model.pt"
BUILTIN_TASKS = tuple(f"mathematical_logic/task_{idx}" for idx in range(1, 6))


@dataclass(frozen=True)
class DatasetConfig:
    output: Path = DEFAULT_DATASET
    samples: int = 2400
    builtin_ratio: float = 0.45
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

    def __init__(self, num_classes: int = NUM_TILE_CLASSES) -> None:
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
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )
        self._interpolate = torch.nn.functional.interpolate

    def forward(self, images: Any) -> dict[str, Any]:
        features = self.encoder(images)
        tile_logits = self.tile_head(features)
        heatmap_logits = self.heatmap_head(features)
        heatmap_logits = self._interpolate(
            heatmap_logits,
            size=(MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        return {"tile_logits": tile_logits, "heatmap_logits": heatmap_logits}


class PerceptionDataset(_BaseTorchDataset()):
    def __init__(self, dataset_path: str | Path, *, indices: np.ndarray, augment: bool = False) -> None:
        torch, _, Dataset = _torch_modules()
        super().__init__()
        del Dataset
        data = np.load(dataset_path)
        self.images = data["images"][indices]
        self.grids = data["grids"][indices]
        self.player_centers = data["player_centers"][indices]
        self.monster_centers = data["monster_centers"][indices]
        self.monster_masks = data["monster_masks"][indices]
        self.augment = bool(augment)
        self.torch = torch

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self.images[index].astype(np.float32) / 255.0
        if self.augment:
            image = _augment_image(image)
        image_tensor = self.torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        grid_tensor = self.torch.from_numpy(self.grids[index].astype(np.int64))
        heatmap = _make_heatmaps(
            self.player_centers[index],
            self.monster_centers[index],
            self.monster_masks[index],
        )
        return {
            "image": image_tensor,
            "grid": grid_tensor,
            "heatmap": self.torch.from_numpy(heatmap).float(),
        }


def collect_dataset(config: DatasetConfig) -> Path:
    """采集 perception 数据。

    数据来源包含公开任务 rollout 和随机单房间地图。标签来自 structured obs，
    只用于训练阶段；最终 PerceptionEngine.extract 不会读取 info。
    """
    rng = random.Random(config.seed)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    images: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    player_centers: list[np.ndarray] = []
    monster_centers: list[np.ndarray] = []
    monster_masks: list[np.ndarray] = []

    builtin_samples = int(config.samples * config.builtin_ratio)
    random_samples = max(0, config.samples - builtin_samples)
    _collect_builtin_samples(
        target_count=builtin_samples,
        rng=rng,
        images=images,
        grids=grids,
        player_centers=player_centers,
        monster_centers=monster_centers,
        monster_masks=monster_masks,
        max_monsters=config.max_monsters,
    )
    _collect_random_room_samples(
        target_count=random_samples,
        rng=rng,
        images=images,
        grids=grids,
        player_centers=player_centers,
        monster_centers=monster_centers,
        monster_masks=monster_masks,
        max_monsters=config.max_monsters,
        data_dir=output.parent,
    )

    np.savez_compressed(
        output,
        images=np.stack(images).astype(np.uint8),
        grids=np.stack(grids).astype(np.uint8),
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
    indices = np.arange(sample_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = max(1, int(sample_count * 0.85))
    train_indices = indices[:split]
    val_indices = indices[split:] if split < sample_count else indices[: max(1, sample_count // 10)]

    train_ds = PerceptionDataset(dataset_path, indices=train_indices, augment=True)
    val_ds = PerceptionDataset(dataset_path, indices=val_indices, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinyPerceptionCNN().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tile_weights = _tile_class_weights(data["grids"]).to(resolved_device)
    tile_loss_fn = nn.CrossEntropyLoss(weight=tile_weights)
    heatmap_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(12.0, device=resolved_device))

    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    patience = 4
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for batch in train_loader:
            images = batch["image"].to(resolved_device)
            grids = batch["grid"].to(resolved_device)
            heatmaps = batch["heatmap"].to(resolved_device)
            output = model(images)
            tile_loss = tile_loss_fn(output["tile_logits"], grids)
            heatmap_loss = heatmap_loss_fn(output["heatmap_logits"], heatmaps)
            loss = tile_loss + 0.35 * heatmap_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
            train_batches += 1

        metrics = evaluate_model(model, val_loader, device=resolved_device)
        metrics["train_loss"] = train_loss / max(1, train_batches)
        print(
            f"epoch={epoch:02d} train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} tile_acc={metrics['tile_acc']:.4f} "
            f"player_err={metrics['player_center_error_px']:.2f}px "
            f"monster_recall={metrics['monster_tile_recall']:.4f}",
            flush=True,
        )
        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            best_metrics = dict(metrics)
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_tile_classes": NUM_TILE_CLASSES,
                    "metrics": best_metrics,
                    "dataset": str(dataset_path),
                },
                weights_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience and metrics["tile_acc"] > 0.97:
                break
    return best_metrics


def evaluate_model(model: Any, loader: Any, *, device: Any) -> dict[str, float]:
    torch, nn, _ = _torch_modules()
    tile_loss_fn = nn.CrossEntropyLoss()
    heatmap_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(12.0, device=device))
    model.eval()
    total_loss = 0.0
    total_tiles = 0
    correct_tiles = 0
    player_errors: list[float] = []
    monster_hits = 0
    monster_total = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            grids = batch["grid"].to(device)
            heatmaps = batch["heatmap"].to(device)
            output = model(images)
            loss = tile_loss_fn(output["tile_logits"], grids) + 0.35 * heatmap_loss_fn(
                output["heatmap_logits"],
                heatmaps,
            )
            total_loss += float(loss.cpu())
            batches += 1

            pred_grid = output["tile_logits"].argmax(dim=1)
            correct_tiles += int((pred_grid == grids).sum().cpu())
            total_tiles += int(grids.numel())

            probs = torch.sigmoid(output["heatmap_logits"]).cpu().numpy()
            true_heatmaps = heatmaps.cpu().numpy()
            for sample_idx in range(probs.shape[0]):
                pred_player_y, pred_player_x = np.unravel_index(
                    int(np.argmax(probs[sample_idx, 0])),
                    probs[sample_idx, 0].shape,
                )
                true_player_y, true_player_x = np.unravel_index(
                    int(np.argmax(true_heatmaps[sample_idx, 0])),
                    true_heatmaps[sample_idx, 0].shape,
                )
                player_errors.append(
                    math.hypot(pred_player_x - true_player_x, pred_player_y - true_player_y)
                )

                pred_monster_tiles = _tiles_from_heatmap(probs[sample_idx, 1], threshold=0.35, max_peaks=8)
                true_monster_tiles = set(zip(*np.where(grids[sample_idx].cpu().numpy() == TILE_MONSTER)[::-1]))
                monster_total += len(true_monster_tiles)
                monster_hits += len(set(pred_monster_tiles) & true_monster_tiles)

    return {
        "val_loss": total_loss / max(1, batches),
        "tile_acc": correct_tiles / max(1, total_tiles),
        "player_center_error_px": float(np.mean(player_errors)) if player_errors else 0.0,
        "monster_tile_recall": monster_hits / max(1, monster_total),
    }


def _tile_class_weights(grids: np.ndarray) -> Any:
    torch, _, _ = _torch_modules()
    counts = np.bincount(grids.reshape(-1).astype(np.int64), minlength=NUM_TILE_CLASSES).astype(np.float32)
    # 频率越低权重越高；sqrt 可以避免极少类权重过大导致训练不稳定。
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
    model = TinyPerceptionCNN(num_classes=int(checkpoint.get("num_tile_classes", NUM_TILE_CLASSES)))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, torch


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
    image = frame.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).float().to(resolved_device)
    with torch.no_grad():
        output = model(tensor)
        grid = output["tile_logits"].argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        heatmaps = torch.sigmoid(output["heatmap_logits"])[0].cpu().numpy()

    player_conf = float(heatmaps[0].max())
    player = None
    if player_conf >= confidence_threshold:
        y, x = np.unravel_index(int(np.argmax(heatmaps[0])), heatmaps[0].shape)
        player = {"center_px": (float(x), float(y)), "confidence": player_conf}
    elif np.any(grid == TILE_PLAYER):
        y, x = np.argwhere(grid == TILE_PLAYER)[0]
        player = {
            "center_px": (float(x * TILE_SIZE + TILE_SIZE * 0.5), float(y * TILE_SIZE + TILE_SIZE * 0.5)),
            "confidence": 0.0,
        }

    monsters = [
        {"center_px": (float(x), float(y)), "confidence": float(conf), "entity_type": "unknown"}
        for x, y, conf in _peaks_from_heatmap(heatmaps[1], threshold=confidence_threshold, max_peaks=8)
    ]
    if not monsters:
        for y, x in np.argwhere(grid == TILE_MONSTER):
            monsters.append(
                {
                    "center_px": (float(x * TILE_SIZE + TILE_SIZE * 0.5), float(y * TILE_SIZE + TILE_SIZE * 0.5)),
                    "confidence": 0.0,
                    "entity_type": "unknown",
                }
            )

    return {"grid": grid, "player": player, "monsters": tuple(monsters), "heatmaps": heatmaps}


def _collect_builtin_samples(
    *,
    target_count: int,
    rng: random.Random,
    images: list[np.ndarray],
    grids: list[np.ndarray],
    player_centers: list[np.ndarray],
    monster_centers: list[np.ndarray],
    monster_masks: list[np.ndarray],
    max_monsters: int,
) -> None:
    from nesylink.env import make_env

    if target_count <= 0:
        return
    per_task = max(1, math.ceil(target_count / len(BUILTIN_TASKS)))
    for task_id in BUILTIN_TASKS:
        env = make_env(task_id=task_id, observation_mode="full", render_mode="rgb_array")
        label, _info = env.reset(seed=rng.randrange(10_000_000))
        for _ in range(per_task):
            _append_sample(env, label, images, grids, player_centers, monster_centers, monster_masks, max_monsters)
            action = int(rng.randrange(env.action_space.n))
            label, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                label, _info = env.reset(seed=rng.randrange(10_000_000))
            if len(images) >= target_count:
                env.close()
                return
        env.close()


def _collect_random_room_samples(
    *,
    target_count: int,
    rng: random.Random,
    images: list[np.ndarray],
    grids: list[np.ndarray],
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
        label, _info = env.reset(seed=rng.randrange(10_000_000))
        for _ in range(rng.randint(3, 9)):
            _append_sample(env, label, images, grids, player_centers, monster_centers, monster_masks, max_monsters)
            made += 1
            if made >= target_count:
                break
            action = int(rng.randrange(env.action_space.n))
            label, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                label, _info = env.reset(seed=rng.randrange(10_000_000))
        env.close()


def _append_sample(
    env: Any,
    label: dict[str, np.ndarray],
    images: list[np.ndarray],
    grids: list[np.ndarray],
    player_centers: list[np.ndarray],
    monster_centers: list[np.ndarray],
    monster_masks: list[np.ndarray],
    max_monsters: int,
) -> None:
    frame = env.render()[:MAP_PIXEL_HEIGHT, :MAP_PIXEL_WIDTH].astype(np.uint8)
    grid = label["grid"].astype(np.uint8)
    player_center = np.asarray(label["player_position_px"], dtype=np.float32) + TILE_SIZE * 0.5
    centers = np.full((max_monsters, 2), -1.0, dtype=np.float32)
    masks = np.zeros((max_monsters,), dtype=np.bool_)
    raw_centers = np.asarray(label.get("monsters_position_px", []), dtype=np.float32)
    raw_masks = np.asarray(label.get("monsters_active_mask", []), dtype=np.bool_)
    count = min(max_monsters, len(raw_centers), len(raw_masks))
    if count:
        centers[:count] = raw_centers[:count] + TILE_SIZE * 0.5
        masks[:count] = raw_masks[:count]

    images.append(frame)
    grids.append(grid)
    player_centers.append(player_center)
    monster_centers.append(centers)
    monster_masks.append(masks)


def _random_room_payload(rng: random.Random) -> dict[str, Any]:
    occupied: set[tuple[int, int]] = set()
    spawn = (rng.randrange(1, GRID_WIDTH - 1), rng.randrange(1, GRID_HEIGHT - 1))
    occupied.add(spawn)
    layout: list[str] = []
    exit_floor = {(4, 0), (5, 0), (4, GRID_HEIGHT - 1), (5, GRID_HEIGHT - 1), (0, 3), (0, 4), (GRID_WIDTH - 1, 3), (GRID_WIDTH - 1, 4)}
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

    for idx in range(rng.randint(1, 4)):
        if (pos := take_tile()) is not None:
            objects.append({"id": f"chest_{idx}", "kind": "chest", "pos": list(pos), "loot": {"kind": "key"}})
    for idx in range(rng.randint(2, 8)):
        if (pos := take_tile()) is not None:
            objects.append({"id": f"trap_{idx}", "kind": "trap", "pos": list(pos), "damage": 1})
    for idx in range(rng.randint(0, 2)):
        if (pos := take_tile()) is not None:
            objects.append({"id": f"button_{idx}", "kind": "button", "pos": list(pos)})
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
            exits.append(
                {
                    "id": f"{direction}_exit",
                    "direction": direction,
                    "target_room": "random_room",
                    "target_entry": "default",
                    "type": "normal",
                }
            )

    dynamic_objects = []
    if rng.random() < 0.55:
        dynamic_tiles = []
        for _ in range(rng.randint(2, 8)):
            if (pos := take_tile()) is not None:
                dynamic_tiles.append(list(pos))
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


def _augment_image(image: np.ndarray) -> np.ndarray:
    # 颜色增强用于避免模型只记住固定 RGB；测试渲染细节变化时会更稳。
    rng = np.random.default_rng()
    out = image.copy()
    brightness = rng.uniform(-0.12, 0.12)
    contrast = rng.uniform(0.75, 1.35)
    out = (out - 0.5) * contrast + 0.5 + brightness
    if rng.random() < 0.20:
        gray = out.mean(axis=2, keepdims=True)
        out = gray * rng.uniform(0.6, 1.0) + out * rng.uniform(0.0, 0.4)
    channel_scale = rng.uniform(0.75, 1.25, size=(1, 1, 3))
    out = out * channel_scale
    out += rng.normal(0.0, 0.025, size=out.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _peaks_from_heatmap(
    heatmap: np.ndarray,
    *,
    threshold: float,
    max_peaks: int,
    min_distance_px: float = 9.0,
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


def _tiles_from_heatmap(heatmap: np.ndarray, *, threshold: float, max_peaks: int) -> list[tuple[int, int]]:
    return [(x // TILE_SIZE, y // TILE_SIZE) for x, y, _ in _peaks_from_heatmap(heatmap, threshold=threshold, max_peaks=max_peaks)]


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
    collect_parser.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default=str(DEFAULT_DATASET))
    train_parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    train_parser.add_argument("--epochs", type=int, default=14)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")

    combo_parser = subparsers.add_parser("collect-train")
    combo_parser.add_argument("--data", default=str(DEFAULT_DATASET))
    combo_parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    combo_parser.add_argument("--samples", type=int, default=2400)
    combo_parser.add_argument("--builtin-ratio", type=float, default=0.45)
    combo_parser.add_argument("--epochs", type=int, default=14)
    combo_parser.add_argument("--batch-size", type=int, default=64)
    combo_parser.add_argument("--lr", type=float, default=1e-3)
    combo_parser.add_argument("--seed", type=int, default=0)
    combo_parser.add_argument("--device", default="auto")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--data", default=str(DEFAULT_DATASET))
    eval_parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    if args.command == "collect":
        path = collect_dataset(
            DatasetConfig(
                output=Path(args.output),
                samples=args.samples,
                builtin_ratio=args.builtin_ratio,
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
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "collect-train":
        path = collect_dataset(
            DatasetConfig(
                output=Path(args.data),
                samples=args.samples,
                builtin_ratio=args.builtin_ratio,
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
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "eval":
        torch, _, _ = _torch_modules()
        model, _torch = load_model(args.weights, device=args.device)
        data = np.load(args.data)
        indices = np.arange(data["images"].shape[0])
        ds = PerceptionDataset(args.data, indices=indices, augment=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        metrics = evaluate_model(model, loader, device=_resolve_device(torch, args.device))
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
