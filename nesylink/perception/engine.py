from __future__ import annotations

from pathlib import Path

import numpy as np

from nesylink.core.constants import MAP_PIXEL_HEIGHT, TILE_SIZE
from nesylink.core.observation import (
    TILE_BRIDGE,
    TILE_BUTTON,
    TILE_CHEST,
    TILE_EXIT,
    TILE_GAP,
    TILE_MONSTER,
    TILE_NPC,
    TILE_PLAYER,
    TILE_SWITCH,
    TILE_TRAP,
    TILE_WALL,
)
from nesylink.shared import EntityState, SymbolicState


DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("perception_model.pt")


class PerceptionEngine:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self._model = None
        self._torch = None

    def reset(self, obs: np.ndarray) -> None:
        del obs

    def extract(self, obs: np.ndarray) -> SymbolicState:
        """从 raw pixels 中抽取符号状态。

        最终测评不能直接读 info，所以这里只使用 obs。CNN 负责还原 8x10 语义图，
        并给出玩家/怪物中心点；SymbolicState 同时保留 tile 和 pixel 信息。
        """
        if self._model is None:
            self._load_model()

        frame = _normalize_frame(obs)
        from .cnn import predict_frame

        prediction = predict_frame(
            self._model,
            frame,
            torch_module=self._torch,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        grid = prediction["grid"]
        player_entity = _entity_from_detection(prediction.get("player"), "player")
        monster_entities = tuple(
            entity
            for detection in prediction.get("monsters", ())
            if (entity := _entity_from_detection(detection, "monster")) is not None
        )

        player_tile = _first_tile(grid, TILE_PLAYER)
        if player_entity is not None:
            player_tile = player_entity.tile
        if player_tile is None:
            player_tile = (-1, -1)

        monster_tiles = {entity.tile for entity in monster_entities}
        monster_tiles.update(_all_tiles(grid, TILE_MONSTER))

        return SymbolicState(
            player=player_tile,
            walls=frozenset(_all_tiles(grid, TILE_WALL)),
            health=0,
            keys=0,
            monsters=frozenset(monster_tiles),
            monster_types={entity.tile: entity.entity_type for entity in monster_entities},
            chests=frozenset(_all_tiles(grid, TILE_CHEST)),
            traps=frozenset(_all_tiles(grid, TILE_TRAP)),
            exits=frozenset(_all_tiles(grid, TILE_EXIT)),
            buttons=frozenset(_all_tiles(grid, TILE_BUTTON)),
            gaps=frozenset(_all_tiles(grid, TILE_GAP)),
            bridges=frozenset(_all_tiles(grid, TILE_BRIDGE)),
            switches=frozenset(_all_tiles(grid, TILE_SWITCH)),
            player_entity=player_entity,
            monster_entities=monster_entities,
            static_grid=tuple(tuple(int(value) for value in row) for row in grid.tolist()),
        )

    def _load_model(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"perception 权重不存在: {self.weights_path}. "
                "请先运行 `python -m nesylink.perception.cnn collect-train` 训练模型。"
            )
        from .cnn import load_model

        self._model, self._torch = load_model(self.weights_path, device=self.device)


def _normalize_frame(obs: np.ndarray) -> np.ndarray:
    frame = np.asarray(obs)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"perception obs 必须是 HxWx3 RGB 图像，当前 shape={frame.shape}")
    if frame.shape[0] > MAP_PIXEL_HEIGHT:
        frame = frame[:MAP_PIXEL_HEIGHT]
    return frame.astype(np.uint8, copy=False)


def _tile_from_center(center_px: tuple[float, float]) -> tuple[int, int]:
    return int(center_px[0] // TILE_SIZE), int(center_px[1] // TILE_SIZE)


def _entity_from_detection(detection: dict | None, kind: str) -> EntityState | None:
    if not detection:
        return None
    center = tuple(float(value) for value in detection["center_px"])
    tile = _tile_from_center(center)
    bbox = (
        center[0] - TILE_SIZE * 0.5,
        center[1] - TILE_SIZE * 0.5,
        center[0] + TILE_SIZE * 0.5,
        center[1] + TILE_SIZE * 0.5,
    )
    return EntityState(
        tile=tile,
        center_px=center,
        bbox_px=bbox,
        kind=kind,
        entity_type=str(detection.get("entity_type", "")),
        confidence=float(detection.get("confidence", 1.0)),
    )


def _first_tile(grid: np.ndarray, code: int) -> tuple[int, int] | None:
    ys, xs = np.where(grid == code)
    if len(xs) == 0:
        return None
    return int(xs[0]), int(ys[0])


def _all_tiles(grid: np.ndarray, code: int) -> set[tuple[int, int]]:
    ys, xs = np.where(grid == code)
    return {(int(x), int(y)) for y, x in zip(ys, xs)}
