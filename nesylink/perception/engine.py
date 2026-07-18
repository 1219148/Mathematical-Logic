from __future__ import annotations

from dataclasses import replace
import math
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
MONSTER_TRACK_ALPHA = 0.60
MONSTER_TRACK_BETA = 0.10
MONSTER_TRACK_MATCH_DISTANCE_PX = TILE_SIZE * 2.5
PLAYER_TRACK_ALPHA = 0.50
PLAYER_TRACK_BETA = 0.10
ROOM_TRANSITION_DISTANCE_PX = TILE_SIZE * 2.5


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
        self._monster_tracks: list[
            tuple[tuple[float, float], tuple[float, float]]
        ] = []
        self._player_track: tuple[
            tuple[float, float], tuple[float, float]
        ] | None = None
        self._last_player_center: tuple[float, float] | None = None

    def reset(self, obs: np.ndarray) -> None:
        del obs
        self._monster_tracks.clear()
        self._player_track = None
        self._last_player_center = None

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
        opened_chest_tiles = set(prediction.get("opened_chests", ()))
        closed_chest_tiles = set(prediction.get("closed_chests", _all_tiles(grid, TILE_CHEST)))
        detected_chest_tiles = opened_chest_tiles | closed_chest_tiles
        predicted_exit_types = prediction.get("exit_types", {})
        predicted_exit_tiles = set(predicted_exit_types)
        resolved_exit_tiles = (
            predicted_exit_tiles
            if predicted_exit_tiles
            else _all_tiles(grid, TILE_EXIT)
        )
        player_grid_tile = _first_tile(grid, TILE_PLAYER)
        bridge_tiles = _all_tiles(grid, TILE_BRIDGE)
        if player_grid_tile is not None and _tile_between_bridge_neighbors(
            player_grid_tile,
            bridge_tiles,
        ):
            bridge_tiles.add(player_grid_tile)
        bridge_tiles.update(
            exit_tile
            for exit_tile in resolved_exit_tiles
            if any(
                _tile_manhattan(exit_tile, bridge_tile) == 1
                for bridge_tile in bridge_tiles
            )
        )
        if player_grid_tile is not None and _tile_between_bridge_neighbors(
            player_grid_tile,
            bridge_tiles,
        ):
            bridge_tiles.add(player_grid_tile)
        npc_tiles = _all_tiles(grid, TILE_NPC)
        monster_grid_tiles = _all_tiles(grid, TILE_MONSTER) - detected_chest_tiles

        player_entity = _entity_from_detection(prediction.get("player"), "player")
        player_entity = self._stabilize_player_entity(player_entity)
        monster_entities = tuple(
            entity
            for detection in prediction.get("monsters", ())
            if (entity := _entity_from_detection(detection, "monster")) is not None
            and entity.tile not in detected_chest_tiles
            and entity.tile not in npc_tiles
        )
        monster_entities = self._stabilize_monster_entities(
            monster_entities,
            player_entity,
        )

        # The tile head is trained directly against logical occupancy, while the
        # heatmap head estimates a continuous rendering centre.  At an exact tile
        # boundary, converting the centre with floor division can flicker to an
        # adjacent tile even when the occupancy prediction is stable.  Keep the
        # continuous centre in ``player_entity`` but use the learned tile output
        # for symbolic path planning whenever it is available.
        player_tile = player_grid_tile
        if player_tile is None:
            player_tile = player_entity.tile if player_entity is not None else (-1, -1)

        monster_tiles = {entity.tile for entity in monster_entities}
        monster_tiles.update(
            tile
            for tile in monster_grid_tiles
            if not monster_entities
            or all(_tile_manhattan(tile, entity.tile) > 1 for entity in monster_entities)
        )

        return SymbolicState(
            player=player_tile,
            walls=frozenset(_all_tiles(grid, TILE_WALL)),
            health=0,
            keys=0,
            monsters=frozenset(monster_tiles),
            monster_types={entity.tile: entity.entity_type for entity in monster_entities},
            chests=frozenset(closed_chest_tiles),
            traps=frozenset(_all_tiles(grid, TILE_TRAP)),
            exits=frozenset(
                resolved_exit_tiles - detected_chest_tiles
            ),
            exit_types={
                tile: exit_type
                for tile, exit_type in predicted_exit_types.items()
                if tile not in detected_chest_tiles
            },
            buttons=frozenset(_all_tiles(grid, TILE_BUTTON)),
            gaps=frozenset(_all_tiles(grid, TILE_GAP)),
            bridges=frozenset(bridge_tiles),
            switches=frozenset(_all_tiles(grid, TILE_SWITCH)),
            player_entity=player_entity,
            monster_entities=monster_entities,
            static_grid=tuple(tuple(int(value) for value in row) for row in grid.tolist()),
        )

    def _stabilize_player_entity(
        self,
        entity: EntityState | None,
    ) -> EntityState | None:
        """Filter sub-pixel localization noise without changing the logical tile.

        The renderer advances entities in discrete pixel increments, while the
        learned heatmap/offset heads estimate a continuous centre.  A symmetric
        alpha-beta filter preserves constant-velocity motion but suppresses a
        one-frame reversal around a tile centre.  Large room-transition jumps
        start a fresh track rather than being smoothed across rooms.
        """

        if entity is None:
            self._player_track = None
            return None

        observed = entity.center_px
        if self._player_track is None:
            filtered = observed
            velocity = (0.0, 0.0)
        else:
            previous, previous_velocity = self._player_track
            predicted = (
                previous[0] + previous_velocity[0],
                previous[1] + previous_velocity[1],
            )
            if math.dist(observed, predicted) > ROOM_TRANSITION_DISTANCE_PX:
                filtered = observed
                velocity = (0.0, 0.0)
            else:
                residual = (
                    observed[0] - predicted[0],
                    observed[1] - predicted[1],
                )
                filtered = (
                    predicted[0] + PLAYER_TRACK_ALPHA * residual[0],
                    predicted[1] + PLAYER_TRACK_ALPHA * residual[1],
                )
                velocity = (
                    previous_velocity[0] + PLAYER_TRACK_BETA * residual[0],
                    previous_velocity[1] + PLAYER_TRACK_BETA * residual[1],
                )

        self._player_track = (filtered, velocity)
        half_size = TILE_SIZE * 0.5
        return replace(
            entity,
            center_px=filtered,
            bbox_px=(
                filtered[0] - half_size,
                filtered[1] - half_size,
                filtered[0] + half_size,
                filtered[1] + half_size,
            ),
        )

    def _stabilize_monster_entities(
        self,
        entities: tuple[EntityState, ...],
        player: EntityState | None,
    ) -> tuple[EntityState, ...]:
        """Track sub-pixel motion so raster quantization cannot flicker a tile."""

        if player is not None:
            if (
                self._last_player_center is not None
                and math.dist(player.center_px, self._last_player_center)
                > ROOM_TRANSITION_DISTANCE_PX
            ):
                self._monster_tracks.clear()
            self._last_player_center = player.center_px

        candidates = sorted(
            (
                math.dist(entity.center_px, track_center),
                entity_index,
                track_index,
            )
            for entity_index, entity in enumerate(entities)
            for track_index, (track_center, _velocity) in enumerate(
                self._monster_tracks
            )
        )
        entity_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for distance, entity_index, track_index in candidates:
            if distance > MONSTER_TRACK_MATCH_DISTANCE_PX:
                break
            if entity_index in entity_to_track or track_index in used_tracks:
                continue
            entity_to_track[entity_index] = track_index
            used_tracks.add(track_index)

        new_tracks: list[tuple[tuple[float, float], tuple[float, float]]] = []
        stabilized: list[EntityState] = []
        for entity_index, entity in enumerate(entities):
            track_index = entity_to_track.get(entity_index)
            if track_index is None:
                filtered_center = entity.center_px
                velocity = (0.0, 0.0)
            else:
                previous_center, previous_velocity = self._monster_tracks[track_index]
                predicted_center = (
                    previous_center[0] + previous_velocity[0],
                    previous_center[1] + previous_velocity[1],
                )
                residual = (
                    entity.center_px[0] - predicted_center[0],
                    entity.center_px[1] - predicted_center[1],
                )
                filtered_center = (
                    predicted_center[0] + MONSTER_TRACK_ALPHA * residual[0],
                    predicted_center[1] + MONSTER_TRACK_ALPHA * residual[1],
                )
                velocity = (
                    previous_velocity[0] + MONSTER_TRACK_BETA * residual[0],
                    previous_velocity[1] + MONSTER_TRACK_BETA * residual[1],
                )
            new_tracks.append((filtered_center, velocity))
            stabilized.append(
                replace(entity, tile=_tile_from_center(filtered_center))
            )

        self._monster_tracks = new_tracks
        return tuple(stabilized)

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


def _tile_manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _tile_between_bridge_neighbors(
    tile: tuple[int, int],
    bridge_tiles: set[tuple[int, int]],
) -> bool:
    x, y = tile
    return (
        (x - 1, y) in bridge_tiles and (x + 1, y) in bridge_tiles
    ) or (
        (x, y - 1) in bridge_tiles and (x, y + 1) in bridge_tiles
    )


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
