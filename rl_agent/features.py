from __future__ import annotations

from typing import Any

import numpy as np

try:
    from nesylink.core.observation import TILE_PLAYER
except Exception:
    TILE_PLAYER = 2


GridStateKey = tuple[Any, ...]


def state_key_from_grid_obs(obs: dict[str, np.ndarray], info: dict[str, Any], task_id: str | None = None) -> GridStateKey:
    """Encode a structured grid observation into a compact tabular-RL key."""

    grid = np.asarray(obs["grid"], dtype=np.uint8)
    player = tuple(int(value) for value in np.asarray(obs["player_tile"]).tolist())
    health = int(np.asarray(obs.get("health", [0])).reshape(-1)[0])
    keys = int(np.asarray(obs.get("keys", [0])).reshape(-1)[0])

    return _build_key(
        grid=grid,
        player=player,
        health=health,
        keys=keys,
        info=info,
        task_id=task_id,
    )


def state_key_from_symbolic_state(state: Any, info: dict[str, Any], task_id: str | None = None) -> GridStateKey:
    """Encode PerceptionEngine output into the same style of state key."""

    if getattr(state, "static_grid", None):
        grid = np.asarray(state.static_grid, dtype=np.uint8)
    else:
        grid = np.zeros((8, 10), dtype=np.uint8)
        for name, code in (
            ("walls", 1),
            ("monsters", 3),
            ("chests", 4),
            ("exits", 5),
            ("traps", 6),
            ("buttons", 7),
            ("gaps", 9),
            ("bridges", 10),
            ("switches", 11),
        ):
            for x, y in getattr(state, name, frozenset()):
                if 0 <= x < 10 and 0 <= y < 8:
                    grid[y, x] = code
        x, y = getattr(state, "player", (-1, -1))
        if 0 <= x < 10 and 0 <= y < 8:
            grid[y, x] = TILE_PLAYER

    inventory = info.get("inventory", {}) if isinstance(info, dict) else {}
    agent = info.get("agent", {}) if isinstance(info, dict) else {}
    player = tuple(int(value) for value in getattr(state, "player", agent.get("tile", (-1, -1))))
    health = int(agent.get("hp", getattr(state, "health", 0)) or 0)
    keys = int(inventory.get("keys", getattr(state, "keys", 0)) or 0)

    return _build_key(
        grid=grid,
        player=player,
        health=health,
        keys=keys,
        info=info,
        task_id=task_id,
    )


def _build_key(
    *,
    grid: np.ndarray,
    player: tuple[int, int],
    health: int,
    keys: int,
    info: dict[str, Any],
    task_id: str | None,
) -> GridStateKey:
    env_info = info.get("env", {}) if isinstance(info, dict) else {}
    room_id = str(env_info.get("room_id", ""))
    room_coord = tuple(int(v) for v in env_info.get("room_coord", (0, 0)) or (0, 0))

    entities = info.get("entities", {}) if isinstance(info, dict) else {}
    dynamic = info.get("dynamic", {}) if isinstance(info, dict) else {}
    dynamic_tiles = tuple(
        sorted(
            (
                tuple(int(v) for v in item.get("pos", (-1, -1))),
                str(item.get("tile", "")),
            )
            for item in dynamic.get("current_room_tiles", [])
            if isinstance(item, dict)
        )
    )

    normalized_grid = np.asarray(grid, dtype=np.uint8).copy()
    if int(entities.get("chests_remaining", 0) or 0) == 0:
        normalized_grid[normalized_grid == 4] = 0
    if int(entities.get("monsters_remaining", 0) or 0) == 0:
        normalized_grid[normalized_grid == 3] = 0
    if 0 <= player[0] < normalized_grid.shape[1] and 0 <= player[1] < normalized_grid.shape[0]:
        normalized_grid[normalized_grid == TILE_PLAYER] = 0
        normalized_grid[player[1], player[0]] = TILE_PLAYER

    return (
        task_id or "",
        room_id,
        room_coord,
        player,
        min(max(health, 0), 9),
        min(max(keys, 0), 3),
        int(entities.get("monsters_remaining", 0) or 0),
        int(entities.get("chests_remaining", 0) or 0),
        int(entities.get("exits_open", 0) or 0),
        dynamic_tiles,
        tuple(int(value) for value in normalized_grid.reshape(-1).tolist()),
    )


