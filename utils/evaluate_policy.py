"""Evaluate a submitted policy on NesyLink mathematical-logic tasks.

Expected policy interfaces:

- a module-level `act(obs, info) -> int`
- a module-level `policy` object with `.act(obs, info)` or `__call__(obs, info)`
- a `Policy` class with `.act(obs, info)`
- a `make_policy()` function returning any of the above

Example:

    python utils/evaluate_policy.py --policy submissions/student_policy.py
    python utils/evaluate_policy.py --policy submissions.student_policy:make_policy --tasks mathematical_logic/task_3
"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import MAP_PIXEL_HEIGHT, MAP_PIXEL_WIDTH, TILE_SIZE
from nesylink.core.world.loader import load_map
from nesylink.env import make_env
from nesylink.tasks import list_tasks


DEFAULT_TASKS = tuple(f"mathematical_logic/task_{index}" for index in range(1, 6))
COLOR_OBS_VARIANTS = (
    "default",
    "grayscale",
    "dark",
    "bright",
    "high_contrast",
    "inverted",
)
REDRAW_OBS_VARIANTS = (
    "redraw_geometric",
    "redraw_symbols",
)
OBS_VARIANTS = COLOR_OBS_VARIANTS + REDRAW_OBS_VARIANTS
EVAL_STAGE_ORDER = {
    "original": 0,
    "spatial": 1,
    "color": 2,
    "redraw": 3,
}
SPATIAL_TASKS = {
    "mathematical_logic/task_1",
    "mathematical_logic/task_2",
    "mathematical_logic/task_3",
    "mathematical_logic/task_4",
    "mathematical_logic/task_5",
}
SPATIAL_MAP_VARIANTS = (
    "spatial_a",
    "spatial_b",
    "spatial_c",
)

TASK_MILESTONES: dict[str, tuple[str, ...]] = {
    "mathematical_logic/task_3": (
        "monster_killed",
        "key_collected",
    ),
    "mathematical_logic/task_4": (
        "switch_activated",
        "key_collected",
        "door_opened",
        "item_collected",
        "monster_killed",
    ),
}

TASK5_EVENTS = (
    "chest_opened",
    "key_collected",
    "gold_collected",
    "item_collected",
    "agent_healed",
    "button_pressed",
    "room_changed",
    "door_opened",
    "trap_triggered",
    "monster_killed",
    "exit_reached",
    "environment_completed",
    "world_completed",
)
PROGRESS_EVENTS = tuple(
    dict.fromkeys(
        (
            "monster_killed",
            "key_collected",
            *TASK5_EVENTS,
            "agent_dead",
        )
    )
)


@dataclass(frozen=True)
class EpisodePlanEntry:
    task_id: str
    eval_stage: str
    obs_variant: str
    seed: int
    map_variant: str = "default"


@dataclass
class EpisodeResult:
    task_id: str
    eval_stage: str
    obs_variant: str
    seed: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    success: bool
    terminal_reason: str | None
    event_counts: dict[str, int]
    milestones: dict[str, bool]
    map_variant: str = "default"


@dataclass(frozen=True)
class PolicyBinding:
    policy: Any
    receives_task_id: bool


def split_policy_spec(spec: str) -> tuple[str, str | None]:
    if ":" not in spec:
        return spec, None
    target, attr = spec.rsplit(":", 1)
    return target, attr or None


def load_module(target: str):
    path = Path(target)
    if path.suffix == ".py" or path.exists():
        module_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not module_path.exists():
            raise FileNotFoundError(f"policy file not found: {module_path}")
        module_name = f"_nesylink_policy_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load policy module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def load_policy(policy_spec: str):
    target, attr = split_policy_spec(policy_spec)
    module = load_module(target)

    candidate_names = (attr,) if attr else ("make_policy", "Policy", "policy", "act")
    for name in candidate_names:
        if name is None or not hasattr(module, name):
            continue
        candidate = getattr(module, name)
        if name == "make_policy":
            return candidate()
        if name == "Policy" and isinstance(candidate, type):
            return candidate()
        return candidate

    expected = ", ".join(candidate_names)
    raise AttributeError(f"policy module must expose one of: {expected}")


def parse_task_policy_specs(specs: list[str] | None) -> dict[str, str]:
    policies: dict[str, str] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--task-policy must use TASK_ID=POLICY_SPEC format, got: {spec}")
        task_id, policy_spec = spec.split("=", 1)
        task_id = task_id.strip()
        policy_spec = policy_spec.strip()
        if not task_id or not policy_spec:
            raise ValueError(f"--task-policy must use TASK_ID=POLICY_SPEC format, got: {spec}")
        policies[task_id] = policy_spec
    return policies


def resolve_policies(
    *,
    default_policy_spec: str | None,
    task_policy_specs: list[str] | None,
    task_ids: list[str],
) -> dict[str, PolicyBinding]:
    task_policy_map = parse_task_policy_specs(task_policy_specs)
    missing_task_ids = [
        task_id
        for task_id in task_ids
        if task_id not in task_policy_map and default_policy_spec is None
    ]
    if missing_task_ids:
        missing = ", ".join(missing_task_ids)
        raise ValueError(f"missing policy for task(s): {missing}")

    loaded_by_spec: dict[str, Any] = {}
    if default_policy_spec is not None:
        loaded_by_spec[default_policy_spec] = load_policy(default_policy_spec)

    policies: dict[str, PolicyBinding] = {}
    for task_id in task_ids:
        policy_spec = task_policy_map.get(task_id, default_policy_spec)
        assert policy_spec is not None
        if policy_spec not in loaded_by_spec:
            loaded_by_spec[policy_spec] = load_policy(policy_spec)
        policies[task_id] = PolicyBinding(
            policy=loaded_by_spec[policy_spec],
            receives_task_id=task_id in task_policy_map,
        )
    return policies


def reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset", None)
    if reset is None:
        return
    if not callable(reset):
        raise TypeError("policy.reset must be callable")
    reset()


def call_policy(policy: Any, obs: np.ndarray, info: dict[str, Any]) -> int:
    actor: Callable[..., Any]
    if hasattr(policy, "act"):
        actor = policy.act
    elif callable(policy):
        actor = policy
    else:
        raise TypeError("policy must be callable or expose an act(obs, info) method")

    try:
        action = actor(obs, info)
    except TypeError:
        action = actor(obs)

    if isinstance(action, dict):
        action = action.get("action")
    if isinstance(action, (tuple, list)) and action:
        action = action[0]
    return int(np.asarray(action).item())


def build_policy_info(
    *,
    info_mode: str,
    raw_info: dict[str, Any],
    last_reward: float,
    task_id: str | None,
) -> dict[str, Any]:
    if info_mode == "full":
        return raw_info
    if info_mode != "safe":
        raise ValueError(f"unknown info mode {info_mode!r}")
    return build_safe_info(
        raw_info=raw_info,
        last_reward=last_reward,
        task_id=task_id,
    )


def build_safe_info(
    *,
    raw_info: dict[str, Any],
    last_reward: float,
    task_id: str | None,
) -> dict[str, Any]:
    policy_info = {
        "last_reward": float(last_reward),
        "inventory": _safe_inventory_info(raw_info),
    }
    if task_id is not None:
        policy_info["task_id"] = task_id
    return policy_info


def _safe_inventory_info(raw_info: dict[str, Any]) -> dict[str, Any]:
    inventory = raw_info.get("inventory", {})
    player = raw_info.get("player", {})
    if not isinstance(inventory, dict) and isinstance(player, dict):
        inventory = player.get("inventory", {})
    inventory = inventory if isinstance(inventory, dict) else {}
    return {
        "gold": copy.deepcopy(inventory.get("gold")),
        "keys": copy.deepcopy(inventory.get("keys")),
        "items": copy.deepcopy(inventory.get("items", [])),
        "tools": copy.deepcopy(inventory.get("tools", [])),
        "equipped": copy.deepcopy(inventory.get("equipped", {})),
    }


def apply_obs_variant(
    obs: np.ndarray,
    variant: str,
    *,
    info: dict[str, Any] | None = None,
    env: Any | None = None,
) -> np.ndarray:
    del info
    if variant == "default":
        return obs

    image = np.asarray(obs)
    if variant == "grayscale":
        gray = image.mean(axis=2, keepdims=True).astype(np.uint8)
        return np.repeat(gray, 3, axis=2)
    if variant == "dark":
        return (image.astype(np.float32) * 0.55).clip(0, 255).astype(np.uint8)
    if variant == "bright":
        return (image.astype(np.float32) * 1.35).clip(0, 255).astype(np.uint8)
    if variant == "high_contrast":
        return np.where(image > 127, 255, 0).astype(np.uint8)
    if variant == "inverted":
        return 255 - image
    if variant in REDRAW_OBS_VARIANTS:
        if env is None:
            raise ValueError(f"observation variant {variant!r} requires an environment")
        return redraw_obs_from_state(env, preset=variant.removeprefix("redraw_"), shape=image.shape)

    raise ValueError(f"unknown observation variant: {variant}")


def redraw_obs_from_state(env: Any, *, preset: str, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"redraw variants require an RGB observation shape, got {shape!r}")

    runtime = env.engine.runtime
    room = runtime.room
    player = runtime.player
    palette = _redraw_palette(preset)
    frame = np.zeros(shape, dtype=np.uint8)
    frame[:, :] = palette["floor"]

    _draw_grid(frame, palette["grid"])

    for (col, row), tile_kind in room.dynamic_tiles.items():
        if tile_kind == "gap":
            _draw_circle(frame, *_tile_center(col, row), 7, palette["gap"])
        elif tile_kind == "bridge":
            _fill_tile(frame, col, row, palette["bridge"], pad=3)

    for exit_config in room.exits:
        opened = room.exit_state(exit_config).opened
        color = palette["exit_open"] if opened else palette["exit_closed"]
        for col, row in exit_config.tiles:
            _draw_tile_outline(frame, col, row, color, thickness=3)

    for col, row in room.walls:
        _fill_tile(frame, col, row, palette["wall"], pad=1)

    for trap in room.traps.values():
        if not trap.is_active or room.dynamic_tiles.get(trap.pos) == "bridge":
            continue
        col, row = trap.pos
        if preset == "symbols":
            _draw_triangle(frame, *_tile_bounds(col, row, pad=3), "up", palette["trap"])
        else:
            _draw_x(frame, col, row, palette["trap"])

    for button in room.buttons.values():
        color = palette["button_pressed"] if button.is_pressed else palette["button"]
        _draw_circle(frame, *_tile_center(*button.pos), 5, color)

    for switch in room.switches.values():
        color = palette["switch_pressed"] if switch.is_pressed else palette["switch"]
        if preset == "symbols":
            _draw_slash(frame, *switch.pos, color)
        else:
            _draw_diamond(frame, *_tile_center(*switch.pos), 6, color)

    for chest in room.chests.values():
        if not chest.is_visible:
            continue
        color = palette["chest_open"] if chest.is_open else palette["chest"]
        col, row = chest.pos
        if preset == "symbols":
            _draw_tile_outline(frame, col, row, color, thickness=3, pad=3)
            _draw_circle(frame, *_tile_center(col, row), 3, palette["chest_mark"])
        else:
            _draw_diamond(frame, *_tile_center(col, row), 7, color)

    for npc in room.npcs.values():
        col, row = npc.pos
        if preset == "symbols":
            _draw_circle(frame, *_tile_center(col, row), 5, palette["npc"])
            _fill_rect(
                frame,
                col * TILE_SIZE + 7,
                row * TILE_SIZE + 8,
                col * TILE_SIZE + 9,
                row * TILE_SIZE + 14,
                palette["npc"],
            )
        else:
            _draw_plus(frame, col, row, palette["npc"])

    for monster in room.monsters.values():
        left = int(round(monster.position_px[0]))
        top = int(round(monster.position_px[1]))
        right = left + int(monster.size_px)
        bottom = top + int(monster.size_px)
        if preset == "symbols":
            _fill_rect(frame, left + 3, top + 2, right - 3, bottom - 2, palette["monster"])
            _fill_rect(frame, left + 7, top + 4, left + 9, bottom - 6, palette["monster_mark"])
            _fill_rect(frame, left + 7, bottom - 4, left + 9, bottom - 2, palette["monster_mark"])
        else:
            _draw_hex(frame, left + 8, top + 8, 7, palette["monster"])

    player_left = int(round(player.position_px[0]))
    player_top = int(round(player.position_px[1]))
    player_right = player_left + int(player.size_px)
    player_bottom = player_top + int(player.size_px)
    if preset == "symbols":
        _draw_triangle(
            frame,
            player_left + 2,
            player_top + 2,
            player_right - 2,
            player_bottom - 2,
            player.facing,
            palette["player"],
        )
        _draw_rect_outline(
            frame,
            player_left,
            player_top,
            player_right,
            player_bottom,
            palette["player_outline"],
            thickness=1,
            pad=1,
        )
    else:
        _draw_circle(frame, player_left + 8, player_top + 8, 6, palette["player"])
        _draw_triangle(
            frame,
            player_left + 5,
            player_top + 5,
            player_right - 5,
            player_bottom - 5,
            player.facing,
            palette["player_marker"],
        )

    return frame


def _redraw_palette(preset: str) -> dict[str, tuple[int, int, int]]:
    if preset == "geometric":
        return {
            "floor": (32, 36, 40),
            "grid": (54, 60, 66),
            "wall": (238, 238, 232),
            "gap": (2, 4, 8),
            "bridge": (144, 92, 42),
            "exit_open": (58, 150, 230),
            "exit_closed": (36, 80, 132),
            "trap": (160, 70, 190),
            "button": (38, 194, 104),
            "button_pressed": (18, 108, 64),
            "switch": (236, 154, 48),
            "switch_pressed": (126, 78, 28),
            "chest": (236, 200, 54),
            "chest_open": (136, 112, 34),
            "chest_mark": (32, 36, 40),
            "npc": (234, 92, 168),
            "monster": (224, 58, 58),
            "monster_mark": (255, 255, 255),
            "player": (40, 210, 220),
            "player_marker": (10, 62, 70),
            "player_outline": (255, 255, 255),
        }
    if preset == "symbols":
        return {
            "floor": (202, 206, 208),
            "grid": (176, 182, 186),
            "wall": (18, 20, 22),
            "gap": (0, 0, 0),
            "bridge": (142, 88, 40),
            "exit_open": (12, 92, 192),
            "exit_closed": (40, 54, 78),
            "trap": (74, 34, 112),
            "button": (0, 154, 74),
            "button_pressed": (0, 90, 46),
            "switch": (218, 112, 24),
            "switch_pressed": (118, 60, 18),
            "chest": (214, 164, 18),
            "chest_open": (116, 86, 12),
            "chest_mark": (18, 20, 22),
            "npc": (198, 52, 132),
            "monster": (184, 32, 38),
            "monster_mark": (255, 246, 210),
            "player": (248, 248, 244),
            "player_marker": (18, 20, 22),
            "player_outline": (18, 20, 22),
        }
    raise ValueError(f"unknown redraw preset: {preset}")


def _tile_bounds(col: int, row: int, *, pad: int = 0) -> tuple[int, int, int, int]:
    return (
        col * TILE_SIZE + pad,
        row * TILE_SIZE + pad,
        (col + 1) * TILE_SIZE - pad,
        (row + 1) * TILE_SIZE - pad,
    )


def _tile_center(col: int, row: int) -> tuple[int, int]:
    return col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2


def _clip_rect(frame: np.ndarray, left: int, top: int, right: int, bottom: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(frame.shape[1], left)),
        max(0, min(frame.shape[0], top)),
        max(0, min(frame.shape[1], right)),
        max(0, min(frame.shape[0], bottom)),
    )


def _fill_rect(
    frame: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    left, top, right, bottom = _clip_rect(frame, left, top, right, bottom)
    if right > left and bottom > top:
        frame[top:bottom, left:right] = color


def _fill_tile(frame: np.ndarray, col: int, row: int, color: tuple[int, int, int], *, pad: int = 0) -> None:
    _fill_rect(frame, *_tile_bounds(col, row, pad=pad), color)


def _draw_grid(frame: np.ndarray, color: tuple[int, int, int]) -> None:
    for x in range(0, min(MAP_PIXEL_WIDTH, frame.shape[1]), TILE_SIZE):
        _fill_rect(frame, x, 0, x + 1, min(MAP_PIXEL_HEIGHT, frame.shape[0]), color)
    for y in range(0, min(MAP_PIXEL_HEIGHT, frame.shape[0]), TILE_SIZE):
        _fill_rect(frame, 0, y, min(MAP_PIXEL_WIDTH, frame.shape[1]), y + 1, color)


def _draw_tile_outline(
    frame: np.ndarray,
    col: int,
    row: int,
    color: tuple[int, int, int],
    *,
    thickness: int,
    pad: int = 0,
) -> None:
    left, top, right, bottom = _tile_bounds(col, row, pad=pad)
    _draw_rect_outline(frame, left, top, right, bottom, color, thickness=thickness)


def _draw_rect_outline(
    frame: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
    *,
    thickness: int,
    pad: int = 0,
) -> None:
    left += pad
    top += pad
    right -= pad
    bottom -= pad
    _fill_rect(frame, left, top, right, top + thickness, color)
    _fill_rect(frame, left, bottom - thickness, right, bottom, color)
    _fill_rect(frame, left, top, left + thickness, bottom, color)
    _fill_rect(frame, right - thickness, top, right, bottom, color)


def _draw_circle(frame: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    left, top, right, bottom = _clip_rect(frame, cx - radius, cy - radius, cx + radius + 1, cy + radius + 1)
    if right <= left or bottom <= top:
        return
    yy, xx = np.ogrid[top:bottom, left:right]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    frame[top:bottom, left:right][mask] = color


def _draw_diamond(frame: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    left, top, right, bottom = _clip_rect(frame, cx - radius, cy - radius, cx + radius + 1, cy + radius + 1)
    if right <= left or bottom <= top:
        return
    yy, xx = np.ogrid[top:bottom, left:right]
    mask = np.abs(xx - cx) + np.abs(yy - cy) <= radius
    frame[top:bottom, left:right][mask] = color


def _draw_hex(frame: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    _fill_triangle(frame, (cx - radius, cy), (cx - radius // 2, cy - radius), (cx, cy - radius), color)
    _fill_triangle(frame, (cx, cy - radius), (cx + radius // 2, cy - radius), (cx + radius, cy), color)
    _fill_triangle(frame, (cx - radius, cy), (cx - radius // 2, cy + radius), (cx, cy + radius), color)
    _fill_triangle(frame, (cx, cy + radius), (cx + radius // 2, cy + radius), (cx + radius, cy), color)
    _fill_rect(frame, cx - radius // 2, cy - radius, cx + radius // 2 + 1, cy + radius + 1, color)


def _draw_triangle(
    frame: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    direction: str,
    color: tuple[int, int, int],
) -> None:
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    points_by_direction = {
        "up": ((cx, top), (left, bottom), (right, bottom)),
        "down": ((left, top), (right, top), (cx, bottom)),
        "left": ((left, cy), (right, top), (right, bottom)),
        "right": ((left, top), (right, cy), (left, bottom)),
    }
    p1, p2, p3 = points_by_direction.get(direction, points_by_direction["down"])
    _fill_triangle(frame, p1, p2, p3, color)


def _fill_triangle(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    p3: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    left = min(p1[0], p2[0], p3[0])
    right = max(p1[0], p2[0], p3[0]) + 1
    top = min(p1[1], p2[1], p3[1])
    bottom = max(p1[1], p2[1], p3[1]) + 1
    left, top, right, bottom = _clip_rect(frame, left, top, right, bottom)
    if right <= left or bottom <= top:
        return
    yy, xx = np.mgrid[top:bottom, left:right]
    denominator = ((p2[1] - p3[1]) * (p1[0] - p3[0])) + ((p3[0] - p2[0]) * (p1[1] - p3[1]))
    if denominator == 0:
        return
    a = ((p2[1] - p3[1]) * (xx - p3[0]) + (p3[0] - p2[0]) * (yy - p3[1])) / denominator
    b = ((p3[1] - p1[1]) * (xx - p3[0]) + (p1[0] - p3[0]) * (yy - p3[1])) / denominator
    c = 1 - a - b
    mask = (a >= 0) & (b >= 0) & (c >= 0)
    frame[top:bottom, left:right][mask] = color


def _draw_x(frame: np.ndarray, col: int, row: int, color: tuple[int, int, int]) -> None:
    left, top, _, _ = _tile_bounds(col, row)
    for offset in range(3, TILE_SIZE - 3):
        _fill_rect(frame, left + offset, top + offset, left + offset + 2, top + offset + 2, color)
        _fill_rect(
            frame,
            left + TILE_SIZE - offset - 2,
            top + offset,
            left + TILE_SIZE - offset,
            top + offset + 2,
            color,
        )


def _draw_slash(frame: np.ndarray, col: int, row: int, color: tuple[int, int, int]) -> None:
    left, top, _, _ = _tile_bounds(col, row)
    for offset in range(3, TILE_SIZE - 3):
        _fill_rect(
            frame,
            left + offset,
            top + TILE_SIZE - offset - 2,
            left + offset + 2,
            top + TILE_SIZE - offset,
            color,
        )


def _draw_plus(frame: np.ndarray, col: int, row: int, color: tuple[int, int, int]) -> None:
    cx, cy = _tile_center(col, row)
    _fill_rect(frame, cx - 2, cy - 7, cx + 2, cy + 7, color)
    _fill_rect(frame, cx - 7, cy - 2, cx + 7, cy + 2, color)


def event_names(info: dict[str, Any]) -> list[str]:
    names = [
        str(record.get("name"))
        for record in info.get("events", {}).get("records", [])
        if isinstance(record, dict) and record.get("name") is not None
    ]
    game = info.get("game", {})
    if game.get("world_completed", False) or info.get("terminal_reason") == "world_completed":
        names.append("world_completed")
    if game.get("dead", False) or info.get("terminal_reason") == "agent_dead":
        names.append("agent_dead")
    return names


def is_success(info: dict[str, Any], terminated: bool) -> bool:
    return bool(
        info.get("game", {}).get("world_completed", False)
        or info.get("terminal_reason") == "world_completed"
        or (terminated and info.get("reward", {}).get("terminated_reason") == "world_completed")
    )


def milestone_names(task_id: str) -> tuple[str, ...]:
    if task_id == "mathematical_logic/task_5":
        return TASK5_EVENTS
    return TASK_MILESTONES.get(task_id, ())


def build_episode_plan(
    *,
    task_ids: list[str],
    seed: int,
    num_envs: int,
    obs_variants: list[str],
    robustness_suite: bool,
) -> list[EpisodePlanEntry]:
    if num_envs < 1:
        raise ValueError("--num-envs must be >= 1")

    if not robustness_suite:
        return [
            EpisodePlanEntry(
                task_id=task_id,
                eval_stage=obs_variant,
                obs_variant=obs_variant,
                seed=seed + episode_index,
            )
            for obs_variant in obs_variants
            for task_id in task_ids
            for episode_index in range(num_envs)
        ]

    color_variants = ["grayscale", "dark", "bright", "high_contrast", "inverted"]
    plan: list[EpisodePlanEntry] = []
    for task_id in task_ids:
        original_count, spatial_count, color_count = _split_episode_counts(num_envs, (0.6, 0.3, 0.1))

        for episode_index in range(original_count):
            plan.append(
                EpisodePlanEntry(
                    task_id=task_id,
                    eval_stage="original",
                    obs_variant="default",
                    seed=seed + episode_index,
                )
            )
        for episode_index in range(spatial_count):
            plan.append(
                EpisodePlanEntry(
                    task_id=task_id,
                    eval_stage="spatial",
                    obs_variant="default",
                    seed=seed + episode_index,
                    map_variant=SPATIAL_MAP_VARIANTS[episode_index % len(SPATIAL_MAP_VARIANTS)],
                )
            )
        for episode_index in range(color_count):
            plan.append(
                EpisodePlanEntry(
                    task_id=task_id,
                    eval_stage="color",
                    obs_variant=color_variants[episode_index % len(color_variants)],
                    seed=seed + episode_index,
                )
            )
    return plan


def _split_episode_counts(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remaining = total - sum(counts)
    remainders = sorted(
        range(len(ratios)),
        key=lambda index: (raw_counts[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1
    return tuple(counts)


def run_episode(
    *,
    policy: Any,
    task_id: str,
    eval_stage: str,
    seed: int,
    max_steps: int | None,
    render_mode: str | None,
    obs_variant: str,
    action_repeat: int | None,
    map_variant: str = "default",
    info_mode: str = "safe",
    policy_task_id: str | None = None,
) -> EpisodeResult:
    env_kwargs: dict[str, Any] = {
        "observation_mode": "pixels",
        "render_mode": render_mode,
    }
    if max_steps is not None:
        env_kwargs["max_steps"] = max_steps
    if action_repeat is not None:
        env_kwargs["action_repeat"] = action_repeat
    if map_variant == "default":
        env = make_env(task_id=task_id, **env_kwargs)
    else:
        variant_map_path = materialize_spatial_map_variant(task_id, map_variant, seed=seed)
        env = make_env(
            task_id=task_id,
            map_path=variant_map_path,
            **env_kwargs,
        )
    reset_policy(policy)

    raw_obs, raw_info = env.reset(seed=seed)
    obs = apply_obs_variant(raw_obs, obs_variant, info=raw_info, env=env)
    event_counter: Counter[str] = Counter()
    total_reward = 0.0
    last_reward = 0.0
    terminated = False
    truncated = False
    steps = 0
    policy_info = build_policy_info(
        info_mode=info_mode,
        raw_info=raw_info,
        last_reward=last_reward,
        task_id=policy_task_id,
    )

    try:
        while not (terminated or truncated):
            action = call_policy(policy, obs, policy_info)
            if not env.action_space.contains(action):
                raise ValueError(f"policy returned invalid action {action!r} for {task_id}")
            raw_obs, reward, terminated, truncated, raw_info = env.step(action)
            obs = apply_obs_variant(raw_obs, obs_variant, info=raw_info, env=env)
            steps += 1
            last_reward = float(reward)
            total_reward += last_reward
            event_counter.update(event_names(raw_info))
            policy_info = build_policy_info(
                info_mode=info_mode,
                raw_info=raw_info,
                last_reward=last_reward,
                task_id=policy_task_id,
            )
    finally:
        env.close()

    milestones = {
        name: event_counter.get(name, 0) > 0
        for name in milestone_names(task_id)
    }
    return EpisodeResult(
        task_id=task_id,
        eval_stage=eval_stage,
        obs_variant=obs_variant,
        seed=seed,
        steps=steps,
        total_reward=total_reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
        success=is_success(raw_info, terminated),
        terminal_reason=raw_info.get("terminal_reason"),
        event_counts=dict(sorted(event_counter.items())),
        milestones=milestones,
        map_variant=map_variant,
    )


def materialize_spatial_map_variant(task_id: str, variant: str, *, seed: int) -> Path:
    if task_id not in SPATIAL_TASKS:
        allowed_tasks = ", ".join(sorted(SPATIAL_TASKS))
        raise ValueError(f"spatial map variants are only defined for {allowed_tasks}, got {task_id!r}")
    if variant not in SPATIAL_MAP_VARIANTS:
        allowed = ", ".join(SPATIAL_MAP_VARIANTS)
        raise ValueError(f"unknown spatial map variant {variant!r}, allowed: {allowed}")

    source_map = load_map(map_id=task_id)
    temp_root = Path(tempfile.mkdtemp(prefix="nesylink_spatial_"))
    del seed

    if source_map.name == "dungeon.json":
        target_task_dir = temp_root / source_map.parent.name
        shutil.copytree(source_map.parent, target_task_dir)
        target_map = target_task_dir / "dungeon.json"
        _patch_spatial_dungeon(task_id, variant, target_task_dir)
        return target_map

    target_map = temp_root / source_map.name
    payload = _read_json_file(source_map)
    _patch_spatial_room(task_id, variant, payload)
    _write_json_file(target_map, payload)
    return target_map


def _patch_spatial_dungeon(task_id: str, variant: str, task_dir: Path) -> None:
    if task_id == "mathematical_logic/task_3":
        patches = _task3_spatial_patch(variant)
    elif task_id == "mathematical_logic/task_4":
        patches = _task4_spatial_patch(variant)
    elif task_id == "mathematical_logic/task_5":
        patches = _task5_spatial_patch(variant)
    else:
        raise ValueError(f"dungeon spatial variants are not defined for {task_id!r}")
    for room_file, room_patch in patches.items():
        path = task_dir / "rooms" / room_file
        payload = _read_json_file(path)
        _apply_room_patch(payload, room_patch)
        _write_json_file(path, payload)


def _patch_spatial_room(task_id: str, variant: str, payload: dict[str, Any]) -> None:
    if task_id == "mathematical_logic/task_1":
        room_patch = _task1_spatial_patch(variant)
    elif task_id == "mathematical_logic/task_2":
        room_patch = _task2_spatial_patch(variant)
    else:
        raise ValueError(f"single-room spatial variants are not defined for {task_id!r}")
    _apply_room_patch(payload, room_patch)


def _task1_spatial_patch(variant: str) -> dict[str, Any]:
    patches = {
        "spatial_a": {
            "spawns": {"default": [6, 6], "from_south": [6, 6]},
            "objects": {"chest_key": [2, 3]},
            "layout": [
                "..........",
                "..........",
                "#...######",
                "..........",
                "..#.......",
                "######....",
                "..........",
                "..........",
            ],
        },
        "spatial_b": {
            "spawns": {"default": [3, 6], "from_south": [3, 6]},
            "objects": {"chest_key": [7, 4]},
            "layout": [
                "..........",
                "....#.....",
                "##...#####",
                "..........",
                "..........",
                "######....",
                "..........",
                "..........",
            ],
        },
        "spatial_c": {
            "spawns": {"default": [8, 6], "from_south": [8, 6]},
            "objects": {"chest_key": [3, 1]},
            "layout": [
                "..........",
                "..........",
                "###...####",
                "..........",
                ".....#....",
                "######....",
                "..........",
                "..........",
            ],
        },
    }
    return patches[variant]


def _task2_spatial_patch(variant: str) -> dict[str, Any]:
    patches = {
        "spatial_a": {
            "spawns": {"default": [7, 4], "from_east": [8, 4]},
            "objects": {"chest_key": [3, 3], "monster_chaser_left": [4, 2]},
        },
        "spatial_b": {
            "spawns": {"default": [6, 2], "from_east": [8, 4]},
            "objects": {"chest_key": [2, 5], "monster_chaser_left": [5, 4]},
        },
        "spatial_c": {
            "spawns": {"default": [8, 5], "from_east": [8, 4]},
            "objects": {"chest_key": [4, 3], "monster_chaser_left": [2, 4]},
        },
    }
    return patches[variant]


def _task3_spatial_patch(variant: str) -> dict[str, dict[str, Any]]:
    patches = {
        "spatial_a": {
            "start_room.json": {
                "spawns": {"default": [5, 5], "from_west": [1, 5], "from_east": [8, 5]},
                "objects": {"start_hint": [4, 2]},
            },
            "monster_hall.json": {
                "spawns": {"default": [8, 5], "from_east": [8, 5], "from_west": [1, 5]},
                "objects": {"hall_chaser": [4, 2]},
            },
            "key_room.json": {
                "spawns": {"default": [8, 5], "from_east": [8, 5]},
                "objects": {"return_key_chest": [4, 4]},
            },
        },
        "spatial_b": {
            "start_room.json": {
                "spawns": {"default": [3, 3], "from_west": [1, 3], "from_east": [8, 3]},
                "objects": {"start_hint": [5, 1]},
            },
            "monster_hall.json": {
                "spawns": {"default": [8, 3], "from_east": [8, 3], "from_west": [1, 3]},
                "objects": {"hall_chaser": [6, 5]},
            },
            "key_room.json": {
                "spawns": {"default": [8, 3], "from_east": [8, 3]},
                "objects": {"return_key_chest": [6, 2]},
            },
        },
        "spatial_c": {
            "start_room.json": {
                "spawns": {"default": [6, 4], "from_west": [1, 4], "from_east": [8, 4]},
                "objects": {"start_hint": [3, 1]},
            },
            "monster_hall.json": {
                "spawns": {"default": [8, 4], "from_east": [8, 4], "from_west": [1, 4]},
                "objects": {"hall_chaser": [4, 4]},
            },
            "key_room.json": {
                "spawns": {"default": [8, 4], "from_east": [8, 4]},
                "objects": {"return_key_chest": [3, 5]},
            },
        },
    }
    return patches[variant]


def _task4_spatial_patch(variant: str) -> dict[str, dict[str, Any]]:
    patches = {
        "spatial_a": {
            "west.json": {"objects": {"bridge_switch": [5, 3]}},
            "east.json": {"objects": {"sword_chest": [6, 3]}},
            "north.json": {"objects": {"key_chest": [5, 2]}},
            "south.json": {"objects": {"guardian": [5, 4]}},
            "center.json": {"objects": {"final_chest": [5, 4]}},
        },
        "spatial_b": {
            "west.json": {"objects": {"bridge_switch": [3, 4]}},
            "east.json": {"objects": {"sword_chest": [4, 2]}},
            "north.json": {"objects": {"key_chest": [3, 3]}},
            "south.json": {"objects": {"guardian": [3, 4]}},
            "center.json": {"objects": {"final_chest": [3, 3]}},
        },
        "spatial_c": {
            "west.json": {"objects": {"bridge_switch": [6, 5]}},
            "east.json": {"objects": {"sword_chest": [7, 5]}},
            "north.json": {"objects": {"key_chest": [6, 4]}},
            "south.json": {"objects": {"guardian": [6, 5]}},
            "center.json": {"objects": {"final_chest": [6, 4]}},
        },
    }
    return patches[variant]


def _task5_spatial_patch(variant: str) -> dict[str, dict[str, Any]]:
    patches = {
        "spatial_a": {
            "room_0_0.json": {
                "objects": {"chest_1": [3, 2], "npc_1": [8, 6], "button_1": [2, 5], "monster_1": [7, 3]},
            },
            "room_1_0.json": {
                "objects": {"chest_1": [7, 2], "npc_1": [6, 6], "monster_1": [8, 5]},
            },
            "room_0_1.json": {
                "objects": {"chest_1": [8, 4], "npc_1": [1, 2], "monster_1": [6, 5], "trap_1": [1, 4]},
            },
            "room_-1_0.json": {
                "objects": {"chest_1": [2, 5], "npc_1": [8, 6], "monster_1": [2, 3], "monster_2": [6, 2]},
            },
        },
        "spatial_b": {
            "room_0_0.json": {
                "objects": {"chest_1": [4, 1], "npc_1": [8, 5], "button_1": [1, 6], "monster_1": [7, 2]},
            },
            "room_1_0.json": {
                "objects": {"chest_1": [6, 1], "npc_1": [8, 6], "monster_1": [7, 6]},
            },
            "room_0_1.json": {
                "objects": {"chest_1": [7, 5], "npc_1": [3, 1], "monster_1": [6, 4], "trap_1": [2, 5]},
            },
            "room_-1_0.json": {
                "objects": {"chest_1": [3, 6], "npc_1": [7, 5], "monster_1": [3, 4], "monster_2": [6, 4]},
            },
        },
        "spatial_c": {
            "room_0_0.json": {
                "objects": {"chest_1": [2, 2], "npc_1": [8, 4], "button_1": [3, 6], "monster_1": [7, 5]},
            },
            "room_1_0.json": {
                "objects": {"chest_1": [8, 1], "npc_1": [6, 5], "monster_1": [8, 6]},
            },
            "room_0_1.json": {
                "objects": {"chest_1": [8, 6], "npc_1": [1, 1], "monster_1": [7, 6], "trap_1": [1, 6]},
            },
            "room_-1_0.json": {
                "objects": {"chest_1": [1, 6], "npc_1": [8, 5], "monster_1": [2, 5], "monster_2": [7, 3]},
            },
        },
    }
    return patches[variant]


def _apply_room_patch(payload: dict[str, Any], patch: dict[str, Any]) -> None:
    if "layout" in patch:
        payload["layout"] = patch["layout"]
    for spawn_name, pos in patch.get("spawns", {}).items():
        payload.setdefault("spawns", {})[spawn_name] = pos
    object_positions = patch.get("objects", {})
    if object_positions:
        for entry in payload.get("objects", []):
            object_id = entry.get("id")
            if object_id in object_positions:
                entry["pos"] = object_positions[object_id]


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    by_task: dict[tuple[str, str], list[EpisodeResult]] = {}
    for result in results:
        by_task.setdefault((result.task_id, result.eval_stage), []).append(result)

    summary: dict[str, Any] = {}
    sorted_groups = sorted(
        by_task.items(),
        key=lambda item: (item[0][0], EVAL_STAGE_ORDER.get(item[0][1], 100), item[0][1]),
    )
    for (task_id, eval_stage), task_results in sorted_groups:
        episodes = len(task_results)
        event_totals: Counter[str] = Counter()
        milestone_successes: Counter[str] = Counter()
        progress_successes: Counter[str] = Counter()
        obs_variant_counts: Counter[str] = Counter()
        map_variant_counts: Counter[str] = Counter()
        for result in task_results:
            obs_variant_counts.update([result.obs_variant])
            map_variant_counts.update([result.map_variant])
            event_totals.update(result.event_counts)
            for name, achieved in result.milestones.items():
                if achieved:
                    milestone_successes[name] += 1
            for name in PROGRESS_EVENTS:
                if result.event_counts.get(name, 0) > 0:
                    progress_successes[name] += 1
        progress_rates = {
            name: progress_successes[name] / episodes
            for name in PROGRESS_EVENTS
            if progress_successes[name] > 0
        }
        summary[f"{task_id} [{eval_stage}]"] = {
            "task_id": task_id,
            "eval_stage": eval_stage,
            "obs_variant_counts": dict(sorted(obs_variant_counts.items())),
            "map_variant_counts": dict(sorted(map_variant_counts.items())),
            "episodes": episodes,
            "success_rate": sum(result.success for result in task_results) / episodes,
            "avg_steps": sum(result.steps for result in task_results) / episodes,
            "avg_reward": sum(result.total_reward for result in task_results) / episodes,
            "milestone_rates": {
                name: milestone_successes[name] / episodes
                for name in milestone_names(task_id)
            },
            "progress_rates": progress_rates,
            "event_totals": dict(sorted(event_totals.items())),
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    for label, stats in summary.items():
        task_id = stats["task_id"]
        print(f"\n{label}")
        print(f"  episodes:     {stats['episodes']}")
        print(f"  success_rate: {stats['success_rate']:.3f}")
        print(f"  avg_steps:    {stats['avg_steps']:.1f}")
        print(f"  avg_reward:   {stats['avg_reward']:.3f}")
        print(f"  variants:     {stats['obs_variant_counts']}")
        print(f"  map_variants: {stats['map_variant_counts']}")
        if stats["milestone_rates"]:
            print("  milestones:")
            for name, rate in stats["milestone_rates"].items():
                print(f"    {name}: {rate:.3f}")
        if stats["progress_rates"]:
            print("  progress:")
            for name, rate in stats["progress_rates"].items():
                print(f"    {name}: {rate:.3f}")
        if task_id == "mathematical_logic/task_5":
            print("  game_event_totals:")
            for name in TASK5_EVENTS:
                print(f"    {name}: {stats['event_totals'].get(name, 0)}")


def parse_args() -> argparse.Namespace:
    task_ids = [task.task_id for task in list_tasks()]
    parser = argparse.ArgumentParser(description="Evaluate a NesyLink policy submission.")
    parser.add_argument(
        "--policy",
        default=None,
        help="Shared policy module or file, optionally with :attribute.",
    )
    parser.add_argument(
        "--task-policy",
        action="append",
        default=[],
        help=(
            "Task-specific policy in TASK_ID=POLICY_SPEC format. Can be repeated. "
            "Overrides --policy for that task."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=task_ids,
        help="Task IDs to evaluate.",
    )
    parser.add_argument("--num-envs", type=int, default=100, help="Number of episodes/env instances per task.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed. Episode seed is seed + episode index.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override task max_steps during evaluation.")
    parser.add_argument("--action-repeat", type=int, default=None, help="Override task action_repeat during evaluation.")
    parser.add_argument("--render-mode", default=None, choices=["rgb_array"], help="Optional render mode.")
    parser.add_argument(
        "--info-mode",
        choices=["safe", "full"],
        default="safe",
        help="Information passed to the policy. Use safe for official evaluation; full is for local debugging.",
    )
    parser.add_argument(
        "--obs-variants",
        nargs="+",
        default=["default"],
        choices=OBS_VARIANTS,
        help="Pixel-observation variants to apply only inside this evaluator.",
    )
    parser.add_argument(
        "--robustness-suite",
        action="store_true",
        help=(
            "Run a proportional robustness suite: 60%% original episodes, "
            "30%% spatial map perturbation episodes, and 10%% color-shift episodes."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for detailed JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.action_repeat is not None and args.action_repeat < 1:
        raise ValueError("--action-repeat must be >= 1")

    policy_bindings_by_task = resolve_policies(
        default_policy_spec=args.policy,
        task_policy_specs=args.task_policy,
        task_ids=args.tasks,
    )
    results: list[EpisodeResult] = []
    episode_plan = build_episode_plan(
        task_ids=args.tasks,
        seed=args.seed,
        num_envs=args.num_envs,
        obs_variants=args.obs_variants,
        robustness_suite=args.robustness_suite,
    )
    for entry in episode_plan:
        policy_binding = policy_bindings_by_task[entry.task_id]
        result = run_episode(
            policy=policy_binding.policy,
            task_id=entry.task_id,
            eval_stage=entry.eval_stage,
            seed=entry.seed,
            max_steps=args.max_steps,
            render_mode=args.render_mode,
            obs_variant=entry.obs_variant,
            action_repeat=args.action_repeat,
            map_variant=entry.map_variant,
            info_mode=args.info_mode,
            policy_task_id=entry.task_id if policy_binding.receives_task_id else None,
        )
        results.append(result)
        print(
            f"{entry.task_id} stage={entry.eval_stage} obs_variant={entry.obs_variant} "
            f"map_variant={entry.map_variant} "
            f"seed={entry.seed} success={result.success} steps={result.steps} "
            f"reward={result.total_reward:.3f}"
        )

    summary = summarize(results)
    print_summary(summary)

    if args.json_out is not None:
        payload = {
            "summary": summary,
            "episodes": [asdict(result) for result in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
