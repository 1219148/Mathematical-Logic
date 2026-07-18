"""Run the submitted policy with an environment-truth perception adapter.

This is an explicit debugging upper bound, not an evaluation mode.  The
policy still receives the normal observation and safe inventory information,
but ``Policy._extract_state`` is replaced by a state built from the local
runtime.  The adapter is intentionally kept outside ``submissions`` so it
cannot be enabled accidentally during official evaluation.

Example:
    .venv/bin/python utils/oracle_debug.py \
        --policy submissions/student_agent.py \
        --task mathematical_logic/task_5 \
        --map-variant spatial_a \
        --out outputs/debug/oracle_task5_spatial_a.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nesylink.core.constants import ACTION_LABELS, TILE_SIZE
from nesylink.core.observation import room_observation
from nesylink.core.state import entity_center_px, entity_rect, tile_from_position_px, tile_center_px
from nesylink.shared import EntityState, SymbolicState
from nesylink.env import make_env
from nesylink.tasks import list_tasks
from utils.evaluate_policy import (
    OBS_VARIANTS,
    SPATIAL_MAP_VARIANTS,
    apply_obs_variant,
    build_safe_info,
    call_policy,
    event_names,
    is_success,
    load_policy,
    materialize_spatial_map_variant,
    reset_policy,
)
from utils.trace_agent import (
    _install_hooks,
    _jsonable,
    _memory_snapshot,
    _objective_snapshot,
    _position,
    _positions,
    _state_snapshot,
)


def _tile_entity(
    tile: tuple[int, int],
    *,
    kind: str,
    entity_type: str = "",
    hp: int | None = None,
) -> EntityState:
    left = float(tile[0] * TILE_SIZE)
    top = float(tile[1] * TILE_SIZE)
    return EntityState(
        tile=tile,
        center_px=tile_center_px(tile),
        bbox_px=(left, top, left + TILE_SIZE, top + TILE_SIZE),
        kind=kind,
        entity_type=entity_type,
        hp=hp,
        confidence=1.0,
    )


def _oracle_state(env: Any) -> SymbolicState:
    """Build the same symbolic contract as perception, using runtime truth."""

    runtime = env.engine.runtime
    room = runtime.room
    player = runtime.player
    player_tile = tile_from_position_px(player.position_px, player.size_px)

    chests = frozenset(
        chest.pos
        for chest in room.chests.values()
        if chest.is_visible and not chest.is_open
    )
    traps = frozenset(
        trap.pos
        for trap in room.traps.values()
        if trap.is_active and room.dynamic_tiles.get(trap.pos) != "bridge"
    )
    exits = frozenset(tile for exit_config in room.exits for tile in exit_config.tiles)
    exit_types = {
        tile: exit_config.exit_type
        for exit_config in room.exits
        for tile in exit_config.tiles
    }
    monsters = frozenset(monster.tile_pos for monster in room.monsters.values())
    monster_types = {
        monster.tile_pos: monster.monster_type for monster in room.monsters.values()
    }
    buttons = frozenset(button.pos for button in room.buttons.values())
    button_pressed = {button.pos: bool(button.is_pressed) for button in room.buttons.values()}
    switches = frozenset(switch.pos for switch in room.switches.values())
    gaps = frozenset(pos for pos, kind in room.dynamic_tiles.items() if kind == "gap")
    bridges = frozenset(pos for pos, kind in room.dynamic_tiles.items() if kind == "bridge")
    dynamic_objects = {
        object_id: room.dynamic_states.get(object_id, dynamic_object.initial_state)
        for object_id, dynamic_object in room.dynamic_objects.items()
    }

    player_entity = EntityState(
        tile=player_tile,
        center_px=entity_center_px(player.position_px, player.size_px),
        bbox_px=entity_rect(player.position_px, player.size_px),
        kind="player",
        confidence=1.0,
    )
    monster_entities = tuple(
        EntityState(
            tile=monster.tile_pos,
            center_px=entity_center_px(monster.position_px, monster.size_px),
            bbox_px=entity_rect(monster.position_px, monster.size_px),
            kind="monster",
            entity_type=monster.monster_type,
            hp=monster.hp,
            confidence=1.0,
        )
        for monster in room.monsters.values()
    )

    return SymbolicState(
        player=player_tile,
        walls=frozenset(room.walls),
        health=int(player.health),
        keys=int(player.keys),
        monsters=monsters,
        monster_types=monster_types,
        chests=chests,
        traps=traps,
        exits=exits,
        exit_types=exit_types,
        gold=int(player.gold),
        items=tuple(player.items),
        buttons=buttons,
        button_pressed=button_pressed,
        gaps=gaps,
        bridges=bridges,
        switches=switches,
        dynamic_objects=dynamic_objects,
        player_entity=player_entity,
        monster_entities=monster_entities,
        static_grid=tuple(tuple(int(value) for value in row) for row in room_observation(room, player)),
        room_id=room.room_id,
    )


def _install_oracle_perception(policy: Any, env: Any) -> None:
    original_extract = getattr(policy, "_extract_state", None)
    if not callable(original_extract):
        raise TypeError("oracle debug requires a policy exposing _extract_state")

    def extract_from_runtime(self: Any, *args: Any, **kwargs: Any) -> SymbolicState:
        del args, kwargs
        state = _oracle_state(env)
        self.last_raw_state = state
        return state

    policy._extract_state = MethodType(extract_from_runtime, policy)


def _parse_args() -> argparse.Namespace:
    task_ids = [task.task_id for task in list_tasks()]
    parser = argparse.ArgumentParser(
        description="Debug a policy with environment-truth symbolic perception."
    )
    parser.add_argument("--policy", required=True, help="Policy module or Python file.")
    parser.add_argument("--task", default="mathematical_logic/task_5", choices=task_ids)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--map-variant",
        default="default",
        choices=("default", *SPATIAL_MAP_VARIANTS),
    )
    parser.add_argument("--obs-variant", default="default", choices=OBS_VARIANTS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env_kwargs: dict[str, Any] = {"observation_mode": "pixels", "render_mode": None}
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.map_variant == "default":
        env = make_env(task_id=args.task, **env_kwargs)
    else:
        map_path = materialize_spatial_map_variant(args.task, args.map_variant, seed=args.seed)
        env = make_env(task_id=args.task, map_path=map_path, **env_kwargs)

    policy = load_policy(args.policy)
    pending: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    total_reward = 0.0
    last_reward = 0.0
    terminated = False
    truncated = False
    raw_info: dict[str, Any] = {}

    try:
        raw_obs, raw_info = env.reset(seed=args.seed)
        _install_oracle_perception(policy, env)
        hooked = _install_hooks(policy, pending)
        reset_policy(policy)
        obs = apply_obs_variant(raw_obs, args.obs_variant, info=raw_info, env=env)
        policy_info = build_safe_info(
            raw_info=raw_info,
            last_reward=last_reward,
            task_id=None,
        )

        while not (terminated or truncated):
            action = call_policy(policy, obs, policy_info)
            if not env.action_space.contains(action):
                raise ValueError(f"policy returned invalid action {action!r}")
            record = {
                "step": len(records) + 1,
                "action": int(action),
                "action_name": ACTION_LABELS.get(int(action), str(action)),
                "last_reward": float(last_reward),
                "policy_info": policy_info,
                "oracle_perception": True,
                "hooked": hooked,
                "reason": pending.get("reason", "policy_call"),
                "stages": list(pending.get("stages", [])),
                "objective": _objective_snapshot(pending.get("objective")),
                "route_monsters": _positions(pending.get("route_monsters", ())),
                "state": pending.get("state_after"),
                "memory": pending.get("memory", _memory_snapshot(policy)),
                "controller": {
                    "committed_action": getattr(policy, "committed_action", None),
                    "committed_ticks_remaining": getattr(policy, "committed_ticks_remaining", None),
                    "committed_target_tile": _position(getattr(policy, "committed_target_tile", None)),
                    "stuck_ticks": getattr(policy, "stuck_ticks", None),
                    "pending_interaction": repr(getattr(policy, "pending_interaction", None)),
                    "awaiting_interaction": repr(getattr(policy, "awaiting_interaction", None)),
                    "shield_cooldown": getattr(policy, "shield_cooldown", None),
                },
            }
            raw_obs, reward, terminated, truncated, raw_info = env.step(action)
            record["reward_after_action"] = float(reward)
            record["events"] = event_names(raw_info)
            record["runtime"] = {
                "room_id": getattr(env.engine.runtime.room, "room_id", None),
                "player_health": getattr(env.engine.runtime.player, "health", None),
            }
            record["terminal_reason"] = raw_info.get("terminal_reason")
            record["terminated"] = bool(terminated)
            record["truncated"] = bool(truncated)
            records.append(record)

            total_reward += float(reward)
            last_reward = float(reward)
            obs = apply_obs_variant(raw_obs, args.obs_variant, info=raw_info, env=env)
            policy_info = build_safe_info(
                raw_info=raw_info,
                last_reward=last_reward,
                task_id=None,
            )
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "out": str(args.out),
                "task": args.task,
                "map_variant": args.map_variant,
                "obs_variant": args.obs_variant,
                "oracle_perception": True,
                "steps": len(records),
                "total_reward": total_reward,
                "success": is_success(raw_info, terminated),
                "terminal_reason": raw_info.get("terminal_reason"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
