"""Record one JSON line for every submitted-agent decision.

The trace uses the same safe policy interface as ``evaluate_policy.py``. It
adds observability by wrapping methods on the loaded policy object; it does not
change the observation or action passed to the policy.

Example:
    python utils/trace_agent.py \
        --policy submissions/student_agent.py \
        --task mathematical_logic/task_5 \
        --map-variant spatial_a \
        --out outputs/debug/task5_spatial_a_trace.jsonl
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
from nesylink.core.constants import ACTION_LABELS
from nesylink.env import make_env
from nesylink.tasks import list_tasks


def _position(value: Any) -> list[int] | None:
    if value is None:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, IndexError, ValueError):
        return None


def _positions(values: Any) -> list[list[int]]:
    if values is None:
        return []
    result = [_position(value) for value in values]
    return sorted(value for value in result if value is not None)


def _jsonable(value: Any) -> Any:
    """Convert small policy/debug values into stable JSON-compatible data."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _entity_snapshot(entity: Any) -> dict[str, Any] | None:
    if entity is None:
        return None
    return {
        "tile": _position(getattr(entity, "tile", None)),
        "center_px": _position(getattr(entity, "center_px", None)),
        "bbox_px": [int(value) for value in getattr(entity, "bbox_px", ())]
        if getattr(entity, "bbox_px", None) is not None
        else None,
        "confidence": getattr(entity, "confidence", None),
    }


def _state_snapshot(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    result: dict[str, Any] = {
        "player": _position(getattr(state, "player", None)),
        "walls": _positions(getattr(state, "walls", ())),
        "monsters": _positions(getattr(state, "monsters", ())),
        "chests": _positions(getattr(state, "chests", ())),
        "traps": _positions(getattr(state, "traps", ())),
        "exits": _positions(getattr(state, "exits", ())),
        "gaps": _positions(getattr(state, "gaps", ())),
        "bridges": _positions(getattr(state, "bridges", ())),
        "switches": _positions(getattr(state, "switches", ())),
        "buttons": _positions(getattr(state, "buttons", ())),
        "exit_types": _jsonable(getattr(state, "exit_types", {}) or {}),
        "player_entity": _entity_snapshot(getattr(state, "player_entity", None)),
    }
    return result


def _objective_snapshot(objective: Any) -> dict[str, Any] | None:
    if objective is None:
        return None
    return {
        "kind": getattr(objective, "kind", None),
        "targets": _positions(getattr(objective, "targets", ())),
        "side": getattr(objective, "side", None),
        "interaction_kind": getattr(objective, "interaction_kind", None),
        "search": getattr(objective, "search", None),
        "safe": bool(getattr(objective, "safe", False)),
        "constrain_bridge": bool(getattr(objective, "constrain_bridge", True)),
        "combat_required": bool(getattr(objective, "combat_required", False)),
        "contact_only": bool(getattr(objective, "contact_only", False)),
    }


def _memory_snapshot(policy: Any) -> dict[str, Any]:
    room_signature = getattr(policy, "current_room_signature", None)
    room_memory = getattr(policy, "room_memory", {}) or {}
    current = room_memory.get(room_signature)
    return {
        "current_room_signature": room_signature,
        "current_room": {
            "visit_count": getattr(current, "visit_count", None),
            "known_chests": _positions(getattr(current, "known_chests", ())),
            "opened_chests": _positions(getattr(current, "opened_chests", ())),
            "pressed_buttons": _positions(getattr(current, "pressed_buttons", ())),
            "has_monster": bool(getattr(current, "has_monster", False)),
            "has_button": bool(getattr(current, "has_button", False)),
            "interaction_done": bool(getattr(current, "interaction_done", False)),
        }
        if current is not None
        else None,
        "inventory_revision": getattr(policy, "inventory_revision", None),
        "hp_estimate": getattr(policy, "hp_estimate", None),
        "remembered_blockers": _positions(getattr(policy, "remembered_blockers", ())),
        "attempted_exit": repr(getattr(policy, "attempted_exit", None)),
    }


def _install_hooks(policy: Any, pending: dict[str, Any]) -> bool:
    """Capture internal planner stages when the policy exposes them."""

    if not hasattr(policy, "act") or not callable(policy.act):
        return False

    original_act = policy.act

    def wrapped_act(self: Any, obs: Any, info: Any) -> int:
        pending.clear()
        pending["before_committed_action"] = getattr(self, "committed_action", None)
        pending["before_committed_ticks"] = getattr(self, "committed_ticks_remaining", 0)
        action = original_act(obs, info)
        pending["action"] = int(action)
        state = pending.get("state")
        objective = pending.get("objective")
        stages = pending.get("stages", [])

        if pending.get("shield_action") is not None:
            reason = "emergency_shield"
        elif pending.get("recovery_action") is not None:
            reason = "stuck_recovery"
        elif pending.get("interaction_action") is not None:
            reason = "interaction_or_interaction_confirmation"
        elif pending.get("route_monsters"):
            reason = "route_monster_is_adjacent_or_direct_route_is_being_checked"
        elif objective is not None:
            planner = pending.get("planner", "planner")
            kind = getattr(objective, "kind", "unknown")
            side = getattr(objective, "side", None)
            suffix = f"({side})" if side is not None else ""
            reason = f"{planner}_objective:{kind}{suffix}"
        elif pending.get("before_committed_action") is not None:
            reason = "continue_committed_move"
        elif int(action) == 0:
            reason = "wait_or_no_path"
        else:
            reason = "controller_fallback"

        pending["reason"] = reason
        pending["stages"] = stages
        pending["state_after"] = _state_snapshot(state)
        pending["raw_state_after"] = _state_snapshot(getattr(self, "last_raw_state", None))
        pending["memory"] = _memory_snapshot(self)
        return int(action)

    policy.act = MethodType(wrapped_act, policy)

    def wrap(name: str, *, stage: str | None = None, result_key: str | None = None):
        original = getattr(policy, name, None)
        if original is None or not callable(original):
            return

        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if stage is not None:
                pending.setdefault("stages", []).append(stage)
            if result_key is not None:
                pending[result_key] = result
            return result

        setattr(policy, name, MethodType(wrapped, policy))

    def wrap_state(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_extract(*args, **kwargs)
        pending["state"] = result
        pending["raw_state"] = getattr(self, "last_raw_state", None)
        pending.setdefault("stages", []).append("perception")
        return result

    original_extract = getattr(policy, "_extract_state", None)
    if callable(original_extract):
        policy._extract_state = MethodType(wrap_state, policy)

    wrap("_choose_global_objective", stage="global_planner", result_key="objective")
    wrap("_choose_local_objective", stage="local_planner", result_key="objective")
    wrap("_route_combat_targets", stage="route_combat_check", result_key="route_monsters")
    wrap("_execute_objective", stage="execute_objective", result_key="executed_action")
    wrap("_advance_interaction", stage="interaction_gate", result_key="interaction_action")
    wrap("_emergency_shield_action", stage="shield_check", result_key="shield_action")
    wrap("_stuck_recovery_action", stage="stuck_check", result_key="recovery_action")
    wrap("_consume_recovery_forced_action", stage="forced_recovery_check", result_key="forced_action")
    return True


def _parse_args() -> argparse.Namespace:
    task_ids = [task.task_id for task in list_tasks()]
    parser = argparse.ArgumentParser(description="Trace each decision of a NesyLink policy.")
    parser.add_argument("--policy", required=True, help="Policy module or Python file.")
    parser.add_argument("--task", default="mathematical_logic/task_5", choices=task_ids)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map-variant", default="default", choices=("default", *SPATIAL_MAP_VARIANTS))
    parser.add_argument("--obs-variant", default="default", choices=OBS_VARIANTS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--print-interesting",
        action="store_true",
        help="Print attacks, shields, recovery, waits, and non-committed decisions.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    policy = load_policy(args.policy)
    pending: dict[str, Any] = {}
    hooked = _install_hooks(policy, pending)
    reset_policy(policy)

    env_kwargs: dict[str, Any] = {
        "observation_mode": "pixels",
        "render_mode": None,
    }
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.map_variant == "default":
        env = make_env(task_id=args.task, **env_kwargs)
    else:
        map_path = materialize_spatial_map_variant(args.task, args.map_variant, seed=args.seed)
        env = make_env(task_id=args.task, map_path=map_path, **env_kwargs)

    records: list[dict[str, Any]] = []
    total_reward = 0.0
    last_reward = 0.0
    terminated = False
    truncated = False
    raw_info: dict[str, Any] = {}
    obs: np.ndarray

    try:
        raw_obs, raw_info = env.reset(seed=args.seed)
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
                "hooked": hooked,
                "reason": pending.get("reason", "policy_call"),
                "stages": list(pending.get("stages", [])),
                "objective": _objective_snapshot(pending.get("objective")),
                "route_monsters": _positions(pending.get("route_monsters", ())),
                "state": pending.get("state_after"),
                "raw_state": pending.get("raw_state_after"),
                "memory": pending.get("memory", {}),
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

    if args.print_interesting:
        for record in records:
            interesting = (
                record["action_name"] in {"BUTTON_A", "BUTTON_B"}
                or record["reason"] != "continue_committed_move"
                or record["controller"].get("stuck_ticks", 0) >= 10
            )
            if interesting:
                print(
                    f"step={record['step']:04d} action={record['action_name']:<8} "
                    f"reason={record['reason']} objective={record['objective']} "
                    f"player={record['state'].get('player') if record['state'] else None} "
                    f"reward={record.get('reward_after_action', 0.0):.3f} events={record['events']}"
                )

    print(
        json.dumps(
            {
                "out": str(args.out),
                "task": args.task,
                "map_variant": args.map_variant,
                "obs_variant": args.obs_variant,
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
