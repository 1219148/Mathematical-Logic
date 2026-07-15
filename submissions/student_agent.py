from __future__ import annotations

import os
from heapq import heappop, heappush
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable, Literal

from nesylink.core.observation import TILE_NPC
from nesylink.core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
    TILE_SIZE,
)
from nesylink.shared import SymbolicState

from submissions.temporal_filter import TemporalSymbolicFilter


Position = tuple[int, int]
Side = Literal["up", "down", "left", "right"]
ObjectiveKind = Literal["fight", "interact", "shield", "navigate", "go_exit", "idle"]
InteractionKind = Literal["chest", "switch", "button", "monster"]
SearchMode = Literal["bfs", "astar"]
TILE_MOVE_TICKS = 16
EXIT_MOVE_TICKS = 24
EXIT_ALIGNMENT_TICKS = 2
EXIT_ALIGNMENT_TOLERANCE_PX = 1.0
BRIDGE_SCAN_VIEWPOINTS = 1
INTERACTION_SUCCESS_REWARD = 1.5
INTERACTION_VISUAL_CONFIRM_TICKS = 5


ACTION_TO_DELTA: dict[int, Position] = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}
SIDE_TO_ACTION: dict[Side, int] = {
    "up": ACTION_UP,
    "down": ACTION_DOWN,
    "left": ACTION_LEFT,
    "right": ACTION_RIGHT,
}
ACTION_TO_SIDE: dict[int, Side] = {action: side for side, action in SIDE_TO_ACTION.items()}
OPPOSITE_SIDE: dict[Side, Side] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
SIDE_ORDER: tuple[Side, ...] = ("up", "right", "down", "left")


@dataclass
class InteractionIntent:
    target: Position
    action_to_face: int
    kind: InteractionKind


@dataclass
class InteractionAttempt:
    intent: InteractionIntent
    inventory_revision: int
    missing_frames: int = 0


@dataclass(frozen=True)
class RoomFeatures:
    exit_sides: frozenset[Side]
    bridge_sides: frozenset[Side]
    has_bridge: bool
    has_switch: bool
    has_button: bool
    has_chest: bool
    has_monster: bool
    has_trap: bool


@dataclass
class RoomMemory:
    exits: set[Side] = field(default_factory=set)
    known_chests: set[Position] = field(default_factory=set)
    opened_chests: set[Position] = field(default_factory=set)
    pressed_buttons: set[Position] = field(default_factory=set)
    has_monster: bool = False
    has_button: bool = False
    interaction_done: bool = False
    visited_inventory_revisions: set[int] = field(default_factory=set)


@dataclass
class ExitMemory:
    status: str = "unknown"
    leads_to: str | None = None


@dataclass(frozen=True)
class Objective:
    kind: ObjectiveKind
    targets: frozenset[Position] = frozenset()
    side: Side | None = None
    interaction_kind: InteractionKind | None = None
    search: SearchMode = "bfs"
    safe: bool = False
    constrain_bridge: bool = True


@dataclass(frozen=True)
class PlannerConfig:
    monster_danger_cost: float = 2.0
    trap_neighbor_cost: float = 4.0
    unknown_exit_bonus: float = 40.0
    key_blocked_retry_bonus: float = 80.0
    blocked_exit_penalty: float = 50.0
    unfinished_room_bonus: float = 10.0
    completed_room_penalty: float = 15.0
    target_tile_stable_ticks: int = 12
    stuck_recovery_ticks: int = 24
    shield_distance: int = 2
    shield_cooldown_ticks: int = 18


DEFAULT_PLANNER_CONFIG = PlannerConfig()


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def planner_config_from_env() -> PlannerConfig:
    base = DEFAULT_PLANNER_CONFIG
    return PlannerConfig(
        monster_danger_cost=_env_float("NESYLINK_MONSTER_DANGER_COST", base.monster_danger_cost),
        trap_neighbor_cost=_env_float("NESYLINK_TRAP_NEIGHBOR_COST", base.trap_neighbor_cost),
        unknown_exit_bonus=_env_float("NESYLINK_UNKNOWN_EXIT_BONUS", base.unknown_exit_bonus),
        key_blocked_retry_bonus=_env_float("NESYLINK_KEY_BLOCKED_RETRY_BONUS", base.key_blocked_retry_bonus),
        blocked_exit_penalty=_env_float("NESYLINK_BLOCKED_EXIT_PENALTY", base.blocked_exit_penalty),
        unfinished_room_bonus=_env_float("NESYLINK_UNFINISHED_ROOM_BONUS", base.unfinished_room_bonus),
        completed_room_penalty=_env_float("NESYLINK_COMPLETED_ROOM_PENALTY", base.completed_room_penalty),
        target_tile_stable_ticks=_env_int("NESYLINK_TARGET_TILE_STABLE_TICKS", base.target_tile_stable_ticks),
        stuck_recovery_ticks=_env_int("NESYLINK_STUCK_RECOVERY_TICKS", base.stuck_recovery_ticks),
        shield_distance=_env_int("NESYLINK_SHIELD_DISTANCE", base.shield_distance),
        shield_cooldown_ticks=_env_int("NESYLINK_SHIELD_COOLDOWN_TICKS", base.shield_cooldown_ticks),
    )


def neighbors(pos: Position) -> Iterable[Position]:
    x, y = pos
    yield (x, y - 1)
    yield (x, y + 1)
    yield (x - 1, y)
    yield (x + 1, y)


def in_bounds(pos: Position, width: int = 10, height: int = 8) -> bool:
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def action_from_step(current: Position, nxt: Position) -> int:
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]
    for action, delta in ACTION_TO_DELTA.items():
        if delta == (dx, dy):
            return action
    raise ValueError(f"non-adjacent step: {current} -> {nxt}")


def adjacent_action(current: Position, target: Position) -> int | None:
    if manhattan(current, target) != 1:
        return None
    return action_from_step(current, target)


def blocked_tiles(
    state: SymbolicState,
    extra_blocked: set[Position] | None = None,
    extra_unblocked: set[Position] | None = None,
    *,
    constrain_bridge: bool = True,
) -> set[Position]:
    blocked = set(state.walls) | npc_tiles(state) | set(state.traps) | set(state.gaps) | set(state.chests) | set(state.monsters)
    if constrain_bridge and bridge_requires_constrained_walk(state):
        safe_tiles = set(state.bridges) | set(state.exits) | {state.player}
        blocked.update(
            (x, y)
            for x in range(10)
            for y in range(8)
            if (x, y) not in safe_tiles
        )
    if extra_blocked:
        blocked.update(extra_blocked)
    if extra_unblocked:
        blocked.difference_update(extra_unblocked)
    return blocked


def bridge_requires_constrained_walk(state: SymbolicState) -> bool:
    if not state.bridges:
        return False
    if any(neighbor in state.gaps for bridge in state.bridges for neighbor in neighbors(bridge)):
        return True
    return len(bridge_exit_sides(state)) >= 2


def bridge_exit_sides(state: SymbolicState) -> frozenset[Side]:
    sides: set[Side] = set()
    for bridge in state.bridges:
        for exit_tile in state.exits:
            if manhattan(bridge, exit_tile) != 1:
                continue
            if exit_tile[1] == 0:
                sides.add("up")
            elif exit_tile[1] == 7:
                sides.add("down")
            elif exit_tile[0] == 0:
                sides.add("left")
            elif exit_tile[0] == 9:
                sides.add("right")
    return frozenset(sides)


def npc_tiles(state: SymbolicState) -> set[Position]:
    return {
        (x, y)
        for y, row in enumerate(state.static_grid)
        for x, value in enumerate(row)
        if value == TILE_NPC
    }


def is_walkable(
    pos: Position,
    state: SymbolicState,
    extra_blocked: set[Position] | None = None,
    extra_unblocked: set[Position] | None = None,
    *,
    constrain_bridge: bool = True,
) -> bool:
    return in_bounds(pos) and pos not in blocked_tiles(
        state,
        extra_blocked,
        extra_unblocked,
        constrain_bridge=constrain_bridge,
    )


def adjacent_walkable_goals(
    state: SymbolicState,
    targets: set[Position],
    extra_blocked: set[Position] | None = None,
    *,
    constrain_bridge: bool = True,
) -> set[Position]:
    goals: set[Position] = set()
    for target in targets:
        for pos in neighbors(target):
            if is_walkable(pos, state, extra_blocked, constrain_bridge=constrain_bridge):
                goals.add(pos)
    return goals


def monster_goals(state: SymbolicState, extra_blocked: set[Position] | None = None) -> set[Position]:
    return adjacent_walkable_goals(state, set(state.monsters), extra_blocked)


def fallback_exit_goals(state: SymbolicState) -> set[Position]:
    if state.exits:
        return {pos for pos in state.exits if in_bounds(pos)}
    return {(x, 0) for x in range(10) if is_walkable((x, 0), state)}


def side_exits(state: SymbolicState, action: int) -> set[Position]:
    if action == ACTION_LEFT:
        return {pos for pos in state.exits if pos[0] == 0}
    if action == ACTION_RIGHT:
        return {pos for pos in state.exits if pos[0] == 9}
    if action == ACTION_UP:
        return {pos for pos in state.exits if pos[1] == 0}
    if action == ACTION_DOWN:
        return {pos for pos in state.exits if pos[1] == 7}
    return set()


def exit_alignment_action(
    state: SymbolicState,
    side: Side,
    exits: set[Position] | None = None,
) -> int | None:
    """Return a short perpendicular correction toward a visible exit lane."""
    entity = state.player_entity
    target_exits = side_exits(state, SIDE_TO_ACTION[side]) if exits is None else exits
    if entity is None or not target_exits:
        return None

    vertical_exit = side in {"up", "down"}
    player_axis = entity.center_px[0] if vertical_exit else entity.center_px[1]
    lane_indices = {position[0] if vertical_exit else position[1] for position in target_exits}
    lane_centers = [index * TILE_SIZE + TILE_SIZE / 2.0 for index in lane_indices]
    target_axis = min(lane_centers, key=lambda center: abs(center - player_axis))
    offset = target_axis - player_axis
    if abs(offset) <= EXIT_ALIGNMENT_TOLERANCE_PX:
        return None
    if vertical_exit:
        return ACTION_RIGHT if offset > 0 else ACTION_LEFT
    return ACTION_DOWN if offset > 0 else ACTION_UP


def visible_exit_sides(state: SymbolicState) -> frozenset[Side]:
    return frozenset(
        side
        for side, action in SIDE_TO_ACTION.items()
        if side_exits(state, action)
    )


def bridge_connected_sides(state: SymbolicState, width: int = 10, height: int = 8) -> frozenset[Side]:
    if not state.bridges:
        return frozenset()
    sides: set[Side] = set()
    for x, y in state.bridges:
        if y == 0:
            sides.add("up")
        if y == height - 1:
            sides.add("down")
        if x == 0:
            sides.add("left")
        if x == width - 1:
            sides.add("right")
    for exit_x, exit_y in state.exits:
        if not any(manhattan((exit_x, exit_y), bridge) == 1 for bridge in state.bridges):
            continue
        if exit_y == 0:
            sides.add("up")
        if exit_y == height - 1:
            sides.add("down")
        if exit_x == 0:
            sides.add("left")
        if exit_x == width - 1:
            sides.add("right")
    return frozenset(sides)


def room_features(state: SymbolicState) -> RoomFeatures:
    bridge_sides = bridge_connected_sides(state)
    return RoomFeatures(
        exit_sides=visible_exit_sides(state),
        bridge_sides=bridge_sides,
        has_bridge=bool(state.bridges),
        has_switch=bool(state.switches),
        has_button=bool(state.buttons),
        has_chest=bool(state.chests),
        has_monster=bool(state.monsters),
        has_trap=bool(state.traps or state.gaps),
    )


def bridge_primary_side(state: SymbolicState, switch_side: Side | None = None) -> Side | None:
    sides = set(bridge_connected_sides(state))
    if switch_side is not None and len(sides) > 1:
        sides.discard(switch_side)
    for side in SIDE_ORDER:
        if side in sides:
            return side
    return None


def inventory_signature(info: dict) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
    inventory = info.get("inventory", {}) if isinstance(info, dict) else {}

    def number(name: str) -> int:
        try:
            return int(inventory.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    def names(name: str) -> tuple[str, ...]:
        values = inventory.get(name, ())
        if not isinstance(values, (list, tuple, set)):
            return ()
        return tuple(sorted(str(value) for value in values))

    return number("keys"), number("gold"), names("items"), names("tools")


def bfs_path(
    state: SymbolicState,
    goals: set[Position],
    extra_blocked: set[Position] | None = None,
) -> list[Position] | None:
    start = state.player
    if not in_bounds(start) or not goals:
        return None
    if start in goals:
        return [start]

    queue: deque[Position] = deque([start])
    parent: dict[Position, Position | None] = {start: None}

    while queue:
        current = queue.popleft()
        for nxt in neighbors(current):
            if nxt in parent:
                continue
            if not is_walkable(nxt, state, extra_blocked) and nxt not in goals:
                continue
            parent[nxt] = current
            if nxt in goals:
                path: list[Position] = []
                cursor: Position | None = nxt
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent[cursor]
                path.reverse()
                return path
            queue.append(nxt)

    return None


def astar_path(
    state: SymbolicState,
    goals: set[Position],
    extra_blocked: set[Position] | None = None,
    extra_unblocked: set[Position] | None = None,
    config: PlannerConfig | None = None,
    *,
    constrain_bridge: bool = True,
) -> list[Position] | None:
    config = DEFAULT_PLANNER_CONFIG if config is None else config
    start = state.player
    if not in_bounds(start) or not goals:
        return None
    if start in goals:
        return [start]

    blocked = blocked_tiles(state, extra_blocked, extra_unblocked, constrain_bridge=constrain_bridge)
    danger = monster_danger_tiles(state)
    trap_neighbors = {pos for trap in state.traps | state.gaps for pos in neighbors(trap) if in_bounds(pos)}

    frontier: list[tuple[float, int, Position]] = []
    heappush(frontier, (0.0, 0, start))
    parent: dict[Position, Position | None] = {start: None}
    cost_so_far: dict[Position, float] = {start: 0.0}
    counter = 0

    while frontier:
        _, _, current = heappop(frontier)
        if current in goals:
            path: list[Position] = []
            cursor: Position | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return path

        for nxt in neighbors(current):
            if not in_bounds(nxt):
                continue
            if nxt in blocked and nxt not in goals:
                continue

            step_cost = 1.0
            if nxt in danger:
                step_cost += config.monster_danger_cost
            if nxt in trap_neighbors:
                step_cost += config.trap_neighbor_cost
            new_cost = cost_so_far[current] + step_cost
            if new_cost >= cost_so_far.get(nxt, float("inf")):
                continue

            cost_so_far[nxt] = new_cost
            parent[nxt] = current
            counter += 1
            heuristic = min(manhattan(nxt, goal) for goal in goals)
            heappush(frontier, (new_cost + heuristic, counter, nxt))

    return None


def astar_path_to_side(
    state: SymbolicState,
    goals: set[Position],
    side: Side,
    extra_blocked: set[Position] | None = None,
    extra_unblocked: set[Position] | None = None,
    config: PlannerConfig | None = None,
) -> list[Position] | None:
    config = DEFAULT_PLANNER_CONFIG if config is None else config
    start = state.player
    if not in_bounds(start) or not goals:
        return None
    if start in goals:
        return [start]

    blocked = blocked_tiles(state, extra_blocked, extra_unblocked)
    danger = monster_danger_tiles(state)
    trap_neighbors = {pos for trap in state.traps | state.gaps for pos in neighbors(trap) if in_bounds(pos)}

    start_key: tuple[Position, int | None] = (start, None)
    frontier: list[tuple[float, int, tuple[Position, int | None]]] = []
    heappush(frontier, (0.0, 0, start_key))
    parent: dict[tuple[Position, int | None], tuple[Position, int | None] | None] = {start_key: None}
    cost_so_far: dict[tuple[Position, int | None], float] = {start_key: 0.0}
    counter = 0

    while frontier:
        _, _, current_key = heappop(frontier)
        current, previous_action = current_key
        if current in goals:
            path: list[Position] = []
            cursor: tuple[Position, int | None] | None = current_key
            while cursor is not None:
                path.append(cursor[0])
                cursor = parent[cursor]
            path.reverse()
            return path

        for nxt in neighbors(current):
            if not in_bounds(nxt):
                continue
            if nxt in blocked and nxt not in goals:
                continue

            action = action_from_step(current, nxt)
            step_cost = side_step_cost(current, nxt, action, previous_action, side)
            if nxt in danger:
                step_cost += config.monster_danger_cost
            if nxt in trap_neighbors:
                step_cost += config.trap_neighbor_cost

            next_key = (nxt, action)
            new_cost = cost_so_far[current_key] + step_cost
            if new_cost >= cost_so_far.get(next_key, float("inf")):
                continue

            cost_so_far[next_key] = new_cost
            parent[next_key] = current_key
            counter += 1
            heuristic = min(manhattan(nxt, goal) for goal in goals)
            heappush(frontier, (new_cost + heuristic, counter, next_key))

    return None


def side_step_cost(
    current: Position,
    nxt: Position,
    action: int,
    previous_action: int | None,
    side: Side,
) -> float:
    step_cost = 1.0
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]

    if side == "right":
        progress = dx
        perpendicular = dy != 0
    elif side == "left":
        progress = -dx
        perpendicular = dy != 0
    elif side == "down":
        progress = dy
        perpendicular = dx != 0
    else:
        progress = -dy
        perpendicular = dx != 0

    if progress > 0:
        step_cost -= 0.15
    elif progress < 0:
        step_cost += 3.0

    if perpendicular:
        step_cost += 0.35
    if previous_action is not None and previous_action != action:
        step_cost += 0.8
    return max(0.25, step_cost)


def monster_danger_tiles(state: SymbolicState) -> set[Position]:
    danger: set[Position] = set()
    for monster in state.monsters:
        danger.add(monster)
        danger.update(pos for pos in neighbors(monster) if in_bounds(pos))
    return danger


class Policy:
    def __init__(self, perception_engine=None, config: PlannerConfig | None = None) -> None:
        if perception_engine is None:
            from nesylink.perception import PerceptionEngine

            perception_engine = PerceptionEngine(device="cpu")
        self.perception_engine = perception_engine
        self.config = planner_config_from_env() if config is None else config
        self.pending_interaction: InteractionIntent | None = None
        self.awaiting_interaction: InteractionAttempt | None = None
        self.last_raw_state: SymbolicState | None = None
        self.last_reward = 0.0
        self.committed_action: int | None = None
        self.committed_ticks_remaining = 0
        self.committed_target_tile: Position | None = None
        self.committed_target_seen_ticks = 0
        self.committed_allow_exit = False
        self.remembered_blockers: set[Position] = set()
        self.hub_exploration_started = False
        self.current_target_side: Side | None = None
        self.switch_hub_side: Side | None = None
        self.hub_switch_positions: set[Position] = set()
        self.pressed_switch_for_target = False
        self.explored_hub_sides: set[Side] = set()
        self.saw_monster_objective = False
        self.monster_objective_done = False
        self.last_room_signature: str | None = None
        self.room_memory: dict[str, RoomMemory] = {}
        self.exit_memory: dict[tuple[str, Side], ExitMemory] = {}
        self.current_room_signature: str | None = None
        self.attempted_exit: tuple[str, Side] | None = None
        self.global_planner_started = False
        self.temporal_filter = TemporalSymbolicFilter()
        self.inventory_revision = 0
        self.last_inventory_signature: tuple[int, int, tuple[str, ...], tuple[str, ...]] | None = None
        self.inventory_changed_this_step = False
        self.hp_estimate = 5
        self.last_player_tile: Position | None = None
        self.recent_player_tiles: deque[Position] = deque(maxlen=4)
        self.stuck_ticks = 0
        self.hub_search_visits: dict[Position, int] = {}
        self.hub_scanned_frontiers: dict[frozenset[Side], set[Position]] = {}
        self.scanned_bridge_states: set[frozenset[Side]] = set()
        self.post_goal_rotate_bridge = False
        self.post_goal_switch_pressed = False
        self.monster_absence_ticks = 0
        self.combat_attack_pending = False
        self.combat_progress_observed = False
        self.shield_cooldown = 0
        self.recovery_forced_action: int | None = None
        self.recovery_forced_allow_exit = False

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed, task_id
        self.pending_interaction = None
        self.awaiting_interaction = None
        self.last_raw_state = None
        self.last_reward = 0.0
        self.committed_action = None
        self.committed_ticks_remaining = 0
        self.committed_target_tile = None
        self.committed_target_seen_ticks = 0
        self.committed_allow_exit = False
        self.remembered_blockers = set()
        self.hub_exploration_started = False
        self.current_target_side = None
        self.switch_hub_side = None
        self.hub_switch_positions = set()
        self.pressed_switch_for_target = False
        self.explored_hub_sides = set()
        self.saw_monster_objective = False
        self.monster_objective_done = False
        self.last_room_signature = None
        self.room_memory = {}
        self.exit_memory = {}
        self.current_room_signature = None
        self.attempted_exit = None
        self.global_planner_started = False
        self.temporal_filter.reset()
        self.inventory_revision = 0
        self.last_inventory_signature = None
        self.inventory_changed_this_step = False
        self.hp_estimate = 5
        self.last_player_tile = None
        self.recent_player_tiles = deque(maxlen=4)
        self.stuck_ticks = 0
        self.hub_search_visits = {}
        self.hub_scanned_frontiers = {}
        self.scanned_bridge_states = set()
        self.post_goal_rotate_bridge = False
        self.post_goal_switch_pressed = False
        self.monster_absence_ticks = 0
        self.combat_attack_pending = False
        self.combat_progress_observed = False
        self.shield_cooldown = 0
        self.recovery_forced_action = None
        self.recovery_forced_allow_exit = False

    def act(self, obs, info) -> int:
        signature = inventory_signature(info)
        keys, _, items, _ = signature
        self._update_inventory_progress(signature)
        self._update_resource_estimate(info)

        if self.committed_action is not None and self.committed_ticks_remaining > 0:
            state = self._extract_state(obs)
            emergency_shield = self._emergency_shield_action(state)
            if emergency_shield is not None:
                return emergency_shield
            if self._committed_move_complete(state):
                pass
            else:
                recovery_action = self._stuck_recovery_action(state)
                if recovery_action is not None:
                    return recovery_action
                if self._committed_room_changed(state):
                    self._clear_commit()
                else:
                    self.committed_ticks_remaining -= 1
                    action = self.committed_action
                    if self.committed_ticks_remaining == 0:
                        self._clear_commit()
                    return action

        state = self._extract_state(obs)
        self._update_stuck_counter(state)
        features = room_features(state)

        interaction_action = self._advance_interaction(state)
        if interaction_action is not None:
            return interaction_action

        forced_action = self._consume_recovery_forced_action(state)
        if forced_action is not None:
            return forced_action

        if self._should_use_global_planner(features):
            self.global_planner_started = True
            return self._act_global_planner(state, features, keys)

        switch_hub_signal = bool(state.switches and state.walls)
        bridge_hub_signal = bool(state.bridges and state.gaps)
        use_hub_exploration = switch_hub_signal or bridge_hub_signal or self.hub_exploration_started
        if use_hub_exploration:
            self.hub_exploration_started = True
        return self._act_local_planner(state, features, keys, items, use_hub_exploration)

    def _should_use_global_planner(self, features: RoomFeatures) -> bool:
        if self.global_planner_started:
            return True
        return features.has_button

    def _update_inventory_progress(
        self,
        signature: tuple[int, int, tuple[str, ...], tuple[str, ...]],
    ) -> None:
        if self.last_inventory_signature is None:
            self.last_inventory_signature = signature
            self.inventory_changed_this_step = False
            return
        self.inventory_changed_this_step = signature != self.last_inventory_signature
        if self.inventory_changed_this_step:
            self.inventory_revision += 1
            self.last_inventory_signature = signature

    def _update_resource_estimate(self, info: dict) -> None:
        if not isinstance(info, dict):
            return
        try:
            last_reward = float(info.get("last_reward", 0.0))
        except (TypeError, ValueError):
            return
        self.last_reward = last_reward
        if self.combat_attack_pending:
            self.combat_progress_observed = last_reward > 0.25
            self.combat_attack_pending = False
        if last_reward <= -1.0:
            self.hp_estimate = max(1, self.hp_estimate - 1)
        elif last_reward >= 2.0:
            self.hp_estimate = min(5, self.hp_estimate + 1)

    def _path_config(self) -> PlannerConfig:
        if not self.global_planner_started:
            return self.config
        low_hp_pressure = max(0, 3 - self.hp_estimate)
        if low_hp_pressure == 0:
            return self.config
        return replace(
            self.config,
            monster_danger_cost=self.config.monster_danger_cost + 3.0 * low_hp_pressure,
            trap_neighbor_cost=self.config.trap_neighbor_cost + 2.0 * low_hp_pressure,
        )

    def _emergency_shield_action(self, state: SymbolicState) -> int | None:
        if not self.global_planner_started:
            return None
        if self.committed_action not in ACTION_TO_DELTA:
            return None

        step = ACTION_TO_DELTA[self.committed_action]
        next_tile = (state.player[0] + step[0], state.player[1] + step[1])
        for monster in state.monsters:
            distance = manhattan(state.player, monster)
            moving_toward_monster = in_bounds(next_tile) and manhattan(next_tile, monster) < distance
            if distance <= 1 or (distance <= self.config.shield_distance and moving_toward_monster):
                self._clear_commit()
                return ACTION_B
        return None

    def _advance_interaction(self, state: SymbolicState) -> int | None:
        if self.awaiting_interaction is not None:
            attempt = self.awaiting_interaction
            if self._interaction_succeeded(attempt):
                self._confirm_interaction(attempt.intent)
                self.awaiting_interaction = None
            else:
                raw_state = self.last_raw_state or state
                visible_targets = self._interaction_targets(raw_state, attempt.intent.kind)
                if attempt.intent.target in visible_targets:
                    self.awaiting_interaction = None
                else:
                    attempt.missing_frames += 1
                    if attempt.missing_frames < INTERACTION_VISUAL_CONFIRM_TICKS:
                        return ACTION_NOOP
                    self._confirm_interaction(attempt.intent)
                    self.awaiting_interaction = None

        if self.pending_interaction is None:
            return None

        intent = self.pending_interaction
        self.pending_interaction = None
        raw_state = self.last_raw_state or state
        if adjacent_action(raw_state.player, intent.target) != intent.action_to_face:
            goals = adjacent_walkable_goals(raw_state, {intent.target}, self._current_room_blockers())
            return self._move_to_goals(raw_state, goals, self.remembered_blockers)

        if intent.kind in {"chest", "switch", "button"}:
            self.awaiting_interaction = InteractionAttempt(intent, self.inventory_revision)
        if intent.kind == "monster":
            self.combat_attack_pending = True
        return ACTION_A

    def _interaction_succeeded(self, attempt: InteractionAttempt) -> bool:
        return (
            self.inventory_revision > attempt.inventory_revision
            or self.last_reward >= INTERACTION_SUCCESS_REWARD
        )

    def _confirm_interaction(self, intent: InteractionIntent) -> None:
        if intent.kind == "chest":
            self.remembered_blockers.add(intent.target)
            if self.current_room_signature is not None:
                self.room_memory.setdefault(self.current_room_signature, RoomMemory()).opened_chests.add(intent.target)
            return
        if intent.kind == "switch":
            self.pressed_switch_for_target = True
            if self.monster_objective_done and self.post_goal_rotate_bridge:
                self.post_goal_switch_pressed = True
            return
        if intent.kind == "button" and self.current_room_signature is not None:
            memory = self.room_memory.setdefault(self.current_room_signature, RoomMemory())
            memory.pressed_buttons.add(intent.target)
            memory.interaction_done = True

    @staticmethod
    def _interaction_targets(state: SymbolicState, kind: InteractionKind) -> frozenset[Position]:
        if kind == "chest":
            return state.chests
        if kind == "switch":
            return state.switches
        if kind == "button":
            return state.buttons
        if kind == "monster":
            return state.monsters
        return frozenset()

    def _act_local_planner(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        items: tuple[str, ...],
        use_hub_exploration: bool,
    ) -> int:
        if use_hub_exploration:
            room = self._room_signature(features)
            if room != "unknown" and room != self.last_room_signature:
                self.remembered_blockers = set()
                self.last_room_signature = room

            self._remember_hub_context(state, features)
            if state.player == (-1, -1):
                fallback = self._missing_player_hub_action(state, keys, "sword" in items)
                if fallback is not None:
                    return self._commit(fallback, TILE_MOVE_TICKS)

            self._update_monster_progress(state, features)

        has_sword = "sword" in items
        objective = self._choose_local_objective(
            state,
            features,
            keys,
            has_sword,
            use_hub_exploration,
        )
        return self._execute_objective(state, objective)

    def _choose_local_objective(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        has_sword: bool,
        use_hub_exploration: bool,
    ) -> Objective:
        if state.monsters and (has_sword or not use_hub_exploration):
            return Objective("fight", targets=state.monsters)

        chest_targets = frozenset(set(state.chests) - self.remembered_blockers)
        if chest_targets and (keys <= 0 or use_hub_exploration):
            return Objective(
                "interact",
                targets=chest_targets,
                interaction_kind="chest",
            )

        if use_hub_exploration:
            return self._choose_bridge_objective(state, features, keys, has_sword)

        exit_side = self._choose_visible_exit_side(state, keys)
        if exit_side is not None:
            return Objective("go_exit", side=exit_side)
        return Objective("navigate", targets=frozenset(fallback_exit_goals(state)))

    def _update_monster_progress(self, state: SymbolicState, features: RoomFeatures) -> None:
        if state.monsters:
            self.saw_monster_objective = True
            self.monster_absence_ticks = 0
            return
        if not self.saw_monster_objective or features.has_bridge or not self.combat_progress_observed:
            self.monster_absence_ticks = 0
            return

        self.monster_absence_ticks += 1
        if self.monster_absence_ticks >= 3:
            self.monster_objective_done = True

    def _choose_visible_exit_side(self, state: SymbolicState, keys: int) -> Side | None:
        sides = visible_exit_sides(state)
        if not sides:
            return None
        preferred = ("right", "left", "down", "up") if keys > 0 else ("left", "down", "right", "up")
        for side in preferred:
            if side in sides:
                return side
        return next(iter(sides))

    def _choose_bridge_objective(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        has_sword: bool,
    ) -> Objective:
        is_hub_room = features.has_bridge or len(features.exit_sides) > 1

        if is_hub_room and self.post_goal_switch_pressed:
            self.post_goal_rotate_bridge = False
            self.post_goal_switch_pressed = False

        if is_hub_room and self.monster_objective_done and state.chests:
            return Objective(
                "interact",
                targets=state.chests,
                interaction_kind="chest",
                search="astar",
                constrain_bridge=False,
            )

        if is_hub_room and self.monster_objective_done and self.post_goal_rotate_bridge:
            if self.switch_hub_side is None:
                return Objective("idle")
            return Objective("go_exit", side=self.switch_hub_side)

        if is_hub_room and self.monster_objective_done:
            search_goals = adjacent_walkable_goals(state, set(state.chests), self.remembered_blockers)
            if not search_goals and features.has_bridge:
                search_objective = self._bridge_frontier_objective(state)
                if search_objective is not None:
                    return search_objective
                signature = bridge_connected_sides(state)
                if signature in self.scanned_bridge_states or self.switch_hub_side is None:
                    return Objective("idle")
                self.scanned_bridge_states.add(signature)
                self.post_goal_rotate_bridge = True
                self.post_goal_switch_pressed = False
                return Objective("go_exit", side=self.switch_hub_side)
            return Objective("navigate", targets=frozenset(search_goals))

        if is_hub_room:
            target_side = self._choose_target_side(state, keys, has_sword)
            if target_side is None:
                return Objective("idle")
            reachable_sides = features.bridge_sides or features.exit_sides
            if target_side in reachable_sides:
                self.pressed_switch_for_target = False
                return Objective("go_exit", side=target_side)
            switch_side = self.switch_hub_side or bridge_primary_side(state)
            self.pressed_switch_for_target = False
            if switch_side is None:
                return Objective("idle")
            return Objective("go_exit", side=switch_side)

        if self._is_known_switch_room(features) and self.monster_objective_done and self.post_goal_rotate_bridge:
            if not self.post_goal_switch_pressed:
                return self._switch_objective(state)
            return self._center_exit_objective(state, features)

        if self._is_known_switch_room(features) and self.current_target_side is not None and not self.pressed_switch_for_target:
            return self._switch_objective(state)

        self._mark_current_side_explored(features)
        return self._center_exit_objective(state, features)

    def _bridge_frontier_objective(self, state: SymbolicState) -> Objective | None:
        if state.player in state.bridges:
            self.hub_search_visits[state.player] = self.hub_search_visits.get(state.player, 0) + 1

        bridge_tiles = set(state.bridges)
        frontier = {
            tile
            for tile in bridge_tiles
            if is_walkable(tile, state, self.remembered_blockers)
            and any(in_bounds(neighbor) and neighbor not in bridge_tiles for neighbor in neighbors(tile))
            and not any(neighbor in state.exits for neighbor in neighbors(tile))
        }
        signature = bridge_connected_sides(state)
        visited = self.hub_scanned_frontiers.setdefault(signature, set())
        if state.player in frontier:
            visited.add(state.player)
        required_viewpoints = min(BRIDGE_SCAN_VIEWPOINTS, max(1, len(frontier)))
        if len(visited) >= required_viewpoints:
            return None
        ordered = sorted(
            frontier - visited,
            key=lambda tile: (
                self.hub_search_visits.get(tile, 0),
                manhattan(state.player, tile),
                tile,
            ),
        )
        for target in ordered:
            path = bfs_path(state, {target}, self.remembered_blockers)
            if path is None or len(path) < 2:
                continue
            return Objective("navigate", targets=frozenset({target}))
        return None

    def _committed_room_changed(self, state: SymbolicState) -> bool:
        features = room_features(state)
        if self.global_planner_started:
            room = self._structural_room_signature(state, features)
            previous_room = self.current_room_signature
        elif self.hub_exploration_started:
            room = self._room_signature(features)
            previous_room = self.last_room_signature
        else:
            return False
        return room != "unknown" and previous_room is not None and room != previous_room

    def _extract_state(self, obs) -> SymbolicState:
        state = self.perception_engine.extract(obs)
        self.last_raw_state = state
        return self.temporal_filter.update(
            state,
            inventory_changed=self.inventory_changed_this_step,
            suppressed_chests=self._current_room_blockers(),
            trust_consistent_player=(
                self.pending_interaction is not None
                or self.awaiting_interaction is not None
            ),
        )

    def _missing_player_hub_action(self, state: SymbolicState, keys: int, has_sword: bool) -> int | None:
        target_side = self._choose_target_side(state, keys, has_sword)
        visible_sides = bridge_connected_sides(state) or visible_exit_sides(state)
        if target_side in visible_sides:
            return SIDE_TO_ACTION[target_side]
        if self.switch_hub_side is not None:
            return SIDE_TO_ACTION[self.switch_hub_side]
        return self.committed_action

    def _room_signature(self, features: RoomFeatures) -> str:
        if features.has_bridge:
            return "bridge:" + ",".join(sorted(features.bridge_sides))
        pieces = list(sorted(features.exit_sides))
        if features.has_switch:
            pieces.append("switch")
        if features.has_button:
            pieces.append("button")
        if features.has_monster:
            pieces.append("monster")
        return ",".join(pieces) if pieces else "unknown"

    def _remember_hub_context(self, state: SymbolicState, features: RoomFeatures) -> None:
        if self.switch_hub_side is None and features.has_switch and len(features.exit_sides) == 1:
            exit_side = next(iter(features.exit_sides))
            self.switch_hub_side = OPPOSITE_SIDE[exit_side]
        if features.has_switch and self._is_known_switch_room(features):
            self.hub_switch_positions.update(state.switches)

    def _is_known_switch_room(self, features: RoomFeatures) -> bool:
        if len(features.exit_sides) != 1:
            return False
        center_side = OPPOSITE_SIDE[next(iter(features.exit_sides))]
        if self.switch_hub_side is not None:
            return center_side == self.switch_hub_side
        return features.has_switch

    def _mark_current_side_explored(self, features: RoomFeatures) -> None:
        if self._is_known_switch_room(features):
            return
        if len(features.exit_sides) == 1:
            exit_side = next(iter(features.exit_sides))
            self.explored_hub_sides.add(OPPOSITE_SIDE[exit_side])

    def _choose_target_side(self, state: SymbolicState, keys: int, has_sword: bool) -> Side | None:
        active_side = bridge_primary_side(state, self.switch_hub_side)
        if keys <= 0 and active_side is not None:
            self.current_target_side = active_side
            return active_side

        if has_sword and self.monster_objective_done:
            self.current_target_side = None
            return None

        if self.current_target_side is not None and self.current_target_side not in self.explored_hub_sides:
            return self.current_target_side

        candidates: list[Side] = []
        if active_side is not None:
            candidates.append(active_side)
        candidates.extend(SIDE_ORDER)
        for side in candidates:
            if side == self.switch_hub_side:
                continue
            if side in self.explored_hub_sides:
                continue
            self.current_target_side = side
            return side
        return active_side

    def _center_exit_objective(self, state: SymbolicState, features: RoomFeatures) -> Objective:
        if features.exit_sides:
            side = min(features.exit_sides, key=lambda candidate: len(side_exits(state, SIDE_TO_ACTION[candidate])))
            return Objective("go_exit", side=side)
        return Objective("navigate", targets=frozenset(fallback_exit_goals(state)))

    def _switch_objective(self, state: SymbolicState) -> Objective:
        targets = frozenset(set(state.switches) or set(self.hub_switch_positions))
        if not targets:
            return Objective("go_exit", side="right")
        return Objective(
            "interact",
            targets=targets,
            interaction_kind="switch",
        )

    def _act_global_planner(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
    ) -> int:
        if self.shield_cooldown > 0:
            self.shield_cooldown -= 1

        room_sig = self._update_memory(state, features)
        objective = self._choose_global_objective(state, features, keys, room_sig)
        return self._execute_objective(state, objective)

    def _update_memory(self, state: SymbolicState, features: RoomFeatures) -> str:
        room_sig = self._structural_room_signature(state, features)
        if self.current_room_signature is None:
            self.current_room_signature = room_sig
        elif room_sig != self.current_room_signature:
            if self.attempted_exit is not None:
                self.exit_memory[self.attempted_exit] = ExitMemory(status="open", leads_to=room_sig)
            self.current_room_signature = room_sig
            self.remembered_blockers = set()
            self.attempted_exit = None
        elif self.attempted_exit is not None and self.committed_action is None:
            memory = self.exit_memory.setdefault(self.attempted_exit, ExitMemory())
            if memory.status == "unknown":
                memory.status = "blocked"
            self.attempted_exit = None

        memory = self.room_memory.setdefault(room_sig, RoomMemory())
        memory.exits.update(features.exit_sides)
        missing_known_chests = memory.known_chests - set(state.chests)
        if self.inventory_changed_this_step:
            memory.opened_chests.update(missing_known_chests)
        memory.known_chests.update(state.chests)
        memory.has_monster = memory.has_monster or features.has_monster
        memory.has_button = memory.has_button or features.has_button
        if memory.pressed_buttons:
            memory.interaction_done = True
        memory.visited_inventory_revisions.add(self.inventory_revision)
        return room_sig

    def _structural_room_signature(self, state: SymbolicState, features: RoomFeatures) -> str:
        pieces = [
            "exits=" + ",".join(sorted(features.exit_sides)),
            "walls=" + ",".join(f"{x}:{y}" for x, y in sorted(state.walls)),
        ]
        return "|".join(pieces) if any(pieces) else "unknown"

    def _choose_global_objective(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        room_sig: str,
    ) -> Objective:
        memory = self.room_memory.setdefault(room_sig, RoomMemory())
        chest_targets = frozenset(set(state.chests) - memory.opened_chests - self.remembered_blockers)
        adjacent_monster = self._interaction_if_adjacent(state, set(state.monsters), kind="monster")
        if adjacent_monster is not None:
            if any(manhattan(state.player, chest) == 1 for chest in chest_targets):
                return Objective(
                    "interact",
                    targets=chest_targets,
                    interaction_kind="chest",
                    search="astar",
                    safe=True,
                )
            if self._shield_available(state):
                return Objective("shield")
            return Objective("fight", targets=state.monsters, search="astar", safe=True)

        if self._shield_available(state):
            return Objective("shield")

        if chest_targets:
            return Objective(
                "interact",
                targets=chest_targets,
                interaction_kind="chest",
                search="astar",
                safe=bool(state.monsters),
            )

        button_targets = frozenset(set(state.buttons) - memory.pressed_buttons)
        if button_targets:
            return Objective(
                "interact",
                targets=button_targets,
                interaction_kind="button",
                search="astar",
            )

        if len(features.exit_sides) == 1 and not self._current_room_needs_revisit(memory):
            return Objective("go_exit", side=next(iter(features.exit_sides)))

        side = self._choose_exploration_side(state, features, keys, room_sig)
        if side is not None:
            return Objective("go_exit", side=side)
        return Objective("navigate", targets=frozenset(fallback_exit_goals(state)))

    def _current_room_needs_revisit(self, memory: RoomMemory) -> bool:
        return self.inventory_revision not in memory.visited_inventory_revisions

    def _choose_exploration_side(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        room_sig: str,
    ) -> Side | None:
        candidates: list[tuple[int, float, int, Side]] = []
        for order, side in enumerate(SIDE_ORDER):
            if side not in features.exit_sides:
                continue
            memory = self.exit_memory.get((room_sig, side), ExitMemory())
            priority = self._exit_objective_priority(memory, keys)
            score = 0.0
            if memory.status == "unknown":
                score -= self.config.unknown_exit_bonus
            elif memory.status == "blocked" and keys > 0:
                score -= self.config.key_blocked_retry_bonus
            elif memory.status == "blocked":
                score += self.config.blocked_exit_penalty
            elif memory.leads_to is not None:
                score += self._global_room_score(memory.leads_to)

            exits = side_exits(state, SIDE_TO_ACTION[side])
            path = astar_path(
                state,
                self._side_approach_goals(state, side, self._current_room_blockers()),
                self._current_room_blockers() | (set(state.exits) - exits),
                None,
                self._path_config(),
            )
            path_weight = 0.0 if keys > 0 and memory.status in {"unknown", "blocked"} else 1.0
            score += path_weight * float(len(path) if path else 100)
            candidates.append((priority, score, order, side))
        if not candidates:
            return None
        return min(candidates)[3]

    def _exit_objective_priority(self, memory: ExitMemory, keys: int) -> int:
        if memory.status == "blocked" and keys > 0:
            return 0
        if memory.status == "open" and memory.leads_to is not None:
            if self._room_has_unfinished_goal(memory.leads_to):
                return 1
            if self._room_has_unknown_frontier(memory.leads_to):
                return 2
            return 5
        if memory.status == "unknown":
            return 3
        if memory.status == "blocked":
            return 6
        return 4

    def _room_has_unfinished_goal(self, room_sig: str) -> bool:
        room = self.room_memory.get(room_sig)
        if room is None:
            return True
        return bool(
            room.known_chests - room.opened_chests
            or (room.has_button and not room.interaction_done)
        )

    def _room_has_unknown_frontier(self, room_sig: str) -> bool:
        room = self.room_memory.get(room_sig)
        if room is None:
            return True
        return any(
            self.exit_memory.get((room_sig, side), ExitMemory()).status == "unknown"
            for side in room.exits
        )

    def _global_room_score(self, start_room: str) -> float:
        frontier: list[tuple[float, str]] = [(0.0, start_room)]
        best: dict[str, float] = {start_room: 0.0}
        fallback = self.config.completed_room_penalty

        while frontier:
            distance, room_sig = heappop(frontier)
            if distance > best.get(room_sig, float("inf")):
                continue
            room = self.room_memory.get(room_sig)
            if room is None:
                continue

            unopened = room.known_chests - room.opened_chests
            if unopened:
                return distance - self.config.unfinished_room_bonus * (1.0 + len(unopened))

            has_unknown_frontier = any(
                self.exit_memory.get((room_sig, side), ExitMemory()).status == "unknown"
                for side in room.exits
            )
            if has_unknown_frontier:
                fallback = min(fallback, distance - self.config.unknown_exit_bonus * 0.5)

            for side in room.exits:
                edge = self.exit_memory.get((room_sig, side))
                if edge is None or edge.status != "open" or edge.leads_to is None:
                    continue
                risk = 2.0 if room.has_monster else 0.0
                new_distance = distance + 1.0 + risk
                if new_distance >= best.get(edge.leads_to, float("inf")):
                    continue
                best[edge.leads_to] = new_distance
                heappush(frontier, (new_distance, edge.leads_to))

        return fallback

    def _opened_chests_for_current_room(self) -> set[Position]:
        if self.current_room_signature is None:
            return set()
        return set(self.room_memory.setdefault(self.current_room_signature, RoomMemory()).opened_chests)

    def _current_room_blockers(self) -> set[Position]:
        return set(self.remembered_blockers) | self._opened_chests_for_current_room()

    def _shield_if_close(self, state: SymbolicState) -> int | None:
        if self.shield_cooldown > 0:
            return None
        if any(manhattan(state.player, monster) <= self.config.shield_distance for monster in state.monsters):
            self.shield_cooldown = self.config.shield_cooldown_ticks
            return ACTION_B
        return None

    def _shield_available(self, state: SymbolicState) -> bool:
        return self.shield_cooldown <= 0 and any(
            manhattan(state.player, monster) <= self.config.shield_distance
            for monster in state.monsters
        )

    def _go_to_side_via_approach(self, state: SymbolicState, side: Side) -> int:
        direction = SIDE_TO_ACTION[side]
        exits = side_exits(state, direction)
        if state.player in exits:
            alignment = exit_alignment_action(state, side, exits)
            if alignment is not None:
                return self._commit(alignment, EXIT_ALIGNMENT_TICKS, allow_exit=True)
            if self.current_room_signature is not None:
                self.attempted_exit = (self.current_room_signature, side)
            return self._commit(direction, EXIT_MOVE_TICKS, allow_exit=True)
        non_target_exits = set(state.exits) - exits
        extra_blocked = (self._current_room_blockers() - exits) | non_target_exits
        approach = self._side_approach_goals(state, side, extra_blocked)

        if state.player in approach:
            alignment = exit_alignment_action(state, side, exits)
            if alignment is not None:
                return self._commit(alignment, EXIT_ALIGNMENT_TICKS, allow_exit=True)
            if self.current_room_signature is not None:
                self.attempted_exit = (self.current_room_signature, side)
            dx, dy = ACTION_TO_DELTA[direction]
            target = (state.player[0] + dx, state.player[1] + dy)
            target_tile = target if in_bounds(target) and target not in state.exits else None
            return self._commit(
                direction,
                TILE_MOVE_TICKS,
                target_tile=target_tile,
                allow_exit=True,
            )
        preferred_approach = self._nearest_side_approach_goals(state, side, approach)
        path = astar_path_to_side(state, preferred_approach, side, extra_blocked, None, self._path_config())
        if path is None and preferred_approach != approach:
            path = astar_path_to_side(state, approach, side, extra_blocked, None, self._path_config())
        if path is None or len(path) < 2:
            return ACTION_NOOP
        if path[1] in non_target_exits:
            self.remembered_blockers.add(path[1])
            return ACTION_NOOP
        target_tile = path[1] if path[1] not in state.exits else None
        return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS, target_tile=target_tile)

    def _nearest_side_approach_goals(self, state: SymbolicState, side: Side, approach: set[Position]) -> set[Position]:
        if not approach:
            return set()
        if side in {"left", "right"}:
            best = min(abs(pos[1] - state.player[1]) for pos in approach)
            return {pos for pos in approach if abs(pos[1] - state.player[1]) == best}
        best = min(abs(pos[0] - state.player[0]) for pos in approach)
        return {pos for pos in approach if abs(pos[0] - state.player[0]) == best}

    def _side_approach_goals(
        self,
        state: SymbolicState,
        side: Side,
        extra_blocked: set[Position] | None = None,
        extra_unblocked: set[Position] | None = None,
    ) -> set[Position]:
        exits = side_exits(state, SIDE_TO_ACTION[side])
        if side == "up":
            return {
                (x, 1)
                for x, y in exits
                if (x, y) not in (extra_blocked or set())
                and is_walkable((x, 1), state, extra_blocked, extra_unblocked)
            }
        if side == "down":
            return {
                (x, 6)
                for x, y in exits
                if (x, y) not in (extra_blocked or set())
                and is_walkable((x, 6), state, extra_blocked, extra_unblocked)
            }
        if side == "left":
            return {
                (1, y)
                for x, y in exits
                if (x, y) not in (extra_blocked or set())
                and is_walkable((1, y), state, extra_blocked, extra_unblocked)
            }
        return {
            (8, y)
            for x, y in exits
            if (x, y) not in (extra_blocked or set())
            and is_walkable((8, y), state, extra_blocked, extra_unblocked)
        }

    def _queue_interaction(self, intent: InteractionIntent) -> int:
        self.pending_interaction = intent
        return intent.action_to_face

    def _execute_objective(self, state: SymbolicState, objective: Objective) -> int:
        if objective.kind == "shield":
            return self._shield_if_close(state) or ACTION_NOOP
        if objective.kind == "fight":
            return self._execute_fight(state, objective)
        if objective.kind == "interact":
            return self._execute_interaction(state, objective)
        if objective.kind == "navigate":
            return self._navigate_objective(state, objective)
        if objective.kind == "go_exit" and objective.side is not None:
            return self._go_to_side_via_approach(state, objective.side)
        return ACTION_NOOP

    def _execute_fight(self, state: SymbolicState, objective: Objective) -> int:
        targets = set(objective.targets) or set(state.monsters)
        interact = self._interaction_if_adjacent(state, targets, kind="monster")
        if interact is not None:
            if objective.safe:
                shield_action = self._shield_if_close(state)
                if shield_action is not None:
                    return shield_action
            return self._queue_interaction(interact)
        if objective.safe:
            shield_action = self._shield_if_close(state)
            if shield_action is not None:
                return shield_action
        goals = monster_goals(state, self._objective_blockers(objective))
        return self._navigate_objective(state, replace(objective, kind="navigate", targets=frozenset(goals)))

    def _execute_interaction(self, state: SymbolicState, objective: Objective) -> int:
        kind = objective.interaction_kind
        targets = set(objective.targets)
        if kind is None or not targets:
            return ACTION_NOOP
        if kind == "button":
            if state.player in targets:
                if self.current_room_signature is not None:
                    self.room_memory.setdefault(
                        self.current_room_signature,
                        RoomMemory(),
                    ).pressed_buttons.add(state.player)
                return ACTION_NOOP
            return self._navigate_objective(state, replace(objective, kind="navigate"))

        interact = self._interaction_if_adjacent(state, targets, kind=kind)
        if interact is not None:
            return self._queue_interaction(interact)
        if objective.safe:
            shield_action = self._shield_if_close(state)
            if shield_action is not None:
                return shield_action
        goals = adjacent_walkable_goals(
            state,
            targets,
            self._objective_blockers(objective),
            constrain_bridge=objective.constrain_bridge,
        )
        if objective.safe:
            aligned_goals = {
                goal
                for goal in goals
                if goal[0] == state.player[0] or goal[1] == state.player[1]
            }
            if aligned_goals:
                goals = aligned_goals
        action = self._navigate_objective(
            state,
            replace(objective, kind="navigate", targets=frozenset(goals)),
        )
        if action != ACTION_NOOP or not objective.safe or not state.monsters:
            return action
        return self._execute_fight(
            state,
            Objective(
                "fight",
                targets=state.monsters,
                search=objective.search,
                safe=True,
            ),
        )

    def _navigate_objective(self, state: SymbolicState, objective: Objective) -> int:
        goals = set(objective.targets)
        blockers = self._objective_blockers(objective)
        return self._move_to_goals(
            state,
            goals,
            blockers,
            search=objective.search,
            constrain_bridge=objective.constrain_bridge,
        )

    def _objective_blockers(self, objective: Objective) -> set[Position]:
        if objective.search == "astar":
            return self._current_room_blockers()
        return set(self.remembered_blockers)

    def _interaction_if_adjacent(
        self,
        state: SymbolicState,
        targets: set[Position],
        *,
        kind: InteractionKind,
    ) -> InteractionIntent | None:
        if not targets:
            return None
        target = min(targets, key=lambda pos: manhattan(state.player, pos))
        face_action = adjacent_action(state.player, target)
        if face_action is None:
            return None
        return InteractionIntent(target=target, action_to_face=face_action, kind=kind)

    def _move_to_goals(
        self,
        state: SymbolicState,
        goals: set[Position],
        blocked: set[Position],
        *,
        search: SearchMode = "bfs",
        constrain_bridge: bool = True,
    ) -> int:
        if search == "astar":
            path = astar_path(
                state,
                goals,
                blocked,
                None,
                self._path_config(),
                constrain_bridge=constrain_bridge,
            )
        else:
            path = bfs_path(state, goals, blocked)
        if path is None or len(path) < 2:
            return ACTION_NOOP
        target_tile = path[1] if path[1] not in state.exits else None
        return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS, target_tile=target_tile)

    def _stuck_recovery_action(self, state: SymbolicState) -> int | None:
        self._update_stuck_counter(state)
        if self.stuck_ticks < self.config.stuck_recovery_ticks or self.committed_action not in ACTION_TO_DELTA:
            return None
        original_action = self.committed_action
        original_allow_exit = self.committed_allow_exit
        dx, dy = ACTION_TO_DELTA[self.committed_action]
        blocker = (state.player[0] + dx, state.player[1] + dy)
        visible_blockers = (
            set(state.walls)
            | set(state.traps)
            | set(state.gaps)
            | set(state.chests)
            | set(state.monsters)
            | npc_tiles(state)
        )
        if in_bounds(blocker) and blocker in visible_blockers:
            self.remembered_blockers.add(blocker)
            self._clear_commit()
            self.stuck_ticks = 0
            self.recent_player_tiles.clear()
            return ACTION_NOOP

        if original_allow_exit:
            side = ACTION_TO_SIDE[original_action]
            alignment = exit_alignment_action(state, side)
            if alignment is not None:
                self._clear_commit()
                self.stuck_ticks = 0
                self.recent_player_tiles.clear()
                return self._commit(alignment, EXIT_ALIGNMENT_TICKS, allow_exit=True)

        recovery_actions = (
            (ACTION_DOWN, ACTION_UP)
            if self.committed_action in {ACTION_LEFT, ACTION_RIGHT}
            else (ACTION_LEFT, ACTION_RIGHT)
        )
        self._clear_commit()
        self.stuck_ticks = 0
        self.recent_player_tiles.clear()
        for action in recovery_actions:
            step = ACTION_TO_DELTA[action]
            target = (state.player[0] + step[0], state.player[1] + step[1])
            if not original_allow_exit and self._in_exit_buffer(target, state):
                continue
            if is_walkable(target, state, self._current_room_blockers()):
                self.recovery_forced_action = original_action
                self.recovery_forced_allow_exit = original_allow_exit
                return self._commit(action, TILE_MOVE_TICKS, target_tile=target)
        return ACTION_NOOP

    def _consume_recovery_forced_action(self, state: SymbolicState) -> int | None:
        if self.recovery_forced_action is None:
            return None
        action = self.recovery_forced_action
        allow_exit = self.recovery_forced_allow_exit
        self.recovery_forced_action = None
        self.recovery_forced_allow_exit = False
        step = ACTION_TO_DELTA.get(action)
        if step is None:
            return None
        target = (state.player[0] + step[0], state.player[1] + step[1])
        if not is_walkable(target, state, self._current_room_blockers()):
            return None
        if not allow_exit and self._in_exit_buffer(target, state):
            return None
        return self._commit(
            action,
            TILE_MOVE_TICKS,
            target_tile=target,
            allow_exit=allow_exit,
        )

    @staticmethod
    def _in_exit_buffer(target: Position, state: SymbolicState) -> bool:
        return any(manhattan(target, exit_tile) <= 1 for exit_tile in state.exits)

    def _update_stuck_counter(self, state: SymbolicState) -> None:
        if state.player == self.last_player_tile:
            self.stuck_ticks += 1
            return

        self.last_player_tile = state.player
        self.stuck_ticks = 0
        self.recent_player_tiles.append(state.player)
        if (
            len(self.recent_player_tiles) == 4
            and self.recent_player_tiles[0] == self.recent_player_tiles[2]
            and self.recent_player_tiles[1] == self.recent_player_tiles[3]
            and self.recent_player_tiles[0] != self.recent_player_tiles[1]
        ):
            self.stuck_ticks = self.config.stuck_recovery_ticks

    def _committed_move_complete(self, state: SymbolicState) -> bool:
        if self.committed_action not in ACTION_TO_DELTA or self.committed_target_tile is None:
            return False
        if state.player != self.committed_target_tile:
            self.committed_target_seen_ticks = 0
            return False
        self.committed_target_seen_ticks += 1
        if self.committed_target_seen_ticks < self.config.target_tile_stable_ticks:
            return False
        self._clear_commit()
        return True

    def _clear_commit(self) -> None:
        self.committed_action = None
        self.committed_ticks_remaining = 0
        self.committed_target_tile = None
        self.committed_target_seen_ticks = 0
        self.committed_allow_exit = False

    def _commit(
        self,
        action: int | None,
        ticks: int,
        target_tile: Position | None = None,
        *,
        allow_exit: bool = False,
    ) -> int:
        if action is None:
            return ACTION_NOOP
        self.committed_action = action
        self.committed_ticks_remaining = max(0, ticks - 1)
        self.committed_target_tile = target_tile
        self.committed_target_seen_ticks = 0
        self.committed_allow_exit = allow_exit
        return action


def make_policy() -> Policy:
    return Policy()
