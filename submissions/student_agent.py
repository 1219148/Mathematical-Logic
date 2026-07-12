from __future__ import annotations

import os
from heapq import heappop, heappush
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Iterable

from nesylink.core.observation import TILE_NPC
from nesylink.core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
)
from nesylink.shared import SymbolicState


Position = tuple[int, int]
Side = str
TILE_MOVE_TICKS = 16
EXIT_MOVE_TICKS = 24


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
    kind: str


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
    has_trap: bool = False
    visits: int = 0


@dataclass
class ExitMemory:
    status: str = "unknown"
    leads_to: str | None = None


@dataclass(frozen=True)
class Objective:
    kind: str
    targets: frozenset[Position] = frozenset()
    side: Side | None = None


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
) -> set[Position]:
    blocked = set(state.walls) | npc_tiles(state) | set(state.traps) | set(state.gaps) | set(state.chests) | set(state.monsters)
    if bridge_requires_constrained_walk(state):
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
) -> bool:
    return in_bounds(pos) and pos not in blocked_tiles(state, extra_blocked, extra_unblocked)


def adjacent_walkable_goals(
    state: SymbolicState,
    targets: set[Position],
    extra_blocked: set[Position] | None = None,
) -> set[Position]:
    goals: set[Position] = set()
    for target in targets:
        for pos in neighbors(target):
            if is_walkable(pos, state, extra_blocked):
                goals.add(pos)
    return goals


def monster_goals(state: SymbolicState, extra_blocked: set[Position] | None = None) -> set[Position]:
    return adjacent_walkable_goals(state, set(state.monsters), extra_blocked)


def task1_exit_goals(state: SymbolicState) -> set[Position]:
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


def center_bridge_goals(state: SymbolicState) -> set[Position]:
    anchor = center_bridge_anchor(state)
    if anchor is None:
        return set()
    return adjacent_walkable_goals(state, {anchor})


def center_bridge_anchor(state: SymbolicState) -> Position | None:
    if not state.bridges:
        return None
    bridge_tiles = set(state.bridges)
    xs = [x for x, _ in bridge_tiles]
    ys = [y for _, y in bridge_tiles]
    mid_x = sum(xs) / len(xs)
    mid_y = sum(ys) / len(ys)
    return min(
        bridge_tiles,
        key=lambda pos: (
            -sum(neighbor in bridge_tiles for neighbor in neighbors(pos)),
            abs(pos[0] - mid_x) + abs(pos[1] - mid_y),
            pos[0],
            pos[1],
        ),
    )


def inventory_items(info: dict) -> tuple[str, ...]:
    inventory = info.get("inventory", {}) if isinstance(info, dict) else {}
    items = inventory.get("items", ())
    if isinstance(items, (list, tuple, set)):
        return tuple(str(item) for item in items)
    return ()


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


def inventory_keys(info: dict) -> int:
    inventory = info.get("inventory", {}) if isinstance(info, dict) else {}
    try:
        return int(inventory.get("keys", 0))
    except (TypeError, ValueError):
        return 0


class Policy:
    def __init__(self, perception_engine=None, config: PlannerConfig | None = None) -> None:
        if perception_engine is None:
            from nesylink.perception import PerceptionEngine

            perception_engine = PerceptionEngine(device="cpu")
        self.perception_engine = perception_engine
        self.config = planner_config_from_env() if config is None else config
        self.task_id: str | None = None
        self.pending_interaction: InteractionIntent | None = None
        self.committed_action: int | None = None
        self.committed_ticks_remaining = 0
        self.committed_target_tile: Position | None = None
        self.committed_target_seen_ticks = 0
        self.remembered_blockers: set[Position] = set()
        self.hub_exploration_started = False
        self.current_target_side: Side | None = None
        self.switch_hub_side: Side | None = None
        self.pressed_switch_for_target = False
        self.explored_hub_sides: set[Side] = set()
        self.saw_monster_objective = False
        self.monster_objective_done = False
        self.last_room_signature: str | None = None
        self.room_memory: dict[str, RoomMemory] = {}
        self.exit_memory: dict[tuple[str, Side], ExitMemory] = {}
        self.current_room_signature: str | None = None
        self.attempted_exit: tuple[str, Side] | None = None
        self.last_player_tile: Position | None = None
        self.last_reliable_player_tile: Position | None = None
        self.stuck_ticks = 0
        self.shield_cooldown = 0

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        self.task_id = task_id
        self.pending_interaction = None
        self.committed_action = None
        self.committed_ticks_remaining = 0
        self.committed_target_tile = None
        self.committed_target_seen_ticks = 0
        self.remembered_blockers = set()
        self.hub_exploration_started = False
        self.current_target_side = None
        self.switch_hub_side = None
        self.pressed_switch_for_target = False
        self.explored_hub_sides = set()
        self.saw_monster_objective = False
        self.monster_objective_done = False
        self.last_room_signature = None
        self.room_memory = {}
        self.exit_memory = {}
        self.current_room_signature = None
        self.attempted_exit = None
        self.last_player_tile = None
        self.last_reliable_player_tile = None
        self.stuck_ticks = 0
        self.shield_cooldown = 0

    def act(self, obs, info) -> int:
        if self.committed_action is not None and self.committed_ticks_remaining > 0:
            state = self._extract_state(obs)
            if self._committed_move_complete(state):
                pass
            elif self._stuck_recovery_action(state):
                return ACTION_NOOP
            else:
                task5_room_changed = False
                if self.task_id == "mathematical_logic/task_5":
                    current_kind = self._structural_room_signature(state, room_features(state))
                    task5_room_changed = (
                        current_kind != "unknown"
                        and self.current_room_signature is not None
                        and current_kind != self.current_room_signature
                    )
                if task5_room_changed or (self.task_id != "mathematical_logic/task_5" and self.hub_exploration_started and self._symbolic_room_changed(obs)):
                    self._clear_commit()
                else:
                    self.committed_ticks_remaining -= 1
                    action = self.committed_action
                    if self.committed_ticks_remaining == 0:
                        self._clear_commit()
                    return action

        if self.committed_action is not None and self.committed_ticks_remaining > 0:
            self.committed_ticks_remaining -= 1
            action = self.committed_action
            if self.committed_ticks_remaining == 0:
                self._clear_commit()
            return action

        state = self._extract_state(obs)
        self._update_stuck_counter(state)
        features = room_features(state)
        keys = inventory_keys(info)
        items = inventory_items(info)

        if self.task_id == "mathematical_logic/task_5":
            return self._act_task5(state, features, keys)

        use_hub_exploration = self.task_id == "mathematical_logic/task_4" or (
            self.task_id is None and (state.switches or state.bridges or self.hub_exploration_started)
        )
        if use_hub_exploration:
            self.hub_exploration_started = True
        return self._act_from_features(state, features, keys, items, use_hub_exploration)

    def _act_from_features(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        items: tuple[str, ...],
        use_hub_exploration: bool,
    ) -> int:
        if self.pending_interaction is not None:
            intent = self.pending_interaction
            self.pending_interaction = None
            if adjacent_action(state.player, intent.target) == intent.action_to_face:
                if intent.kind == "chest":
                    self.remembered_blockers.add(intent.target)
                if intent.kind == "switch":
                    self.pressed_switch_for_target = True
                return ACTION_A

        if use_hub_exploration:
            room = self._room_signature(features)
            if room != "unknown" and room != self.last_room_signature:
                self.remembered_blockers = set()
                self.last_room_signature = room

            self._remember_hub_context(features)
            if state.player == (-1, -1):
                fallback = self._missing_player_hub_action(state, keys, "sword" in items)
                if fallback is not None:
                    return self._commit(fallback, TILE_MOVE_TICKS)

            if state.monsters:
                self.saw_monster_objective = True
            if self.saw_monster_objective and not state.monsters and not features.has_bridge:
                self.monster_objective_done = True

        has_sword = "sword" in items

        if state.monsters and (has_sword or not use_hub_exploration):
            interact = self._interaction_if_adjacent(state, set(state.monsters), kind="monster")
            if interact is not None:
                self.pending_interaction = interact
                return interact.action_to_face
            return self._move_toward(state, monster_goals(state, self.remembered_blockers))

        chest_targets = set(state.chests) - self.remembered_blockers
        if chest_targets and (keys <= 0 or use_hub_exploration):
            interact = self._interaction_if_adjacent(state, chest_targets, kind="chest")
            if interact is not None:
                self.pending_interaction = interact
                return interact.action_to_face
            return self._move_toward(state, adjacent_walkable_goals(state, chest_targets, self.remembered_blockers))

        if use_hub_exploration:
            hub_action = self._hub_exploration_action(state, features, keys, has_sword)
            if hub_action is not None:
                return hub_action

        exit_side = self._choose_visible_exit_side(state, keys)
        if exit_side is not None:
            return self._go_to_side_via_approach(state, exit_side)

        return self._move_toward(state, task1_exit_goals(state))

    def _choose_visible_exit_side(self, state: SymbolicState, keys: int) -> Side | None:
        sides = visible_exit_sides(state)
        if not sides:
            return None
        preferred = ("right", "left", "down", "up") if keys > 0 else ("left", "down", "right", "up")
        for side in preferred:
            if side in sides:
                return side
        return next(iter(sides))

    def _hub_exploration_action(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        has_sword: bool,
    ) -> int | None:
        is_hub_room = features.has_bridge or len(features.exit_sides) > 1

        if is_hub_room and self.monster_objective_done and state.chests:
            return self._open_nearest_chest(state)

        if is_hub_room and self.monster_objective_done:
            search_goals = adjacent_walkable_goals(state, set(state.chests), self.remembered_blockers)
            if not search_goals and features.has_bridge:
                anchor = center_bridge_anchor(state)
                if anchor is not None:
                    interact = self._interaction_if_adjacent(state, {anchor}, kind="chest")
                    if interact is not None:
                        self.pending_interaction = interact
                        return interact.action_to_face
                    search_goals = center_bridge_goals(state)
            return self._move_toward(state, search_goals)

        if is_hub_room:
            target_side = self._choose_target_side(state, keys, has_sword)
            if target_side is None:
                return ACTION_NOOP
            reachable_sides = features.bridge_sides or features.exit_sides
            if target_side in reachable_sides:
                self.pressed_switch_for_target = False
                return self._go_to_side_via_approach(state, target_side)
            switch_side = self.switch_hub_side or bridge_primary_side(state)
            self.pressed_switch_for_target = False
            if switch_side is None:
                return ACTION_NOOP
            return self._go_to_side_via_approach(state, switch_side)

        if self._is_known_switch_room(features) and self.current_target_side is not None and not self.pressed_switch_for_target:
            return self._press_switch(state)

        self._mark_current_side_explored(features)
        return self._return_to_center(state, features)

    def _symbolic_room_changed(self, obs) -> bool:
        state = self._extract_state(obs)
        room = self._room_signature(room_features(state))
        return room != "unknown" and self.last_room_signature is not None and room != self.last_room_signature

    def _extract_state(self, obs) -> SymbolicState:
        state = self.perception_engine.extract(obs)
        return self._stabilize_player_tile(state)

    def _stabilize_player_tile(self, state: SymbolicState) -> SymbolicState:
        entity = state.player_entity
        if entity is None:
            self.last_reliable_player_tile = state.player
            return state

        center_tile = (
            int(entity.center_px[0] // 16),
            int(entity.center_px[1] // 16),
        )
        if not self._valid_player_tile(center_tile, state):
            self.last_reliable_player_tile = state.player
            return state

        player = center_tile
        if entity.confidence < 0.5 and self._valid_player_tile(state.player, state):
            player = state.player
        elif state.player in self._current_room_blockers() and center_tile not in self._current_room_blockers():
            player = center_tile
        if (
            self.last_reliable_player_tile is not None
            and manhattan(player, self.last_reliable_player_tile) > 3
            and manhattan(center_tile, self.last_reliable_player_tile) <= 3
        ):
            player = center_tile

        self.last_reliable_player_tile = player
        if player == state.player and entity.tile == player:
            return state
        return replace(state, player=player, player_entity=replace(entity, tile=player))

    def _valid_player_tile(self, tile: Position, state: SymbolicState) -> bool:
        return (
            in_bounds(tile)
            and tile not in state.walls
            and tile not in state.traps
            and tile not in state.gaps
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

    def _remember_hub_context(self, features: RoomFeatures) -> None:
        if self.switch_hub_side is None and features.has_switch and len(features.exit_sides) == 1:
            exit_side = next(iter(features.exit_sides))
            self.switch_hub_side = OPPOSITE_SIDE[exit_side]

    def _is_known_switch_room(self, features: RoomFeatures) -> bool:
        if not features.has_switch or len(features.exit_sides) != 1:
            return False
        center_side = OPPOSITE_SIDE[next(iter(features.exit_sides))]
        return self.switch_hub_side is None or center_side == self.switch_hub_side

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

    def _open_nearest_chest(self, state: SymbolicState) -> int:
        chest_targets = set(state.chests) - self.remembered_blockers
        interact = self._interaction_if_adjacent(state, chest_targets, kind="chest")
        if interact is not None:
            self.pending_interaction = interact
            return interact.action_to_face
        return self._move_toward(state, adjacent_walkable_goals(state, chest_targets, self.remembered_blockers))

    def _return_to_center(self, state: SymbolicState, features: RoomFeatures) -> int:
        if features.exit_sides:
            side = min(features.exit_sides, key=lambda candidate: len(side_exits(state, SIDE_TO_ACTION[candidate])))
            return self._go_to_side_via_approach(state, side)
        return self._move_toward(state, task1_exit_goals(state))

    def _press_switch(self, state: SymbolicState) -> int:
        targets = set(state.switches)
        if not targets:
            return self._go_to_side_via_approach(state, "right")
        interact = self._interaction_if_adjacent(state, targets, kind="switch")
        if interact is not None:
            self.pending_interaction = interact
            return interact.action_to_face
        return self._move_toward(state, adjacent_walkable_goals(state, targets, self.remembered_blockers))

    def _act_task5(self, state: SymbolicState, features: RoomFeatures, keys: int) -> int:
        if self.shield_cooldown > 0:
            self.shield_cooldown -= 1

        room_sig = self._update_memory(state, features)

        if self.pending_interaction is not None:
            intent = self.pending_interaction
            self.pending_interaction = None
            if adjacent_action(state.player, intent.target) == intent.action_to_face:
                if intent.kind == "chest":
                    self.remembered_blockers.add(intent.target)
                    self.room_memory.setdefault(room_sig, RoomMemory()).opened_chests.add(intent.target)
                if intent.kind == "button":
                    self.room_memory.setdefault(room_sig, RoomMemory()).pressed_buttons.add(intent.target)
                return ACTION_A

        adjacent_monster = self._interaction_if_adjacent(state, set(state.monsters), kind="monster")
        if adjacent_monster is not None:
            chest_targets = set(state.chests) - self._opened_chests_for_current_room() - self.remembered_blockers
            if any(manhattan(state.player, chest) == 1 for chest in chest_targets):
                return ACTION_A
            shield_action = self._shield_if_close(state)
            if shield_action is not None:
                return shield_action
            self.pending_interaction = adjacent_monster
            return adjacent_monster.action_to_face

        shield_action = self._shield_if_close(state)
        if shield_action is not None:
            return shield_action

        objective = self._choose_objective(state, features, keys, room_sig)
        if objective.kind == "combat_chest":
            return self._combat_chest_action(state, set(objective.targets))
        if objective.kind == "open_chest":
            return self._open_targets(state, set(objective.targets), "chest")
        if objective.kind == "press_button":
            return self._open_targets(state, set(objective.targets), "button")
        if objective.kind == "go_exit" and objective.side is not None:
            return self._go_to_side_via_approach(state, objective.side)

        return self._move_toward(state, task1_exit_goals(state))

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
        memory.known_chests.update(state.chests)
        memory.has_monster = memory.has_monster or features.has_monster
        memory.has_trap = memory.has_trap or features.has_trap
        memory.visits += 1
        return room_sig

    def _structural_room_signature(self, state: SymbolicState, features: RoomFeatures) -> str:
        pieces = [
            "exits=" + ",".join(sorted(features.exit_sides)),
            "walls=" + ",".join(f"{x}:{y}" for x, y in sorted(state.walls)),
            "npc=" + ",".join(f"{x}:{y}" for x, y in sorted(npc_tiles(state))),
        ]
        return "|".join(pieces) if any(pieces) else "unknown"

    def _choose_objective(
        self,
        state: SymbolicState,
        features: RoomFeatures,
        keys: int,
        room_sig: str,
    ) -> Objective:
        memory = self.room_memory.setdefault(room_sig, RoomMemory())
        chest_targets = frozenset(set(state.chests) - memory.opened_chests - self.remembered_blockers)
        if len(features.exit_sides) <= 1:
            if chest_targets and state.monsters:
                return Objective("combat_chest", chest_targets)
            if chest_targets:
                return Objective("open_chest", chest_targets)
        else:
            if chest_targets:
                if state.monsters:
                    return Objective("combat_chest", chest_targets)
                return Objective("open_chest", chest_targets)

        button_targets = frozenset(set(state.buttons) - memory.pressed_buttons)
        if button_targets:
            return Objective("press_button", button_targets)

        if len(features.exit_sides) == 1:
            return Objective("go_exit", side=next(iter(features.exit_sides)))

        side = self._choose_exploration_side(state, features, keys, room_sig)
        if side is not None:
            return Objective("go_exit", side=side)
        return Objective("idle")

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
                self.config,
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
        return bool(room.known_chests - room.opened_chests)

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

    def _combat_chest_action(self, state: SymbolicState, chest_targets: set[Position]) -> int:
        for chest in sorted(chest_targets, key=lambda pos: manhattan(state.player, pos)):
            if manhattan(state.player, chest) == 1:
                self.pending_interaction = None
                if self.current_room_signature is not None:
                    self.room_memory.setdefault(self.current_room_signature, RoomMemory()).opened_chests.add(chest)
                self.remembered_blockers.add(chest)
                return ACTION_A

        adjacent_monster = self._interaction_if_adjacent(state, set(state.monsters), kind="monster")
        if adjacent_monster is not None:
            shield_action = self._shield_if_close(state)
            if shield_action is not None:
                return shield_action
            self.pending_interaction = adjacent_monster
            return adjacent_monster.action_to_face

        nearby_monster_distance = min((manhattan(state.player, monster) for monster in state.monsters), default=99)
        if nearby_monster_distance <= self.config.shield_distance:
            shield_action = self._shield_if_close(state)
            if shield_action is not None:
                return shield_action

        chest_goals = adjacent_walkable_goals(state, chest_targets, self._current_room_blockers())
        aligned_goals = {
            goal
            for goal in chest_goals
            if goal[0] == state.player[0] or goal[1] == state.player[1]
        }
        path = astar_path(state, aligned_goals or chest_goals, self._current_room_blockers(), None, self.config)
        if path is not None and len(path) >= 2:
            target_tile = path[1] if path[1] not in state.exits else None
            return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS, target_tile=target_tile)

        monster_targets = monster_goals(state, self._current_room_blockers())
        path_to_monster = astar_path(state, monster_targets, self._current_room_blockers(), None, self.config)
        if path_to_monster is not None and len(path_to_monster) >= 2:
            target_tile = path_to_monster[1] if path_to_monster[1] not in state.exits else None
            return self._commit(action_from_step(path_to_monster[0], path_to_monster[1]), TILE_MOVE_TICKS, target_tile=target_tile)
        return ACTION_NOOP

    def _open_targets(self, state: SymbolicState, targets: set[Position], kind: str) -> int:
        if kind == "button":
            if state.player in targets and self.current_room_signature is not None:
                self.room_memory.setdefault(self.current_room_signature, RoomMemory()).pressed_buttons.add(state.player)
                return ACTION_NOOP
            path = astar_path(state, targets, self._current_room_blockers(), None, self.config)
            if path is None or len(path) < 2:
                return ACTION_NOOP
            if path[1] in targets and self.current_room_signature is not None:
                self.room_memory.setdefault(self.current_room_signature, RoomMemory()).pressed_buttons.add(path[1])
            return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS, target_tile=path[1])
        interact = self._interaction_if_adjacent(state, targets, kind=kind)
        if interact is not None:
            self.pending_interaction = interact
            return interact.action_to_face
        return self._move_toward_astar(state, adjacent_walkable_goals(state, targets, self._current_room_blockers()))

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

    def _go_to_side_via_approach(self, state: SymbolicState, side: Side) -> int:
        direction = SIDE_TO_ACTION[side]
        exits = side_exits(state, direction)
        if state.player in exits:
            if self.current_room_signature is not None:
                self.attempted_exit = (self.current_room_signature, side)
            return self._commit(direction, EXIT_MOVE_TICKS)
        non_target_exits = set(state.exits) - exits
        extra_blocked = self._current_room_blockers() | non_target_exits
        approach = self._side_approach_goals(state, side, extra_blocked)

        if state.player in approach:
            if self.current_room_signature is not None:
                self.attempted_exit = (self.current_room_signature, side)
            dx, dy = ACTION_TO_DELTA[direction]
            target = (state.player[0] + dx, state.player[1] + dy)
            target_tile = target if in_bounds(target) and target not in state.exits else None
            return self._commit(direction, TILE_MOVE_TICKS, target_tile=target_tile)
        preferred_approach = self._nearest_side_approach_goals(state, side, approach)
        path = astar_path_to_side(state, preferred_approach, side, extra_blocked, None, self.config)
        if path is None and preferred_approach != approach:
            path = astar_path_to_side(state, approach, side, extra_blocked, None, self.config)
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
            return {(x, 1) for x, _ in exits if is_walkable((x, 1), state, extra_blocked, extra_unblocked)}
        if side == "down":
            return {(x, 6) for x, _ in exits if is_walkable((x, 6), state, extra_blocked, extra_unblocked)}
        if side == "left":
            return {(1, y) for _, y in exits if is_walkable((1, y), state, extra_blocked, extra_unblocked)}
        return {(8, y) for _, y in exits if is_walkable((8, y), state, extra_blocked, extra_unblocked)}

    def _interaction_if_adjacent(
        self,
        state: SymbolicState,
        targets: set[Position],
        *,
        kind: str,
    ) -> InteractionIntent | None:
        if not targets:
            return None
        target = min(targets, key=lambda pos: manhattan(state.player, pos))
        face_action = adjacent_action(state.player, target)
        if face_action is None:
            return None
        return InteractionIntent(target=target, action_to_face=face_action, kind=kind)

    def _move_toward(
        self,
        state: SymbolicState,
        goals: set[Position],
        extra_blocked: set[Position] | None = None,
    ) -> int:
        path = bfs_path(state, goals, self.remembered_blockers if extra_blocked is None else extra_blocked)
        if path is None or len(path) < 2:
            return ACTION_NOOP
        target_tile = path[1] if path[1] not in state.exits else None
        return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS, target_tile=target_tile)

    def _move_toward_astar(
        self,
        state: SymbolicState,
        goals: set[Position],
        extra_blocked: set[Position] | None = None,
    ) -> int:
        path = astar_path(
            state,
            goals,
            self._current_room_blockers() if extra_blocked is None else extra_blocked,
            None,
            self.config,
        )
        if path is None or len(path) < 2:
            return ACTION_NOOP
        target_tile = path[1] if path[1] not in state.exits else None
        return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS, target_tile=target_tile)

    def _stuck_recovery_action(self, state: SymbolicState) -> bool:
        self._update_stuck_counter(state)
        if self.stuck_ticks < self.config.stuck_recovery_ticks or self.committed_action not in ACTION_TO_DELTA:
            return False
        dx, dy = ACTION_TO_DELTA[self.committed_action]
        blocker = (state.player[0] + dx, state.player[1] + dy)
        if in_bounds(blocker):
            self.remembered_blockers.add(blocker)
        self._clear_commit()
        self.stuck_ticks = 0
        return True

    def _update_stuck_counter(self, state: SymbolicState) -> None:
        if state.player == self.last_player_tile:
            self.stuck_ticks += 1
        else:
            self.last_player_tile = state.player
            self.stuck_ticks = 0

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

    def _commit(self, action: int | None, ticks: int, target_tile: Position | None = None) -> int:
        if action is None:
            return ACTION_NOOP
        self.committed_action = action
        self.committed_ticks_remaining = max(0, ticks - 1)
        self.committed_target_tile = target_tile
        self.committed_target_seen_ticks = 0
        return action


def make_policy() -> Policy:
    return Policy()
