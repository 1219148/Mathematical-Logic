from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from nesylink.core.constants import (
    ACTION_A,
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
    kind: str


@dataclass(frozen=True)
class RoomFeatures:
    exit_sides: frozenset[Side]
    bridge_sides: frozenset[Side]
    has_bridge: bool
    has_switch: bool
    has_chest: bool
    has_monster: bool
    has_trap: bool


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


def blocked_tiles(state: SymbolicState, extra_blocked: set[Position] | None = None) -> set[Position]:
    blocked = set(state.walls) | set(state.traps) | set(state.gaps) | set(state.chests) | set(state.monsters)
    if state.bridges:
        safe_tiles = set(state.bridges) | set(state.exits) | {state.player}
        blocked.update(
            (x, y)
            for x in range(10)
            for y in range(8)
            if (x, y) not in safe_tiles
        )
    if extra_blocked:
        blocked.update(extra_blocked)
    return blocked


def is_walkable(pos: Position, state: SymbolicState, extra_blocked: set[Position] | None = None) -> bool:
    return in_bounds(pos) and pos not in blocked_tiles(state, extra_blocked)


def task1_chest_goals(state: SymbolicState) -> set[Position]:
    return adjacent_walkable_goals(state, set(state.chests))


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


def has_horizontal_dungeon_exits(state: SymbolicState) -> bool:
    has_left = bool(side_exits(state, ACTION_LEFT))
    has_right = bool(side_exits(state, ACTION_RIGHT))
    has_vertical = bool(side_exits(state, ACTION_UP) or side_exits(state, ACTION_DOWN))
    return has_left and has_right and not has_vertical


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


def inventory_keys(info: dict) -> int:
    inventory = info.get("inventory", {}) if isinstance(info, dict) else {}
    try:
        return int(inventory.get("keys", 0))
    except (TypeError, ValueError):
        return 0


def outward_exit_action(player: Position) -> int | None:
    x, y = player
    if y == 0:
        return ACTION_UP
    if y == 7:
        return ACTION_DOWN
    if x == 0:
        return ACTION_LEFT
    if x == 9:
        return ACTION_RIGHT
    return None


class Policy:
    def __init__(self, perception_engine=None) -> None:
        if perception_engine is None:
            from nesylink.perception import PerceptionEngine

            perception_engine = PerceptionEngine(device="cpu")
        self.perception_engine = perception_engine
        self.task_id: str | None = None
        self.pending_interaction: InteractionIntent | None = None
        self.committed_action: int | None = None
        self.committed_ticks_remaining = 0
        self.remembered_blockers: set[Position] = set()
        self.hub_exploration_started = False
        self.current_target_side: Side | None = None
        self.switch_hub_side: Side | None = None
        self.pressed_switch_for_target = False
        self.explored_hub_sides: set[Side] = set()
        self.saw_monster_objective = False
        self.monster_objective_done = False
        self.last_room_signature: str | None = None

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        del seed
        self.task_id = task_id
        self.pending_interaction = None
        self.committed_action = None
        self.committed_ticks_remaining = 0
        self.remembered_blockers = set()
        self.hub_exploration_started = False
        self.current_target_side = None
        self.switch_hub_side = None
        self.pressed_switch_for_target = False
        self.explored_hub_sides = set()
        self.saw_monster_objective = False
        self.monster_objective_done = False
        self.last_room_signature = None

    def act(self, obs, info) -> int:
        if self.committed_action is not None and self.committed_ticks_remaining > 0:
            if self.hub_exploration_started and self._symbolic_room_changed(obs):
                self.committed_action = None
                self.committed_ticks_remaining = 0
            else:
                self.committed_ticks_remaining -= 1
                action = self.committed_action
                if self.committed_ticks_remaining == 0:
                    self.committed_action = None
                return action

        if self.committed_action is not None and self.committed_ticks_remaining > 0:
            self.committed_ticks_remaining -= 1
            action = self.committed_action
            if self.committed_ticks_remaining == 0:
                self.committed_action = None
            return action

        state = self.perception_engine.extract(obs)
        features = room_features(state)
        keys = inventory_keys(info)
        items = inventory_items(info)

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

        if has_horizontal_dungeon_exits(state):
            desired_action = ACTION_LEFT if keys <= 0 else ACTION_RIGHT
            desired_exits = side_exits(state, desired_action)
            if state.player in desired_exits:
                return self._commit(desired_action, EXIT_MOVE_TICKS)
            return self._move_toward(state, desired_exits)

        if state.exits and state.player in state.exits:
            return self._commit(outward_exit_action(state.player), EXIT_MOVE_TICKS)

        return self._move_toward(state, task1_exit_goals(state))

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
                return self._go_to_side(state, target_side)
            switch_side = self.switch_hub_side or bridge_primary_side(state)
            self.pressed_switch_for_target = False
            if switch_side is None:
                return ACTION_NOOP
            return self._go_to_side(state, switch_side)

        if self._is_known_switch_room(features) and self.current_target_side is not None and not self.pressed_switch_for_target:
            return self._press_switch(state)

        self._mark_current_side_explored(features)
        return self._return_to_center(state, features)

    def _symbolic_room_changed(self, obs) -> bool:
        state = self.perception_engine.extract(obs)
        room = self._room_signature(room_features(state))
        return room != "unknown" and self.last_room_signature is not None and room != self.last_room_signature

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
        if features.has_chest:
            pieces.append("chest")
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
            return self._go_to_side(state, side)
        return self._move_toward(state, task1_exit_goals(state))

    def _press_switch(self, state: SymbolicState) -> int:
        targets = set(state.switches)
        if not targets:
            return self._go_to_exit(state, ACTION_RIGHT)
        interact = self._interaction_if_adjacent(state, targets, kind="switch")
        if interact is not None:
            self.pending_interaction = interact
            return interact.action_to_face
        return self._move_toward(state, adjacent_walkable_goals(state, targets, self.remembered_blockers))

    def _go_to_exit(self, state: SymbolicState, direction: int) -> int:
        exits = side_exits(state, direction)
        if state.player in exits:
            return self._commit(direction, EXIT_MOVE_TICKS)
        return self._move_toward(state, exits)

    def _go_to_side(self, state: SymbolicState, side: Side) -> int:
        return self._go_to_exit(state, SIDE_TO_ACTION[side])

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

    def _move_toward(self, state: SymbolicState, goals: set[Position]) -> int:
        path = bfs_path(state, goals, self.remembered_blockers)
        if path is None or len(path) < 2:
            return ACTION_NOOP
        return self._commit(action_from_step(path[0], path[1]), TILE_MOVE_TICKS)

    def _commit(self, action: int | None, ticks: int) -> int:
        if action is None:
            return ACTION_NOOP
        self.committed_action = action
        self.committed_ticks_remaining = max(0, ticks - 1)
        return action


def make_policy() -> Policy:
    return Policy()
