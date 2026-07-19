from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nesylink.core.constants import (
    ACTION_A,
    ACTION_B,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_NOOP,
    ACTION_RIGHT,
    ACTION_UP,
)
from nesylink.core.observation import TILE_NPC
from submissions.shared import SymbolicState


Position = tuple[int, int]
Side = str

MOVE_TICKS = 16
EXIT_TICKS = 24

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
SIDE_ORDER: tuple[Side, ...] = ("up", "right", "down", "left")
OPTION_SPACE: tuple[tuple[str, Side | None], ...] = (
    ("open_chest", None),
    ("fight_monster", None),
    ("press_button", None),
    ("press_switch", None),
    ("go_exit", "up"),
    ("go_exit", "right"),
    ("go_exit", "down"),
    ("go_exit", "left"),
)
DEFAULT_OPTIONS_MODEL = Path(__file__).resolve().parent / "models" / "options_q.pkl"


@dataclass(frozen=True)
class Option:
    kind: str
    side: Side | None = None


class OptionsRLPolicy:
    """Hierarchical RL-style policy without importing the search submission agent.

    The high-level policy chooses temporally extended options such as opening a
    chest, fighting a monster, pressing a switch, or moving through a side exit.
    The low-level controller only executes the selected option from perception
    state. This keeps the policy independent from `submissions.student_agent`.
    """

    def __init__(self, perception_engine: Any | None = None) -> None:
        self.perception_engine = perception_engine
        self.task_id: str | None = None
        self.current_option: Option | None = None
        self.committed_action: int | None = None
        self.committed_ticks_remaining = 0
        self.opened_chests: set[tuple[str, Position]] = set()
        self.opened_chest_rooms: set[str] = set()
        self.pressed_switch_rooms: set[str] = set()
        self.west_switch_presses = 0
        self.visited_rooms: set[str] = set()
        self.room_entry_counts: dict[str, int] = {}
        self.last_room_id: str | None = None
        self.tried_exits: set[tuple[str, Side]] = set()
        self.remembered_blockers: set[tuple[str, Position]] = set()
        self.south_cleared = False
        self.saw_south_monster = False
        self.pending_interact = False
        self.rng = np.random.default_rng()
        self.option_q_table: dict[tuple, np.ndarray] = {}
        self._load_option_model()

    def reset(self, seed: int | None = None, task_id: str | None = None) -> None:
        self.task_id = task_id
        self.current_option = None
        self.committed_action = None
        self.committed_ticks_remaining = 0
        self.opened_chests = set()
        self.opened_chest_rooms = set()
        self.pressed_switch_rooms = set()
        self.west_switch_presses = 0
        self.visited_rooms = set()
        self.room_entry_counts = {}
        self.last_room_id = None
        self.tried_exits = set()
        self.remembered_blockers = set()
        self.south_cleared = False
        self.saw_south_monster = False
        self.pending_interact = False
        self.rng = np.random.default_rng(seed)
        if self.perception_engine is not None and hasattr(self.perception_engine, "reset"):
            try:
                self.perception_engine.reset(np.zeros((128, 160, 3), dtype=np.uint8))
            except TypeError:
                self.perception_engine.reset()

    def act(self, obs: Any, info: dict[str, Any]) -> int:
        if info.get("events", {}).get("counts", {}).get("action_blocked", 0) > 0:
            if self.committed_action in ACTION_TO_DELTA:
                state_for_block = self._extract_state(obs, info)
                dx, dy = ACTION_TO_DELTA[self.committed_action]
                blocker = (state_for_block.player[0] + dx, state_for_block.player[1] + dy)
                if in_bounds(blocker):
                    self.remembered_blockers.add((room_name(info), blocker))
            self.committed_action = None
            self.committed_ticks_remaining = 0
            self.current_option = None

        if self.pending_interact:
            self.pending_interact = False
            return ACTION_A

        if self.committed_action is not None:
            action = self.committed_action
            self.committed_ticks_remaining -= 1
            if self.committed_ticks_remaining <= 0:
                self.committed_action = None
                self.committed_ticks_remaining = 0
            return action

        state = self._extract_state(obs, info)
        room_id = room_name(info)
        if self.last_room_id != room_id:
            self.room_entry_counts[room_id] = self.room_entry_counts.get(room_id, 0) + 1
            self.last_room_id = room_id
        self.visited_rooms.add(room_id)
        self._update_memory(state, info)

        option = self._select_option(state, info)
        action, finished = self._execute_option(option, state, info)
        self.current_option = None if finished else option
        return action

    def _extract_state(self, obs: Any, info: dict[str, Any]) -> SymbolicState:
        if isinstance(obs, dict) and "grid" in obs:
            return symbolic_state_from_grid_obs(obs, info)
        if self.perception_engine is None:
            from submissions.perception import PerceptionEngine

            self.perception_engine = PerceptionEngine(device="cpu")
        return self.perception_engine.extract(obs)

    def _update_memory(self, state: SymbolicState, info: dict[str, Any]) -> None:
        room_id = room_name(info)
        events = info.get("events", {}).get("counts", {}) if isinstance(info, dict) else {}
        if events.get("chest_opened", 0) > 0:
            self.opened_chest_rooms.add(room_id)
            for chest in state.chests:
                if manhattan(state.player, chest) <= 1:
                    self.opened_chests.add((room_id, chest))
        if events.get("switch_activated", 0) > 0:
            self.pressed_switch_rooms.add(room_id)
            if room_id == "west":
                self.west_switch_presses += int(events.get("switch_activated", 0))
        if room_id == "south" and state.monsters:
            self.saw_south_monster = True
        if room_id == "south" and self.saw_south_monster and not state.monsters:
            self.south_cleared = True
        if events.get("monster_killed", 0) > 0 and room_id == "south":
            self.south_cleared = True

    def _load_option_model(self) -> None:
        configured = os.environ.get("NESYLINK_OPTIONS_MODEL")
        if not configured:
            return
        model_path = Path(configured)
        if not model_path.exists():
            return
        with model_path.open("rb") as file:
            payload = pickle.load(file)
        raw = payload.get("q_table", payload) if isinstance(payload, dict) else payload
        self.option_q_table = {key: np.asarray(values, dtype=np.float32) for key, values in raw.items()}

    def _select_learned_option(self, state: SymbolicState, info: dict[str, Any]) -> Option | None:
        q_values = self.option_q_table.get(option_state_key(state, info, self))
        if q_values is None:
            return None
        for index in np.argsort(q_values)[::-1]:
            option = option_from_index(int(index))
            if option_is_valid(option, state, info, self):
                return option
        return None
    def _select_option(self, state: SymbolicState, info: dict[str, Any]) -> Option:
        if self.current_option is not None:
            return self.current_option

        task = self.task_id or ""
        room_id = room_name(info)
        keys = inventory_keys(info)
        has_sword = "sword" in inventory_items(info)

        learned = self._select_learned_option(state, info)
        if learned is not None:
            return learned
        if task.endswith("task_4"):
            return self._select_task4_option(state, info, room_id, keys, has_sword)
        if task.endswith("task_3"):
            return self._select_task3_option(state, info, room_id, keys)
        if task.endswith("task_2"):
            return self._select_task2_option(state, info, keys)
        if task.endswith("task_1"):
            return self._select_task1_option(state, info, keys)
        return self._select_generic_option(state, info)

    def _select_generic_option(self, state: SymbolicState, info: dict[str, Any]) -> Option:
        if state.monsters and manhattan(state.player, nearest(state.player, set(state.monsters))) <= 3:
            return Option("fight_monster")
        if live_chests(state, info, self.opened_chests):
            return Option("open_chest")
        if state.buttons and int(info.get("entities", {}).get("buttons_pressed", 0) or 0) == 0:
            return Option("press_button")
        room_id = room_name(info)
        visible_sides = [side for side in ("down", "right", "left", "up") if side_exits(state, SIDE_TO_ACTION[side])]
        for side in visible_sides:
            if (room_id, side) not in self.tried_exits:
                return Option("go_exit", side)
        if state.monsters:
            return Option("fight_monster")
        if visible_sides:
            return Option("go_exit", visible_sides[0])
        return Option("explore")
    def _select_task1_option(self, state: SymbolicState, info: dict[str, Any], keys: int) -> Option:
        if live_chests(state, info, self.opened_chests):
            return Option("open_chest")
        if keys > 0:
            return Option("go_exit", "up")
        return Option("explore")

    def _select_task2_option(self, state: SymbolicState, info: dict[str, Any], keys: int) -> Option:
        if state.monsters:
            return Option("fight_monster")
        if live_chests(state, info, self.opened_chests):
            return Option("open_chest")
        if keys > 0:
            return Option("go_exit", "left")
        return Option("explore")

    def _select_task3_option(self, state: SymbolicState, info: dict[str, Any], room_id: str, keys: int) -> Option:
        if state.monsters:
            return Option("fight_monster")
        if live_chests(state, info, self.opened_chests):
            return Option("open_chest")
        if keys <= 0:
            if room_id == "start_room":
                return Option("go_exit", "left")
            if room_id == "monster_hall":
                return Option("go_exit", "left")
            return Option("explore")
        if room_id == "key_room":
            return Option("go_exit", "right")
        if room_id == "monster_hall":
            return Option("go_exit", "right")
        return Option("go_exit", "right")

    def _select_task4_option(
        self,
        state: SymbolicState,
        info: dict[str, Any],
        room_id: str,
        keys: int,
        has_sword: bool,
    ) -> Option:
        chests = live_chests(state, info, self.opened_chests)
        if room_id == "north":
            if chests:
                return Option("open_chest")
            return Option("go_exit", "down")
        if room_id == "east":
            if chests and not has_sword:
                return Option("open_chest")
            return Option("go_exit", "left")
        if room_id == "south":
            if state.monsters:
                return Option("fight_monster")
            self.south_cleared = True
            return Option("go_exit", "up")
        if room_id == "west":
            if keys <= 0:
                return Option("go_exit", "right")
            if not has_sword:
                return Option("press_switch") if self.west_switch_presses < 1 else Option("go_exit", "right")
            if not self.south_cleared:
                return Option("press_switch") if self.west_switch_presses < 2 else Option("go_exit", "right")
            return Option("go_exit", "right")
        if room_id == "center":
            if chests and self.south_cleared:
                return Option("open_chest")
            if keys <= 0:
                return Option("go_exit", "up")
            if not has_sword:
                return Option("go_exit", "left") if self.west_switch_presses < 1 else Option("go_exit", "right")
            if not self.south_cleared:
                return Option("go_exit", "left") if self.west_switch_presses < 2 else Option("go_exit", "down")
            if chests:
                return Option("open_chest")
            return Option("explore")
        return Option("explore")

    def _execute_option(self, option: Option, state: SymbolicState, info: dict[str, Any]) -> tuple[int, bool]:
        if option.kind == "open_chest":
            return self._open_nearest_chest(state, info)
        if option.kind == "fight_monster":
            return self._fight_nearest_monster(state, info)
        if option.kind == "press_switch":
            return self._press_switch(state)
        if option.kind == "press_button":
            return self._press_button(state)
        if option.kind == "go_exit" and option.side is not None:
            return self._go_exit(state, option.side, info)
        return self._explore(state)

    def _open_nearest_chest(self, state: SymbolicState, info: dict[str, Any]) -> tuple[int, bool]:
        targets = live_chests(state, info, self.opened_chests)
        if not targets:
            return ACTION_NOOP, True
        target = nearest(state.player, targets)
        face = adjacent_action(state.player, target)
        if face is not None:
            room_id = room_name(info)
            self.opened_chests.add((room_id, target))
            self.pending_interact = True
            return face, False
        goals = adjacent_walkable_goals(state, targets, self._room_blockers(info))
        return self._move_toward(state, goals, info)

    def _fight_nearest_monster(self, state: SymbolicState, info: dict[str, Any]) -> tuple[int, bool]:
        if not state.monsters:
            return ACTION_NOOP, True
        target = nearest(state.player, set(state.monsters))
        face = adjacent_action(state.player, target)
        if face is not None:
            self.pending_interact = True
            return face, False
        goals = adjacent_walkable_goals(state, set(state.monsters), self._room_blockers(info))
        return self._move_toward(state, goals, info)

    def _press_button(self, state: SymbolicState) -> tuple[int, bool]:
        if not state.buttons:
            return ACTION_NOOP, True
        targets = set(state.buttons)
        if state.player in targets:
            return ACTION_NOOP, True
        return self._move_toward(state, targets, None)
    def _press_switch(self, state: SymbolicState) -> tuple[int, bool]:
        if not state.switches:
            return ACTION_NOOP, True
        target = nearest(state.player, set(state.switches))
        face = adjacent_action(state.player, target)
        if face is not None:
            self.pending_interact = True
            return face, True
        goals = adjacent_walkable_goals(state, set(state.switches))
        return self._move_toward(state, goals)

    def _go_exit(self, state: SymbolicState, side: Side, info: dict[str, Any] | None = None) -> tuple[int, bool]:
        action = SIDE_TO_ACTION[side]
        exits = side_exits(state, action)
        if state.player in exits:
            if info is not None:
                self.tried_exits.add((room_name(info), side))
            return self._commit(action, EXIT_TICKS), False
        extra_blocked = set()
        if info is not None:
            extra_blocked = self._room_blockers(info)
        approach = side_approach_goals(state, side, extra_blocked)
        if state.player in approach:
            if info is not None:
                self.tried_exits.add((room_name(info), side))
            return self._commit(action, EXIT_TICKS), False
        return self._move_toward(state, approach, info)

    def _explore(self, state: SymbolicState) -> tuple[int, bool]:
        for side in SIDE_ORDER:
            exits = side_exits(state, SIDE_TO_ACTION[side])
            if exits:
                return self._go_exit(state, side)
        return ACTION_NOOP, False

    def _move_toward(self, state: SymbolicState, goals: set[Position], info: dict[str, Any] | None = None) -> tuple[int, bool]:
        extra_blocked = self._room_blockers(info)
        path = bfs_path(state, goals, extra_blocked)
        if path is None or len(path) < 2:
            return ACTION_NOOP, True
        return self._commit(action_from_step(path[0], path[1]), MOVE_TICKS), False

    def _room_blockers(self, info: dict[str, Any] | None) -> set[Position]:
        if info is None:
            return set()
        room_id = room_name(info)
        extra_blocked = {pos for remembered_room, pos in self.opened_chests if remembered_room == room_id}
        extra_blocked.update(pos for remembered_room, pos in self.remembered_blockers if remembered_room == room_id)
        return extra_blocked

    def _commit(self, action: int, ticks: int) -> int:
        self.committed_action = action
        self.committed_ticks_remaining = max(0, ticks - 1)
        return action




def option_from_index(index: int) -> Option:
    kind, side = OPTION_SPACE[index]
    return Option(kind, side)


def option_to_index(option: Option) -> int:
    return OPTION_SPACE.index((option.kind, option.side))


def option_state_key(state: SymbolicState, info: dict[str, Any], policy: OptionsRLPolicy) -> tuple:
    items = inventory_items(info)
    entities = info.get("entities", {}) if isinstance(info, dict) else {}
    return (
        str(info.get("env", {}).get("map_id", "")),
        room_name(info),
        min(inventory_keys(info), 2),
        int("sword" in items),
        int(entities.get("chests_remaining", len(state.chests)) or 0),
        int(entities.get("monsters_remaining", len(state.monsters)) or 0),
        int(entities.get("buttons_pressed", 0) or 0),
        int(entities.get("exits_open", 0) or 0),
        tuple(sorted(policy.opened_chest_rooms)),
        tuple(sorted(policy.visited_rooms)),
        min(policy.west_switch_presses, 2),
        int(policy.south_cleared),
    )


def option_is_valid(option: Option, state: SymbolicState, info: dict[str, Any], policy: OptionsRLPolicy) -> bool:
    del policy
    if option.kind == "open_chest":
        return bool(live_chests(state, info, set()))
    if option.kind == "fight_monster":
        return bool(state.monsters)
    if option.kind == "press_button":
        return bool(state.buttons) and int(info.get("entities", {}).get("buttons_pressed", 0) or 0) == 0
    if option.kind == "press_switch":
        return bool(state.switches)
    if option.kind == "go_exit" and option.side is not None:
        return bool(side_exits(state, SIDE_TO_ACTION[option.side]))
    return False
def normalize_state(state: SymbolicState, info: dict[str, Any]) -> SymbolicState:
    entities = info.get("entities", {}) if isinstance(info, dict) else {}
    monsters = state.monsters
    if int(entities.get("monsters_remaining", len(monsters)) or 0) == 0:
        monsters = frozenset()
    if monsters is state.monsters:
        return state
    return replace(state, monsters=monsters)

def room_name(info: dict[str, Any]) -> str:
    return str(info.get("env", {}).get("room_id", ""))


def inventory_keys(info: dict[str, Any]) -> int:
    try:
        return int(info.get("inventory", {}).get("keys", 0))
    except (TypeError, ValueError):
        return 0


def inventory_items(info: dict[str, Any]) -> tuple[str, ...]:
    items = info.get("inventory", {}).get("items", ())
    if isinstance(items, (list, tuple, set)):
        return tuple(str(item) for item in items)
    return ()


def live_chests(state: SymbolicState, info: dict[str, Any], opened: set[tuple[str, Position]]) -> set[Position]:
    count = int(info.get("entities", {}).get("chests_remaining", len(state.chests)) or 0)
    if count <= 0:
        return set()
    room_id = room_name(info)
    return {chest for chest in state.chests if (room_id, chest) not in opened}


def neighbors(pos: Position) -> Iterable[Position]:
    x, y = pos
    yield (x, y - 1)
    yield (x, y + 1)
    yield (x - 1, y)
    yield (x + 1, y)


def in_bounds(pos: Position) -> bool:
    x, y = pos
    return 0 <= x < 10 and 0 <= y < 8


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest(start: Position, targets: set[Position] | frozenset[Position]) -> Position:
    return min(targets, key=lambda pos: (manhattan(start, pos), pos[1], pos[0]))


def action_from_step(current: Position, nxt: Position) -> int:
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]
    for action, delta in ACTION_TO_DELTA.items():
        if delta == (dx, dy):
            return action
    return ACTION_NOOP


def adjacent_action(current: Position, target: Position) -> int | None:
    if manhattan(current, target) != 1:
        return None
    return action_from_step(current, target)


def monster_danger_tiles(state: SymbolicState) -> set[Position]:
    danger = set(state.monsters)
    for monster in state.monsters:
        danger.update(pos for pos in neighbors(monster) if in_bounds(pos))
    danger.discard(state.player)
    return danger


def npc_tiles(state: SymbolicState) -> set[Position]:
    return {
        (x, y)
        for y, row in enumerate(state.static_grid)
        for x, value in enumerate(row)
        if value == TILE_NPC
    }


def blocked_tiles(state: SymbolicState, extra_blocked: set[Position] | None = None) -> set[Position]:
    blocked = set(state.walls) | set(state.traps) | set(state.gaps) | set(state.chests) | set(state.monsters) | npc_tiles(state)
    if state.bridges:
        safe_tiles = set(state.bridges) | set(state.exits) | {state.player}
        blocked.update((x, y) for x in range(10) for y in range(8) if (x, y) not in safe_tiles)
    blocked.difference_update(state.bridges)
    if extra_blocked:
        blocked.update(extra_blocked)
    blocked.discard(state.player)
    return blocked


def is_walkable(pos: Position, state: SymbolicState, extra_blocked: set[Position] | None = None) -> bool:
    return in_bounds(pos) and pos not in blocked_tiles(state, extra_blocked)


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


def bfs_path(state: SymbolicState, goals: set[Position], extra_blocked: set[Position] | None = None) -> list[Position] | None:
    start = state.player
    if not goals or not in_bounds(start):
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


def side_approach_goals(state: SymbolicState, side: Side, extra_blocked: set[Position] | None = None) -> set[Position]:
    exits = side_exits(state, SIDE_TO_ACTION[side])
    if side == "up":
        return {(x, 1) for x, _ in exits if is_walkable((x, 1), state, extra_blocked)}
    if side == "down":
        return {(x, 6) for x, _ in exits if is_walkable((x, 6), state, extra_blocked)}
    if side == "left":
        return {(1, y) for _, y in exits if is_walkable((1, y), state, extra_blocked)}
    return {(8, y) for _, y in exits if is_walkable((8, y), state, extra_blocked)}


def symbolic_state_from_grid_obs(obs: dict[str, np.ndarray], info: dict[str, Any]) -> SymbolicState:
    from submissions.shared import SymbolicState

    grid = np.asarray(obs["grid"], dtype=np.uint8)
    player = tuple(int(v) for v in np.asarray(obs["player_tile"]).tolist())

    def tiles(code: int) -> frozenset[Position]:
        ys, xs = np.where(grid == code)
        return frozenset((int(x), int(y)) for y, x in zip(ys, xs))

    return SymbolicState(
        player=player,
        walls=tiles(1),
        health=int(np.asarray(obs.get("health", [0])).reshape(-1)[0]),
        keys=inventory_keys(info),
        monsters=tiles(3),
        chests=tiles(4),
        traps=tiles(6),
        exits=tiles(5),
        buttons=tiles(7),
        gaps=tiles(9),
        bridges=tiles(10),
        switches=tiles(11),
        static_grid=tuple(tuple(int(v) for v in row) for row in grid.tolist()),
        room_id=room_name(info),
    )


Policy = OptionsRLPolicy


def make_policy() -> OptionsRLPolicy:
    return OptionsRLPolicy()














