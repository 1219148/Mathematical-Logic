from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace

from nesylink.core.observation import TILE_MONSTER
from submissions.shared import SymbolicState


Position = tuple[int, int]
TILE_SIZE = 16


@dataclass(frozen=True)
class TemporalFilterConfig:
    chest_ttl: int = 4
    switch_ttl: int = 64
    exit_type_window: int = 32
    player_disagreement_frames: int = 2
    monster_conflict_frames: int = 2


class TemporalSymbolicFilter:
    """Suppress short visual dropouts without inventing task-specific facts."""

    def __init__(self, config: TemporalFilterConfig | None = None) -> None:
        self.config = config or TemporalFilterConfig()
        self.reset()

    def reset(self) -> None:
        self._player: Position | None = None
        self._raw_player: Position | None = None
        self._center_player: Position | None = None
        self._center_disagreement_ticks = 0
        self._walls: frozenset[Position] = frozenset()
        self._exits: frozenset[Position] = frozenset()
        self._chests: dict[Position, int] = {}
        self._switches: dict[Position, int] = {}
        self._monster_conflicts: dict[Position, int] = {}
        self._exit_type_votes: dict[str, deque[str]] = {}
        self._stable_exit_types: dict[str, str] = {}

    def update(
        self,
        state: SymbolicState,
        *,
        inventory_changed: bool = False,
        suppressed_chests: set[Position] | None = None,
    ) -> SymbolicState:
        if self._room_changed(state):
            self.reset()

        player = self._stabilize_player(state)
        monsters = self._filter_monster_conflicts(state, player)
        suppressed = suppressed_chests or set()
        for position in suppressed:
            self._chests.pop(position, None)
        chests = self._update_static_tracks(
            self._chests,
            frozenset(set(state.chests) - suppressed),
            self.config.chest_ttl,
            clear_missing=inventory_changed,
        )
        switches = self._update_static_tracks(
            self._switches,
            state.switches,
            self.config.switch_ttl,
            clear_missing=False,
        )
        exit_types = self._stabilize_exit_types(state)
        self._raw_player = state.player
        self._walls = state.walls
        self._exits = state.exits
        entity = state.player_entity
        if entity is not None and entity.tile != player:
            entity = replace(entity, tile=player)
        return replace(
            state,
            player=player,
            player_entity=entity,
            monsters=frozenset(monsters),
            monster_types={
                position: kind
                for position, kind in state.monster_types.items()
                if position in monsters
            },
            monster_entities=tuple(
                monster_entity
                for monster_entity in state.monster_entities
                if monster_entity.tile in monsters
            ),
            chests=frozenset(chests),
            switches=frozenset(switches),
            exit_types=exit_types,
        )

    def _filter_monster_conflicts(
        self,
        state: SymbolicState,
        player: Position,
    ) -> set[Position]:
        accepted: set[Position] = set()
        current_conflicts: dict[Position, int] = {}
        for position in state.monsters:
            semantic_support = self._static_tile(state, position) == TILE_MONSTER
            if position != player or semantic_support:
                accepted.add(position)
                continue

            count = self._monster_conflicts.get(position, 0) + 1
            current_conflicts[position] = count
            if count >= self.config.monster_conflict_frames:
                accepted.add(position)

        self._monster_conflicts = current_conflicts
        return accepted

    @staticmethod
    def _static_tile(state: SymbolicState, position: Position) -> int | None:
        x, y = position
        if y < 0 or y >= len(state.static_grid):
            return None
        row = state.static_grid[y]
        if x < 0 or x >= len(row):
            return None
        return row[x]

    def _room_changed(self, state: SymbolicState) -> bool:
        if self._raw_player is None:
            return False
        teleported = _manhattan(self._raw_player, state.player) > 3
        structure_changed = self._walls != state.walls or self._exits != state.exits
        return teleported and structure_changed

    def _stabilize_player(self, state: SymbolicState) -> Position:
        entity = state.player_entity
        candidate = state.player
        center_candidate: Position | None = None
        if entity is not None:
            center_tile = (
                int(entity.center_px[0] // TILE_SIZE),
                int(entity.center_px[1] // TILE_SIZE),
            )
            if self._valid_player(center_tile, state):
                center_candidate = center_tile

        if not self._valid_player(candidate, state):
            if center_candidate is not None:
                candidate = center_candidate
            elif self._player is not None:
                return self._player

        if center_candidate is not None:
            center_is_smooth = (
                self._center_player is None
                or _manhattan(self._center_player, center_candidate) <= 1
            )
            disagrees = _manhattan(candidate, center_candidate) > 1
            if disagrees and center_is_smooth:
                self._center_disagreement_ticks += 1
            else:
                self._center_disagreement_ticks = 0
            self._center_player = center_candidate
            if self._center_disagreement_ticks >= self.config.player_disagreement_frames:
                candidate = center_candidate
        else:
            self._center_player = None
            self._center_disagreement_ticks = 0

        self._player = candidate
        return candidate

    def _stabilize_exit_types(self, state: SymbolicState) -> dict[Position, str]:
        valid_types = {"normal", "locked_key", "conditional"}
        for position in state.exits:
            side = self._exit_side(position)
            exit_type = state.exit_types.get(position)
            if side is None or exit_type not in valid_types:
                continue
            votes = self._exit_type_votes.setdefault(
                side,
                deque(maxlen=self.config.exit_type_window),
            )
            votes.append(exit_type)

        for side, votes in self._exit_type_votes.items():
            counts = Counter(votes)
            if not counts:
                continue
            best_count = max(counts.values())
            candidates = {name for name, count in counts.items() if count == best_count}
            previous = self._stable_exit_types.get(side)
            if previous in candidates:
                self._stable_exit_types[side] = previous
            elif len(candidates) == 1:
                self._stable_exit_types[side] = next(iter(candidates))

        return {
            position: self._stable_exit_types[side]
            for position in state.exits
            if (side := self._exit_side(position)) in self._stable_exit_types
        }

    @staticmethod
    def _exit_side(position: Position) -> str | None:
        x, y = position
        if x == 0:
            return "left"
        if x == 9:
            return "right"
        if y == 0:
            return "up"
        if y == 7:
            return "down"
        return None

    @staticmethod
    def _valid_player(tile: Position, state: SymbolicState) -> bool:
        x, y = tile
        return (
            0 <= x < 10
            and 0 <= y < 8
            and tile not in state.walls
            and tile not in state.traps
            and tile not in state.gaps
        )

    @staticmethod
    def _update_static_tracks(
        tracks: dict[Position, int],
        detected: frozenset[Position],
        ttl: int,
        *,
        clear_missing: bool,
    ) -> set[Position]:
        for position in list(tracks):
            if position in detected:
                continue
            if clear_missing or tracks[position] <= 1:
                del tracks[position]
            else:
                tracks[position] -= 1
        for position in detected:
            tracks[position] = ttl
        return set(tracks)


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
