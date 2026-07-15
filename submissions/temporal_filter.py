from __future__ import annotations

from dataclasses import dataclass, replace

from nesylink.shared import SymbolicState


Position = tuple[int, int]
TILE_SIZE = 16


@dataclass(frozen=True)
class TemporalFilterConfig:
    chest_ttl: int = 4
    switch_ttl: int = 64
    player_boundary_margin_px: float = 2.0


class TemporalSymbolicFilter:
    """Suppress short visual dropouts without inventing task-specific facts."""

    def __init__(self, config: TemporalFilterConfig | None = None) -> None:
        self.config = config or TemporalFilterConfig()
        self.reset()

    def reset(self) -> None:
        self._player: Position | None = None
        self._raw_player: Position | None = None
        self._walls: frozenset[Position] = frozenset()
        self._exits: frozenset[Position] = frozenset()
        self._chests: dict[Position, int] = {}
        self._switches: dict[Position, int] = {}

    def update(
        self,
        state: SymbolicState,
        *,
        inventory_changed: bool = False,
        suppressed_chests: set[Position] | None = None,
        trust_consistent_player: bool = False,
    ) -> SymbolicState:
        if self._room_changed(state):
            self.reset()

        player = self._stabilize_player(
            state,
            trust_consistent_player=trust_consistent_player,
        )
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
            chests=frozenset(chests),
            switches=frozenset(switches),
        )

    def _room_changed(self, state: SymbolicState) -> bool:
        if self._raw_player is None:
            return False
        teleported = _manhattan(self._raw_player, state.player) > 3
        structure_changed = self._walls != state.walls or self._exits != state.exits
        return teleported and structure_changed

    def _stabilize_player(
        self,
        state: SymbolicState,
        *,
        trust_consistent_player: bool,
    ) -> Position:
        entity = state.player_entity
        candidate = state.player
        tile_signals_agree = False
        if entity is not None:
            center_candidate = (
                int(entity.center_px[0] // TILE_SIZE),
                int(entity.center_px[1] // TILE_SIZE),
            )
            if self._valid_player(center_candidate, state):
                tile_signals_agree = state.player == center_candidate
                candidate = center_candidate
            elif not self._valid_player(candidate, state) and self._player is not None:
                candidate = self._player

        if entity is None and self._player is not None and not self._valid_player(candidate, state):
            return self._player
        if self._player is None or entity is None:
            self._player = candidate
            return candidate
        if trust_consistent_player and tile_signals_agree:
            self._player = candidate
            return candidate

        previous = self._player
        margin = self.config.player_boundary_margin_px
        center_x, center_y = entity.center_px
        x, y = candidate
        if x == previous[0] + 1 and center_x < x * TILE_SIZE + margin:
            x = previous[0]
        elif x == previous[0] - 1 and center_x > previous[0] * TILE_SIZE - margin:
            x = previous[0]
        if y == previous[1] + 1 and center_y < y * TILE_SIZE + margin:
            y = previous[1]
        elif y == previous[1] - 1 and center_y > previous[1] * TILE_SIZE - margin:
            y = previous[1]

        stabilized = (x, y)
        if not self._valid_player(stabilized, state):
            stabilized = candidate
        self._player = stabilized
        return stabilized

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
