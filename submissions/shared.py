from __future__ import annotations

from dataclasses import dataclass, field


TilePos = tuple[int, int]
PixelCenter = tuple[float, float]
PixelBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class EntityState:
    """Pixel and tile representation shared by perception and planning."""

    tile: TilePos
    center_px: PixelCenter
    bbox_px: PixelBox
    kind: str
    entity_type: str = ""
    hp: int | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class SymbolicState:
    """Self-contained symbolic observation used by the submitted policy."""

    player: TilePos
    walls: frozenset[TilePos]
    health: int
    keys: int
    monsters: frozenset[TilePos] = field(default_factory=frozenset)
    monster_types: dict[TilePos, str] = field(default_factory=dict)
    chests: frozenset[TilePos] = field(default_factory=frozenset)
    traps: frozenset[TilePos] = field(default_factory=frozenset)
    exits: frozenset[TilePos] = field(default_factory=frozenset)
    exit_types: dict[TilePos, str] = field(default_factory=dict)
    gold: int = 0
    items: tuple[str, ...] = field(default_factory=tuple)
    buttons: frozenset[TilePos] = field(default_factory=frozenset)
    button_pressed: dict[TilePos, bool] = field(default_factory=dict)
    gaps: frozenset[TilePos] = field(default_factory=frozenset)
    bridges: frozenset[TilePos] = field(default_factory=frozenset)
    switches: frozenset[TilePos] = field(default_factory=frozenset)
    dynamic_objects: dict[str, str] = field(default_factory=dict)
    player_entity: EntityState | None = None
    monster_entities: tuple[EntityState, ...] = field(default_factory=tuple)
    static_grid: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    room_id: str = ""


__all__ = ["EntityState", "SymbolicState"]
