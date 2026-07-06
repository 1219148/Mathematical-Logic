from __future__ import annotations

from dataclasses import dataclass, field


TilePos = tuple[int, int]
PixelCenter = tuple[float, float]
PixelBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class EntityState:
    """动态实体的混合表示。

    tile 用于符号规划，center_px/bbox_px 用于像素级执行和碰撞近似。
    """

    tile: TilePos
    center_px: PixelCenter
    bbox_px: PixelBox
    kind: str
    entity_type: str = ""
    hp: int | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class SymbolicState:

    # 玩家位置
    player: TilePos

    # 墙壁位置集合
    walls: frozenset[TilePos]

    # 玩家生命值
    health: int # 玩家生命值

    # 玩家持有钥匙数
    keys: int 

    # 怪物位置集合
    monsters: frozenset[TilePos] = field(default_factory=frozenset) # 怪物位置集合

    # 怪物类型映射：坐标到名称
    monster_types: dict[TilePos, str] = field(default_factory=dict)

    # 未开启宝箱位置集合
    chests: frozenset[TilePos] = field(default_factory=frozenset)

    # 陷阱位置集合
    traps: frozenset[TilePos] = field(default_factory=frozenset)

    # 出口位置集合
    exits: frozenset[TilePos] = field(default_factory=frozenset)

    # 出口类型映射：坐标到类型
    exit_types: dict[TilePos, str] = field(default_factory=dict)
   
    # 玩家金币数
    gold: int = 0

    # 持有物品集合
    items: tuple[str, ...] = field(default_factory=tuple)

    # 按钮位置集合
    buttons: frozenset[TilePos] = field(default_factory=frozenset)
    
    #按钮是否按下：坐标到布尔
    button_pressed: dict[TilePos, bool] = field(default_factory=dict)

    # 缺口和桥是 Task 4/5 中重要的动态地形，仍然按 tile 表示。
    gaps: frozenset[TilePos] = field(default_factory=frozenset)
    bridges: frozenset[TilePos] = field(default_factory=frozenset)
    switches: frozenset[TilePos] = field(default_factory=frozenset)

    # 动态对象标记
    dynamic_objects: dict[str, str] = field(default_factory=dict)

    # 动态实体的像素级辅助信息。保留上面的 player/monsters 字段以兼容旧 planner。
    player_entity: EntityState | None = None
    monster_entities: tuple[EntityState, ...] = field(default_factory=tuple)

    # CNN 输出的完整 8x10 语义图，便于调试和后续 planner 复用。
    static_grid: tuple[tuple[int, ...], ...] = field(default_factory=tuple)

    # 房间标识
    room_id: str = ""
