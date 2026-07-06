from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SymbolicState:

    # 玩家位置
    player: tuple[int, int] 

    # 墙壁位置集合
    walls: frozenset[tuple[int, int]] 

    # 玩家生命值
    health: int # 玩家生命值

    # 玩家持有钥匙数
    keys: int 

    # 怪物位置集合
    monsters: frozenset[tuple[int, int]] = field(default_factory=frozenset) # 怪物位置集合

    # 怪物类型映射：坐标到名称
    monster_types: dict[tuple[int, int], str] = field(default_factory=dict)

    # 未开启宝箱位置集合
    chests: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    # 陷阱位置集合
    traps: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    # 出口位置集合
    exits: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    # 出口类型映射：坐标到类型
    exit_types: dict[tuple[int, int], str] = field(default_factory=dict)
   
    # 玩家金币数
    gold: int = 0

    # 持有物品集合
    items: tuple[str, ...] = field(default_factory=tuple)

    # 按钮位置集合
    buttons: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    
    #按钮是否按下：坐标到布尔
    button_pressed: dict[tuple[int, int], bool] = field(default_factory=dict)

    # 动态对象标记
    dynamic_objects: dict[str, str] = field(default_factory=dict)

    # 房间标识
    room_id: str = ""

