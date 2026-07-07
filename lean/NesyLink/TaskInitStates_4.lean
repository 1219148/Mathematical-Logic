import NesyLink.Environment
open NesyLink

namespace NesyLink

-- Task 4：多房间
-- 起点 west → center（旋转桥） → north（拿钥匙）
--            → center → east（拿剑，lockedKey 需钥匙）
--            → center → south（杀怪，揭示 center 隐藏宝箱）
--            → center → 隐藏宝箱（拿金币）

-- 西房（起点，开局带盾 + 桥开关）
def task4_westLayout : RoomLayout := {
  walls               := [
    (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),
    (0,1),(9,1),
    (0,2),(9,2),
    (0,5),(9,5),
    (0,6),(9,6),
    (0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7)
  ],
  spawns              := [("default", (7, 4)), ("east_door", (8, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (9, 3), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "center", targetEntry := "west_door",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (9, 4), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "center", targetEntry := "west_door",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  switches            := [
    { objectId := "bridge_switch", pos := (4, 4), activation := "interact",
      targetId := "center_bridge", effectType := "cycle_state",
      stateOrder := ["west_to_north", "west_to_east", "west_to_south"] : SwitchInfo }
  ],
  roomId              := "west"
}

-- 中央大厅（全屏深渊 + 旋转桥 + 隐藏宝箱）
def task4_centerLayout : RoomLayout := {
  walls               := [],
  spawns              := [
    ("default", (1, 4)), ("west_door", (1, 4)), ("east_door", (8, 4)),
    ("from_north", (4, 1)), ("from_south", (4, 6))
  ],
  defaultSpawn        := "default",
  traps               := [
    { pos := (0, 0), damage := 1, respawnTo := "default",
      trapType := TrapType.abyss, area := allRoomTiles : TrapInfo }
  ],
  exitInfos           := [
    { pos := (0, 3), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "west", targetEntry := "east_door",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 4), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "west", targetEntry := "east_door",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (9, 3), exitType := ExitType.lockedKey, direction := ExitDirection.east,
      targetRoomId := "east", targetEntry := "west_door",
      requires := { keyCount := some 1 }, completeTask := false : ExitInfo },
    { pos := (9, 4), exitType := ExitType.lockedKey, direction := ExitDirection.east,
      targetRoomId := "east", targetEntry := "west_door",
      requires := { keyCount := some 1 }, completeTask := false : ExitInfo },
    { pos := (4, 0), exitType := ExitType.normal, direction := ExitDirection.north,
      targetRoomId := "north", targetEntry := "from_south",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (5, 0), exitType := ExitType.normal, direction := ExitDirection.north,
      targetRoomId := "north", targetEntry := "from_south",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (4, 7), exitType := ExitType.normal, direction := ExitDirection.south,
      targetRoomId := "south", targetEntry := "from_north",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (5, 7), exitType := ExitType.normal, direction := ExitDirection.south,
      targetRoomId := "south", targetEntry := "from_north",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [],
  hiddenChests        := [(4, 4)],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  dynamicObjects      := [
    { objectId := "center_bridge", kind := "rotating_bridge",
      initialState := "west_to_north",
      states := [
        { stateId := "west_to_north",
          tiles := [
            (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),
            (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),
            (4,0),(5,0),(4,1),(5,1),(4,2),(5,2)
          ] : BridgeState },
        { stateId := "west_to_east",
          tiles := [
            (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),(7,3),(8,3),(9,3),
            (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),(7,4),(8,4),(9,4)
          ] : BridgeState },
        { stateId := "west_to_south",
          tiles := [
            (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),
            (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),
            (4,5),(5,5),(4,6),(5,6),(4,7),(5,7)
          ] : BridgeState }
      ]
    : DynamicObjectInfo }
  ],
  roomId              := "center"
}

-- 北房（钥匙宝箱）
def task4_northLayout : RoomLayout := {
  walls               := [
    (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),
    (0,1),(9,1),
    (0,2),(9,2),
    (0,3),(9,3),
    (0,4),(9,4),
    (0,5),(9,5),
    (0,6),(9,6),
    (0,7),(1,7),(2,7),(3,7),(6,7),(7,7),(8,7),(9,7)
  ],
  spawns              := [("default", (4, 6)), ("from_south", (4, 6))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (4, 7), exitType := ExitType.normal, direction := ExitDirection.south,
      targetRoomId := "center", targetEntry := "from_north",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (5, 7), exitType := ExitType.normal, direction := ExitDirection.south,
      targetRoomId := "center", targetEntry := "from_north",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [(4, 3)],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  roomId              := "north"
}

-- 东房（剑宝箱，需钥匙进入）
def task4_eastLayout : RoomLayout := {
  walls               := [
    (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),
    (0,1),(9,1),
    (0,2),(9,2),
    (0,5),(9,5),
    (0,6),(9,6),
    (0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7)
  ],
  spawns              := [("default", (1, 4)), ("west_door", (1, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (0, 3), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "center", targetEntry := "east_door",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 4), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "center", targetEntry := "east_door",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [(5, 4)],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  roomId              := "east"
}

-- 南房（守卫怪物，击杀后揭示 center 隐藏宝箱）
def task4_southLayout : RoomLayout := {
  walls               := [
    (0,0),(1,0),(2,0),(3,0),(6,0),(7,0),(8,0),(9,0),
    (0,1),(9,1),
    (0,2),(9,2),
    (0,3),(9,3),
    (0,4),(9,4),
    (0,5),(9,5),
    (0,6),(9,6),
    (0,7),(1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),(8,7),(9,7)
  ],
  spawns              := [("default", (4, 1)), ("from_north", (4, 1))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (4, 0), exitType := ExitType.normal, direction := ExitDirection.north,
      targetRoomId := "center", targetEntry := "from_south",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (5, 0), exitType := ExitType.normal, direction := ExitDirection.north,
      targetRoomId := "center", targetEntry := "from_south",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [],
  initialMonsters     := [(4, 4)],
  initialMonsterTypes := [((4, 4), MonsterType.chaser)],
  initialButtons      := [],
  roomId              := "south"
}

-- 映射
def task4_roomMap : List (String × RoomLayout) := [
  ("west",   task4_westLayout),
  ("center", task4_centerLayout),
  ("north",  task4_northLayout),
  ("east",   task4_eastLayout),
  ("south",  task4_southLayout)
]

-- 初始状态：西房开局，自带盾牌
def task4_init : SymbolicState := {
  player       := (7, 4),
  health       := 5,
  keys         := 0,
  gold         := 0,
  items        := ["shield"],
  monsters     := [],
  monsterTypes := [],
  chests       := [],
  hiddenChests := [],
  buttons      := [],
  dynamicStates := [("center_bridge", "west_to_north")],
  room         := task4_westLayout
}

end NesyLink
