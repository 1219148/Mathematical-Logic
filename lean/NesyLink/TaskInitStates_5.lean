import NesyLink.Environment
open NesyLink

namespace NesyLink

-- Task 5：多房间
-- 路线：room_0_0 → 按按钮(2,6) → 南到 room_0_1 拿钥匙
--            → 返回 → 东门(lockedKey) → room_1_0 宝箱(heal)
--            → 返回 → 西门 → room_-1_0 宝箱(gold) + 双怪物
-- 通关条件：打开所有宝箱

-- 起点房间（中央枢纽）
def task5_room00Layout : RoomLayout := {
  walls               := [
    (5,1), (5,2), (3,3), (4,3), (6,5)
  ],
  spawns              := [
    ("default", (1, 1)), ("from_east", (1, 4)),
    ("from_south", (4, 1)), ("from_west", (8, 4))
  ],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (9, 3), exitType := ExitType.lockedKey, direction := ExitDirection.east,
      targetRoomId := "room_1_0", targetEntry := "from_west",
      requires := { keyCount := some 1, consumeKey := true },
      completeTask := false : ExitInfo },
    { pos := (9, 4), exitType := ExitType.lockedKey, direction := ExitDirection.east,
      targetRoomId := "room_1_0", targetEntry := "from_west",
      requires := { keyCount := some 1, consumeKey := true },
      completeTask := false : ExitInfo },
    { pos := (0, 3), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "room_-1_0", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 4), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "room_-1_0", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (4, 7), exitType := ExitType.conditional, direction := ExitDirection.south,
      targetRoomId := "room_0_1", targetEntry := "from_north",
      requires := { buttonId := some "button_1" },
      completeTask := false : ExitInfo },
    { pos := (5, 7), exitType := ExitType.conditional, direction := ExitDirection.south,
      targetRoomId := "room_0_1", targetEntry := "from_north",
      requires := { buttonId := some "button_1" },
      completeTask := false : ExitInfo }
  ],
  initialChests       := [(4, 2)],
  initialMonsters     := [(7, 4)],
  initialMonsterTypes := [((7, 4), MonsterType.chaser)],
  initialButtons      := [("button_1", (2, 6))],
  npcs                := [(7, 6)],
  roomId              := "room_0_0"
}

-- 东房（宝箱 heal，ambusher，lockedKey 门后）
def task5_room10Layout : RoomLayout := {
  walls               := [
    (2,2), (2,3), (2,4), (5,4), (6,4)
  ],
  spawns              := [("default", (1, 4)), ("from_west", (1, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (0, 3), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "room_0_0", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 4), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "room_0_0", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [(7, 1)],
  initialMonsters     := [(7, 5)],
  initialMonsterTypes := [((7, 5), MonsterType.ambusher)],
  npcs                := [(7, 6)],
  roomId              := "room_1_0"
}

-- 南房（钥匙 + 陷阱 + patroller）
def task5_room01Layout : RoomLayout := {
  walls               := [
    (2,2),(3,2),(4,2),(5,2),(6,2),(7,2),
    (4,6)
  ],
  spawns              := [("default", (4, 1)), ("from_north", (4, 1))],
  defaultSpawn        := "default",
  traps               := [
    { pos := (1, 5), damage := 1, respawnTo := "from_north" : TrapInfo }
  ],
  exitInfos           := [
    { pos := (4, 0), exitType := ExitType.normal, direction := ExitDirection.north,
      targetRoomId := "room_0_0", targetEntry := "from_south",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (5, 0), exitType := ExitType.normal, direction := ExitDirection.north,
      targetRoomId := "room_0_0", targetEntry := "from_south",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [(8, 5)],
  initialMonsters     := [(6, 6)],
  initialMonsterTypes := [((6, 6), MonsterType.patroller)],
  npcs                := [(2, 1)],
  roomId              := "room_0_1"
}

-- 西房（金币 + chaser + ambusher）
def task5_roomMinus10Layout : RoomLayout := {
  walls               := [
    (1,2),(2,2),
    (5,5),
    (4,6),(5,6)
  ],
  spawns              := [("default", (8, 4)), ("from_east", (8, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (9, 3), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "room_0_0", targetEntry := "from_west",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (9, 4), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "room_0_0", targetEntry := "from_west",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [(2, 6)],
  initialMonsters     := [(2, 4), (6, 3)],
  initialMonsterTypes := [((2, 4), MonsterType.chaser), ((6, 3), MonsterType.ambusher)],
  npcs                := [(7, 6)],
  roomId              := "room_-1_0"
}

-- 映射
def task5_roomMap : List (String × RoomLayout) := [
  ("room_0_0",  task5_room00Layout),
  ("room_1_0",  task5_room10Layout),
  ("room_0_1",  task5_room01Layout),
  ("room_-1_0", task5_roomMinus10Layout)
]

-- 初始状态：room_0_0 起点(1,1)，场内一只 chaser 和一个宝箱
def task5_init : SymbolicState := {
  player       := (1, 1),
  facing       := Direction.down,
  health       := 5,
  keys         := 0,
  gold         := 0,
  items        := [],
  monsters     := [{ damage := 1, hp := 3, pos := (7, 4), monsterType := MonsterType.chaser : MonsterInfo }],
  monsterTypes := [((7, 4), MonsterType.chaser)],
  chests       := [{ pos := (4, 2), opened := false : ChestInfo }],
  buttons      := [],
  shieldTicks := 0,
  roomMap      := task5_roomMap,
  room         := task5_room00Layout
}

end NesyLink
