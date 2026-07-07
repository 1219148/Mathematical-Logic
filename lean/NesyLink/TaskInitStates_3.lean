import NesyLink.Environment
open NesyLink

namespace NesyLink

-- ============================================================
-- Task 3：多房间（3 个房间），start_room → monster_hall → key_room
-- 路线：起点(4,4) → 西走到 monster_hall → 西走到 key_room 拿钥匙
--       原路返回 → start_room 东出口(lockedKey)通关
-- ============================================================

-- 起点房间
def task3_startLayout : RoomLayout := {
  walls               := [],
  spawns              := [("default", (4, 4)), ("from_west", (1, 4)), ("from_east", (8, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (0, 3), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "monster_hall", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 4), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "monster_hall", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (9, 3), exitType := ExitType.lockedKey, direction := ExitDirection.east,
      targetRoomId := "start_room", targetEntry := "from_east",
      requires := { keyCount := some 1, consumeKey := true },
      completeTask := true : ExitInfo },
    { pos := (9, 4), exitType := ExitType.lockedKey, direction := ExitDirection.east,
      targetRoomId := "start_room", targetEntry := "from_east",
      requires := { keyCount := some 1, consumeKey := true },
      completeTask := true : ExitInfo }
  ],
  initialChests       := [],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  roomId              := "start_room"
}

-- 怪物大厅
def task3_hallLayout : RoomLayout := {
  walls               := [],
  spawns              := [("default", (8, 4)), ("from_east", (8, 4)), ("from_west", (1, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (9, 3), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "start_room", targetEntry := "from_west",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (9, 4), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "start_room", targetEntry := "from_west",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 3), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "key_room", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (0, 4), exitType := ExitType.normal, direction := ExitDirection.west,
      targetRoomId := "key_room", targetEntry := "from_east",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [],
  initialMonsters     := [(5, 3)],
  initialMonsterTypes := [((5, 3), MonsterType.chaser)],
  initialButtons      := [],
  roomId              := "monster_hall"
}

-- 钥匙房
def task3_keyLayout : RoomLayout := {
  walls               := [],
  spawns              := [("default", (8, 4)), ("from_east", (8, 4))],
  defaultSpawn        := "default",
  traps               := [],
  exitInfos           := [
    { pos := (9, 3), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "monster_hall", targetEntry := "from_west",
      requires := {}, completeTask := false : ExitInfo },
    { pos := (9, 4), exitType := ExitType.normal, direction := ExitDirection.east,
      targetRoomId := "monster_hall", targetEntry := "from_west",
      requires := {}, completeTask := false : ExitInfo }
  ],
  initialChests       := [(5, 4)],
  initialMonsters     := [],
  initialMonsterTypes := [],
  initialButtons      := [],
  roomId              := "key_room"
}

-- 地牢映射：房间 ID → 房间布局
def task3_roomMap : List (String × RoomLayout) := [
  ("start_room",   task3_startLayout),
  ("monster_hall", task3_hallLayout),
  ("key_room",     task3_keyLayout)
]

-- 初始状态 = 起点房间
def task3_init : SymbolicState := {
  player       := (4, 4),
  health       := 5,
  keys         := 0,
  gold         := 0,
  items        := [],
  monsters     := [],
  monsterTypes := [],
  chests       := [],
  buttons      := [],
  room         := task3_startLayout
}

end NesyLink
